#include "common.h" // Includes logger.h
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
// #include <iostream> // Replaced by logger.h for LOG_XXX and CONSOLE_XXX
#include <limits>
#include <map>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <regex> // Include the regex library
#include <set>
#include <vector>

// Using namespace std; // Avoid global using namespace std for better practice
// Using namespace cv;  // Avoid global using namespace cv for better practice

static void debugPrint(const cv::Mat &before_board_state,
                       const cv::Mat &next_board_state) {
  LOG_DEBUG << "Board state before vs next:" << std::endl;
  for (int row = 0; row < 19; ++row) {
    std::stringstream ss_row;
    for (int col = 0; col < 19; ++col) {
      int before_stone = before_board_state.at<uchar>(row, col);
      int next_stone = next_board_state.at<uchar>(row, col);
      ss_row << before_stone << "(" << next_stone << ") ";
    }
    LOG_DEBUG << ss_row.str() << std::endl;
  }
}

static void calculateSgfDiff(const cv::Mat &before_board_state,
                             const cv::Mat &next_board_state,
                             std::vector<cv::Point> &black_diff_add,
                             std::vector<cv::Point> &white_diff_add,
                             std::vector<cv::Point> &black_diff_remove,
                             std::vector<cv::Point> &white_diff_remove) {

  for (int row = 0; row < 19; ++row) {
    for (int col = 0; col < 19; ++col) {
      int before_stone = before_board_state.empty()
                             ? EMPTY
                             : before_board_state.at<uchar>(row, col);
      int next_stone = next_board_state.empty()
                           ? EMPTY
                           : next_board_state.at<uchar>(row, col);

      if (before_stone != next_stone) {
        if (next_stone == BLACK) {
          black_diff_add.push_back(cv::Point(col, row));
        } else if (next_stone == WHITE) {
          white_diff_add.push_back(cv::Point(col, row));
        } else if (before_stone == BLACK) {
          black_diff_remove.push_back(cv::Point(col, row));
        } else if (before_stone == WHITE) {
          white_diff_remove.push_back(cv::Point(col, row));
        }
      }
    }
  }
}

static std::string
generateSgfFromDiff(const std::vector<cv::Point> &black_diff_add,
                    const std::vector<cv::Point> &white_diff_add,
                    const std::vector<cv::Point> &black_diff_remove,
                    const std::vector<cv::Point> &white_diff_remove) {
  std::string sgf_move = "";
  if (black_diff_add.size() == 1 &&
      white_diff_add.empty()) { // Rule was BLACK macro (1) and EMPTY macro (0)
    char sgf_col = 'a' + black_diff_add[0].x;
    char sgf_row = 'a' + black_diff_add[0].y;
    sgf_move = ";B[" + std::string(1, sgf_col) + std::string(1, sgf_row) + "]";
    for (cv::Point p : white_diff_remove) {
      char sgf_col_remove = 'a' + p.x;
      char sgf_row_remove = 'a' + p.y;
      sgf_move += "AE[" + std::string(1, sgf_col_remove) +
                  std::string(1, sgf_row_remove) + "]";
    }
  } else if (white_diff_add.size() == 1 &&
             black_diff_add
                 .empty()) { // Rule was BLACK macro (1) and EMPTY macro (0)
    char sgf_col = 'a' + white_diff_add[0].x;
    char sgf_row = 'a' + white_diff_add[0].y;
    sgf_move = ";W[" + std::string(1, sgf_col) + std::string(1, sgf_row) + "]";
    for (cv::Point p : black_diff_remove) {
      char sgf_col_remove = 'a' + p.x;
      char sgf_row_remove = 'a' + p.y;
      sgf_move += "AE[" + std::string(1, sgf_col_remove) +
                  std::string(1, sgf_row_remove) + "]";
    }
  } else if (!black_diff_add.empty() ||
             !white_diff_add.empty()) { // Log only if there were additions
                                        // attempt but rules not met
    LOG_ERROR << "generateSgfFromDiff: Impossible move. Black additions: "
              << black_diff_add.size()
              << ", White additions: " << white_diff_add.size()
              << ". Both should not be >0 or one >1." << std::endl;
  }
  return sgf_move;
}

bool validateSGgfMove(const cv::Mat &before_board_state,
                      const cv::Mat &next_board_state, int prevColor) {
  std::vector<cv::Point> black_diff_add;
  std::vector<cv::Point> white_diff_add;
  std::vector<cv::Point> black_diff_remove;
  std::vector<cv::Point> white_diff_remove;
  calculateSgfDiff(before_board_state, next_board_state, black_diff_add,
                   white_diff_add, black_diff_remove, white_diff_remove);

  // Valid move means:
  // If Black just played (prevColor == BLACK), White is expected to play.
  //   - Exactly 1 white stone added (white_diff_add.size() == 1)
  //   - No black stones added (black_diff_add.empty())
  //   - No white stones removed by White (white_diff_remove.empty() - no
  //   suicide)
  //   - Black stones may be removed by White (black_diff_remove can be
  //   non-empty)
  bool white_played_validly = black_diff_add.empty() &&
                              white_diff_add.size() == 1 &&
                              white_diff_remove.empty();

  // If White just played (prevColor == WHITE), Black is expected to play.
  //   - Exactly 1 black stone added (black_diff_add.size() == 1)
  //   - No white stones added (white_diff_add.empty())
  //   - No black stones removed by Black (black_diff_remove.empty() - no
  //   suicide)
  //   - White stones may be removed by Black (white_diff_remove can be
  //   non-empty)
  bool black_played_validly = black_diff_add.size() == 1 &&
                              white_diff_add.empty() &&
                              black_diff_remove.empty();

  // For the first move of the game (prevColor == EMPTY)
  //   - Either exactly one black stone is added OR exactly one white stone is
  //   added.
  //   - No stones of the other color are added.
  //   - No stones are removed by the player making the first move.
  bool first_move_valid =
      ((black_diff_add.size() == 1 && white_diff_add.empty()) ||
       (white_diff_add.size() == 1 && black_diff_add.empty())) &&
      black_diff_remove.empty() && white_diff_remove.empty();

  bool valid = (prevColor == BLACK && white_played_validly) ||
               (prevColor == WHITE && black_played_validly) ||
               (prevColor == EMPTY && first_move_valid);

  if (!valid) {
    LOG_WARN << "validateSGgfMove: Move deemed invalid. PrevColor: "
             << prevColor << ", B_add: " << black_diff_add.size()
             << ", W_add: " << white_diff_add.size()
             << ", B_rem: " << black_diff_remove.size()
             << ", W_rem: " << white_diff_remove.size() << std::endl;
  } else {
    LOG_DEBUG << "validateSGgfMove: Move valid. PrevColor: " << prevColor
              << ", B_add: " << black_diff_add.size()
              << ", W_add: " << white_diff_add.size()
              << ", B_rem: " << black_diff_remove.size()
              << ", W_rem: " << white_diff_remove.size() << std::endl;
  }
  return valid;
}

std::string determineSGFMove(const cv::Mat &before_board_state,
                             const cv::Mat &next_board_state) {
  std::vector<cv::Point> black_diff_add;
  std::vector<cv::Point> white_diff_add;
  std::vector<cv::Point> black_diff_remove;
  std::vector<cv::Point> white_diff_remove;
  calculateSgfDiff(before_board_state, next_board_state, black_diff_add,
                   white_diff_add, black_diff_remove, white_diff_remove);
  return generateSgfFromDiff(black_diff_add, white_diff_add, black_diff_remove,
                             white_diff_remove);
}

std::string generateSGF(const cv::Mat &board_state,
                        const std::vector<cv::Point2f>
                            &intersections) { // intersections currently unused
  std::ostringstream sgf;
  sgf << "(;FF[4]GM[1]SZ[19]AP[GEM:GoBoardAnalyzer:1.0]\n";

  std::vector<cv::Point> black_stones;
  std::vector<cv::Point> white_stones;

  if (Logger::getGlobalLogLevel() >= LogLevel::DEBUG) {
    LOG_DEBUG << "generateSGF: Board state:" << std::endl;
    for (int row = 0; row < 19; ++row) {
      std::stringstream ss_row_debug;
      for (int col = 0; col < 19; ++col) {
        ss_row_debug << static_cast<int>(board_state.at<uchar>(row, col))
                     << ",";
      }
      LOG_DEBUG << "  " << ss_row_debug.str() << std::endl;
    }
  }
  for (int row = 0; row < 19; ++row) {
    for (int col = 0; col < 19; ++col) {
      int stone = board_state.at<uchar>(row, col);
      if (stone == BLACK) {
        black_stones.push_back(cv::Point(col, row));
      } else if (stone == WHITE) {
        white_stones.push_back(cv::Point(col, row));
      }
    }
  }

  if (!black_stones.empty()) {
    sgf << ";AB";
    for (const auto &stone : black_stones) {
      char sgf_col = 'a' + stone.x;
      char sgf_row = 'a' + stone.y;
      sgf << "[" << sgf_col << sgf_row << "]";
    }
  }

  if (!white_stones.empty()) {
    sgf << ";AW";
    for (const auto &stone : white_stones) {
      char sgf_col = 'a' + stone.x;
      char sgf_row = 'a' + stone.y;
      sgf << "[" << sgf_col << sgf_row << "]";
    }
  }
  sgf << ")\n";
  return sgf.str();
}

SGFHeader parseSGFHeader(const std::string &sgf_content) {
  SGFHeader header; // Default constructor initializes members

  auto extractValue = [&](const std::string &propertyName) -> std::string {
    std::regex prop_regex(propertyName + "\\[([^\\]]*)\\]");
    std::smatch match;
    if (std::regex_search(sgf_content, match, prop_regex) && match.size() > 1) {
      return match[1].str();
    }
    return "";
  };

  std::string gm_value = extractValue("GM");
  if (!gm_value.empty()) {
    try {
      header.gm = std::stoi(gm_value);
    } catch (const std::invalid_argument &e) {
      LOG_ERROR << "Invalid GM value in SGF header: " << gm_value
                << ". Error: " << e.what() << std::endl;
    } catch (const std::out_of_range &e) {
      LOG_ERROR << "GM value out of range in SGF header: " << gm_value
                << ". Error: " << e.what() << std::endl;
    }
  }

  std::string ff_value = extractValue("FF");
  if (!ff_value.empty()) {
    try {
      header.ff = std::stoi(ff_value);
    } catch (const std::invalid_argument &e) {
      LOG_ERROR << "Invalid FF value in SGF header: " << ff_value
                << ". Error: " << e.what() << std::endl;
    } catch (const std::out_of_range &e) {
      LOG_ERROR << "FF value out of range in SGF header: " << ff_value
                << ". Error: " << e.what() << std::endl;
    }
  }

  header.ca = extractValue("CA");
  header.ap = extractValue("AP");

  std::string sz_value = extractValue("SZ");
  if (!sz_value.empty()) {
    try {
      header.sz = std::stoi(sz_value);
    } catch (const std::invalid_argument &e) {
      LOG_ERROR << "Invalid SZ value in SGF header: " << sz_value
                << ". Error: " << e.what() << std::endl;
    } catch (const std::out_of_range &e) {
      LOG_ERROR << "SZ value out of range in SGF header: " << sz_value
                << ". Error: " << e.what() << std::endl;
    }
  }
  return header;
}

void parseSGFGame(const std::string &sgfContent,
                  std::set<std::pair<int, int>> &setupBlack,
                  std::set<std::pair<int, int>> &setupWhite,
                  std::vector<Move> &moves) {
  setupBlack.clear();
  setupWhite.clear();
  moves.clear();

  auto sgfCoordToRowCol = [](const std::string &coord) -> std::pair<int, int> {
    if (coord.size() != 2 || !islower(coord[0]) || !islower(coord[1])) {
      LOG_WARN << "sgfCoordToRowCol: Invalid coordinate format: " << coord
               << std::endl;
      return {-1, -1};
    }
    int col = coord[0] - 'a';
    int row = coord[1] - 'a';
    if (col < 0 || col > 18 || row < 0 || row > 18) {
      LOG_WARN << "sgfCoordToRowCol: Coordinate out of bounds (0-18): " << coord
               << " -> (" << row << "," << col << ")" << std::endl;
      return {-1, -1};
    }
    return {row, col};
  };

  const std::regex coordRegex(R"(\[([a-z]{2})\])");
  const std::regex gamePropertyRegex(
      R"(;(?:(AB|AW)((?:\[[a-z]{2}\])+)|(B|W)(\[[a-z]{2}\])(?:AE((?:\[[a-z]{2}\])+))?))");

  std::sregex_iterator it(sgfContent.begin(), sgfContent.end(),
                          gamePropertyRegex);
  std::sregex_iterator end;

  for (; it != end; ++it) {
    std::smatch match = *it;
    std::string setup_prop_type = match[1].str();

    if (!setup_prop_type.empty()) {
      std::string setup_coords_str = match[2].str();
      std::set<std::pair<int, int>> *current_setup_set = nullptr;
      if (setup_prop_type == "AB") {
        current_setup_set = &setupBlack;
      } else if (setup_prop_type == "AW") {
        current_setup_set = &setupWhite;
      }

      if (current_setup_set) {
        std::sregex_iterator coordIt(setup_coords_str.cbegin(),
                                     setup_coords_str.cend(), coordRegex);
        std::sregex_iterator coordEnd;
        for (; coordIt != coordEnd; ++coordIt) {
          std::smatch coordMatch = *coordIt;
          std::string singleCoordSgf = coordMatch[1].str();
          std::pair<int, int> rc = sgfCoordToRowCol(singleCoordSgf);
          if (rc.first != -1 && rc.second != -1) {
            current_setup_set->insert(rc);
          }
        }
      }
    } else {
      std::string move_player_str = match[3].str();
      if (!move_player_str.empty()) {
        std::string move_coord_str_raw = match[4].str();
        std::string ae_coords_str = match[5].str();

        Move current_move;
        current_move.player = (move_player_str == "B" ? BLACK : WHITE);

        std::sregex_iterator mainMoveIt(move_coord_str_raw.cbegin(),
                                        move_coord_str_raw.cend(), coordRegex);
        if (mainMoveIt != std::sregex_iterator()) {
          std::smatch mainMoveMatch = *mainMoveIt;
          std::string singleCoordSgf = mainMoveMatch[1].str();
          std::pair<int, int> rc = sgfCoordToRowCol(singleCoordSgf);
          if (rc.first != -1 && rc.second != -1) {
            current_move.row = rc.first;
            current_move.col = rc.second;
          } else {
            LOG_ERROR << "Invalid SGF move coordinate during parsing: "
                      << singleCoordSgf << std::endl;
            continue;
          }
        } else {
          LOG_ERROR << "Could not parse SGF move coordinate from: "
                    << move_coord_str_raw << std::endl;
          continue;
        }

        if (!ae_coords_str.empty()) {
          std::sregex_iterator capturedIt(ae_coords_str.cbegin(),
                                          ae_coords_str.cend(), coordRegex);
          std::sregex_iterator capturedEnd;
          for (; capturedIt != capturedEnd; ++capturedIt) {
            std::smatch capturedMatch = *capturedIt;
            std::string singleCapturedSgf = capturedMatch[1].str();
            std::pair<int, int> rc_captured =
                sgfCoordToRowCol(singleCapturedSgf);
            if (rc_captured.first != -1 && rc_captured.second != -1) {
              current_move.capturedStones.insert(rc_captured);
            }
          }
        }
        moves.push_back(current_move);
      }
    }
  }
}

void verifySGF(const cv::Mat &image, const std::string &sgf_data,
               const std::vector<cv::Point2f> &intersections) {
  LOG_INFO << "Verifying SGF data against image..." << std::endl;
  cv::Mat verification_image = image.clone();
  std::set<std::pair<int, int>> setupBlack;
  std::set<std::pair<int, int>> setupWhite;
  std::vector<Move> moves;

  parseSGFGame(sgf_data, setupBlack, setupWhite,
               moves); // parseSGFGame logs its own warnings/errors

  LOG_DEBUG << "Parsed SGF for verification: SetupBlack=" << setupBlack.size()
            << ", SetupWhite=" << setupWhite.size()
            << ", Moves=" << moves.size() << std::endl;

  for (const auto &stone : setupBlack) {
    int row = stone.first;
    int col = stone.second;
    if (row >= 0 && row < 19 && col >= 0 && col < 19 &&
        !intersections.empty()) {
      cv::Point2f pt = intersections[row * 19 + col];
      cv::circle(verification_image, pt, 10, cv::Scalar(0, 0, 0), 2);
      LOG_DEBUG << "Verify: Black setup stone at (" << col << ", " << row
                << ") - Pixel: (" << pt.x << ", " << pt.y << ")" << std::endl;
    }
  }
  for (const auto &stone : setupWhite) {
    int row = stone.first;
    int col = stone.second;
    if (row >= 0 && row < 19 && col >= 0 && col < 19 &&
        !intersections.empty()) {
      cv::Point2f pt = intersections[row * 19 + col];
      cv::circle(verification_image, pt, 10, cv::Scalar(255, 255, 255), 2);
      LOG_DEBUG << "Verify: White setup stone at (" << col << ", " << row
                << ") - Pixel: (" << pt.x << ", " << pt.y << ")" << std::endl;
    }
  }

  for (const auto &move : moves) {
    int row = move.row;
    int col = move.col;
    if (row >= 0 && row < 19 && col >= 0 && col < 19 &&
        !intersections.empty()) {
      cv::Point2f pt = intersections[row * 19 + col];
      if (move.player == BLACK) {
        cv::circle(verification_image, pt, 10, cv::Scalar(0, 0, 255),
                   2); // Black move - Red circle
        LOG_DEBUG << "Verify: Black move at (" << col << ", " << row
                  << ") - Pixel: (" << pt.x << ", " << pt.y << ")" << std::endl;
      } else if (move.player == WHITE) {
        cv::circle(verification_image, pt, 10, cv::Scalar(255, 0, 0),
                   2); // White move - Blue circle
        LOG_DEBUG << "Verify: White move at (" << col << ", " << row
                  << ") - Pixel: (" << pt.x << ", " << pt.y << ")" << std::endl;
      } else if (move.player ==
                 EMPTY) { // Assuming player 0 from parseSGFGame was for AE
        LOG_DEBUG << "Verify: AE property indicates removed stones. Number of "
                     "captured: "
                  << move.capturedStones.size() << std::endl;
        for (const auto &cap_stone : move.capturedStones) {
          if (cap_stone.first >= 0 && cap_stone.first < 19 &&
              cap_stone.second >= 0 && cap_stone.second < 19) {
            cv::Point2f cap_pt =
                intersections[cap_stone.first * 19 + cap_stone.second];
            cv::putText(verification_image, "X", cap_pt,
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0),
                        2); // Cyan X for captured
            LOG_DEBUG << "  Verify: Captured stone marked at ("
                      << cap_stone.second << ", " << cap_stone.first
                      << ") - Pixel: (" << cap_pt.x << ", " << cap_pt.y << ")"
                      << std::endl;
          }
        }
      }
      if (move.player != EMPTY &&
          !move.capturedStones
               .empty()) { // This implies AE was part of B or W node
        LOG_DEBUG << "  Verify: Move by player " << move.player
                  << " also captured " << move.capturedStones.size()
                  << " stones." << std::endl;
        for (const auto &captured : move.capturedStones) {
          LOG_DEBUG << "    Captured at (" << captured.second << ", "
                    << captured.first << ")" << std::endl;
          if (captured.first >= 0 && captured.first < 19 &&
              captured.second >= 0 && captured.second < 19) {
            cv::Point2f cap_pt_detail =
                intersections[captured.first * 19 + captured.second];
            // Mark differently or just log, as the main move circle is already
            // there
            cv::drawMarker(verification_image, cap_pt_detail,
                           cv::Scalar(0, 255, 255), cv::MARKER_DIAMOND, 8,
                           1); // Yellow diamond
          }
        }
      }
    }
  }

  cv::imshow("SGF Verification", verification_image);
  cv::waitKey(0);
  cv::destroyWindow("SGF Verification");
  LOG_INFO << "SGF Verification display finished." << std::endl;
}

bool compareSGF(const std::string &sgf1_content,
                const std::string &sgf2_content) {
  LOG_INFO << "Comparing SGF contents." << std::endl;
  std::set<std::pair<int, int>> setupBlack1, setupBlack2;
  std::set<std::pair<int, int>> setupWhite1, setupWhite2;
  std::vector<Move> moves1, moves2;
  SGFHeader header1 =
      parseSGFHeader(sgf1_content); // parseSGFHeader logs errors
  SGFHeader header2 = parseSGFHeader(sgf2_content);

  parseSGFGame(sgf1_content, setupBlack1, setupWhite1,
               moves1); // parseSGFGame logs errors
  parseSGFGame(sgf2_content, setupBlack2, setupWhite2, moves2);

  bool headersMatch = (header1.gm == header2.gm) &&
                      (header1.ff == header2.ff) && (header1.sz == header2.sz);
  // AP and CA can differ, often not critical for game state comparison

  LOG_DEBUG << "SGF Comparison - HeadersMatch: " << headersMatch << std::endl;
  if (Logger::getGlobalLogLevel() >= LogLevel::DEBUG && !headersMatch) {
    LOG_DEBUG << "  Header1: GM=" << header1.gm << " FF=" << header1.ff
              << " SZ=" << header1.sz << " AP=" << header1.ap
              << " CA=" << header1.ca << std::endl;
    LOG_DEBUG << "  Header2: GM=" << header2.gm << " FF=" << header2.ff
              << " SZ=" << header2.sz << " AP=" << header2.ap
              << " CA=" << header2.ca << std::endl;
  }

  bool setupBlackMatch = (setupBlack1 == setupBlack2);
  LOG_DEBUG << "SGF Comparison - SetupBlackMatch: " << setupBlackMatch
            << std::endl;
  if (Logger::getGlobalLogLevel() >= LogLevel::DEBUG &&
      !setupBlackMatch) { /* Log details */
  }

  bool setupWhiteMatch = (setupWhite1 == setupWhite2);
  LOG_DEBUG << "SGF Comparison - SetupWhiteMatch: " << setupWhiteMatch
            << std::endl;
  if (Logger::getGlobalLogLevel() >= LogLevel::DEBUG &&
      !setupWhiteMatch) { /* Log details */
  }

  bool movesMatch = (moves1 == moves2); // Uses Move::operator==
  LOG_DEBUG << "SGF Comparison - MovesMatch: " << movesMatch
            << " (Size1: " << moves1.size() << ", Size2: " << moves2.size()
            << ")" << std::endl;
  if (Logger::getGlobalLogLevel() >= LogLevel::DEBUG && !movesMatch) {
    size_t min_moves = std::min(moves1.size(), moves2.size());
    for (size_t i = 0; i < min_moves; ++i) {
      if (!(moves1[i] == moves2[i])) {
        LOG_DEBUG << "  Moves differ at index " << i << ":" << std::endl;
        LOG_DEBUG << "    SGF1 Move: P" << moves1[i].player << " ("
                  << moves1[i].row << "," << moves1[i].col
                  << ") Caps:" << moves1[i].capturedStones.size() << std::endl;
        LOG_DEBUG << "    SGF2 Move: P" << moves2[i].player << " ("
                  << moves2[i].row << "," << moves2[i].col
                  << ") Caps:" << moves2[i].capturedStones.size() << std::endl;
        break;
      }
    }
    if (moves1.size() != moves2.size())
      LOG_DEBUG << "  Move lists have different sizes." << std::endl;
  }

  return headersMatch && setupBlackMatch && setupWhiteMatch && movesMatch;
}

void testParseSGFGame() {
  // This function is for testing and direct console output is often desired.
  // Using CONSOLE_OUT for clarity that this is test output.
  // It could also be refactored to use LOG_INFO if test output should go to
  // file.
  std::string sgfContent =
      "(;GM[1]FF[4]SZ[19];AB[cb][gb][hb][ib][mb];AW[ma][jb][kb]"
      "[nb][ob];B[ab];W[cd];AE[ef][fg])";
  std::set<std::pair<int, int>> setupBlack;
  std::set<std::pair<int, int>> setupWhite;
  std::vector<Move> moves;

  CONSOLE_OUT << "--- testParseSGFGame Output ---" << std::endl;
  parseSGFGame(sgfContent, setupBlack, setupWhite,
               moves); // parseSGFGame logs its own warnings

  CONSOLE_OUT << "Black Setup Stones:" << std::endl;
  for (const auto &stone : setupBlack) {
    CONSOLE_OUT << "(" << stone.first << ", " << stone.second << ") ";
  }
  CONSOLE_OUT << std::endl;

  CONSOLE_OUT << "White Setup Stones:" << std::endl;
  for (const auto &stone : setupWhite) {
    CONSOLE_OUT << "(" << stone.first << ", " << stone.second << ") ";
  }
  CONSOLE_OUT << std::endl;

  CONSOLE_OUT << "Moves:" << std::endl;
  for (const auto &move : moves) {
    CONSOLE_OUT << "Player: " << move.player << ", Row: " << move.row
                << ", Col: " << move.col;
    if (!move.capturedStones.empty()) {
      CONSOLE_OUT << "  Captured: ";
      for (const auto &captured : move.capturedStones) {
        CONSOLE_OUT << "(" << captured.first << ", " << captured.second << ") ";
      }
    }
    CONSOLE_OUT << std::endl;
  }
  CONSOLE_OUT << "--- End testParseSGFGame Output ---" << std::endl;
}