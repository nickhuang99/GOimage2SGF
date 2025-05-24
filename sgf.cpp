#include "common.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <regex> // Include the regex library
#include <set>
#include <vector>

using namespace std;
using namespace cv;

static void debugPrint(const Mat &before_board_state,
                       const Mat &next_board_state) {
  for (int row = 0; row < 19; ++row) {
    for (int col = 0; col < 19; ++col) {
      int before_stone = before_board_state.at<uchar>(row, col);
      int next_stone = next_board_state.at<uchar>(row, col);
      cout << before_stone << "(" << next_stone << ") ";
    }
    cout << endl;
  }
}

static void calculateSgfDiff(const Mat &before_board_state,
                             const Mat &next_board_state,
                             vector<Point> &black_diff_add,
                             vector<Point> &white_diff_add,
                             vector<Point> &black_diff_remove,
                             vector<Point> &white_diff_remove) {

  for (int row = 0; row < 19; ++row) {
    for (int col = 0; col < 19; ++col) {
      int before_stone = before_board_state.empty()
                             ? EMPTY
                             : before_board_state.at<uchar>(row, col);
      int next_stone = next_board_state.empty()
                           ? EMPTY
                           : next_board_state.at<uchar>(row, col);

      if (before_stone != next_stone) {
        if (next_stone == BLACK) {                   // Black stone added
          black_diff_add.push_back(Point(col, row)); // Store as (x, y)
        } else if (next_stone == WHITE) {            // White stone added
          white_diff_add.push_back(Point(col, row));
        } else if (before_stone == BLACK) { // Black stone removed
          black_diff_remove.push_back(Point(col, row));
        } else if (before_stone == WHITE) { // White stone removed
          white_diff_remove.push_back(Point(col, row));
        }
      }
    }
  }
}

static string generateSgfFromDiff(const vector<Point> &black_diff_add,
                                  const vector<Point> &white_diff_add,
                                  const vector<Point> &black_diff_remove,
                                  const vector<Point> &white_diff_remove) {
  string sgf_move = "";
  if (black_diff_add.size() == BLACK && white_diff_add.size() == EMPTY) {
    // Black played a stone.
    char sgf_col = 'a' + black_diff_add[0].x;
    char sgf_row = 'a' + black_diff_add[0].y;
    sgf_move = ";B[" + string(1, sgf_col) + string(1, sgf_row) + "]";
    //  Captures are optional
    for (Point p : white_diff_remove) {
      char sgf_col_remove = 'a' + p.x;
      char sgf_row_remove = 'a' + p.y;
      sgf_move +=
          "AE[" + string(1, sgf_col_remove) + string(1, sgf_row_remove) + "]";
    }

  } else if (white_diff_add.size() == BLACK && black_diff_add.size() == EMPTY) {
    // White played a stone.
    char sgf_col = 'a' + white_diff_add[0].x;
    char sgf_row = 'a' + white_diff_add[0].y;
    sgf_move = ";W[" + string(1, sgf_col) + string(1, sgf_row) + "]";
    // Captures are optional
    for (Point p : black_diff_remove) {
      char sgf_col_remove = 'a' + p.x;
      char sgf_row_remove = 'a' + p.y;
      sgf_move +=
          "AE[" + string(1, sgf_col_remove) + string(1, sgf_row_remove) + "]";
    }
  } else {
    cout << "impossible to have both black and white stones added at same time"
         << endl;
  }
  return sgf_move;
}

bool validateSGgfMove(const Mat &before_board_state,
                      const Mat &next_board_state, int prevColor) {
  vector<Point> black_diff_add;
  vector<Point> white_diff_add;
  vector<Point> black_diff_remove;
  vector<Point> white_diff_remove;
  calculateSgfDiff(before_board_state, next_board_state, black_diff_add,
                   white_diff_add, black_diff_remove, white_diff_remove);
  bool black_move_valid = black_diff_add.empty() &&
                          white_diff_add.size() == 1 &&
                          white_diff_remove.empty(); // no suidcide
  bool white_move_valid = black_diff_add.size() == 1 &&
                          white_diff_add.empty() &&
                          black_diff_remove.empty(); // no suidcide
  bool empty_move_valid =
      (!black_diff_add.empty() || !white_diff_add.empty()) &&
      (black_diff_remove.empty() && white_diff_remove.empty());

  bool valid = (prevColor == BLACK && black_move_valid) ||
               (prevColor == WHITE && white_move_valid) ||
               (prevColor == EMPTY && empty_move_valid);
  return valid;
}

// Function to determine the SGF move between two board states
string determineSGFMove(const Mat &before_board_state,
                        const Mat &next_board_state) {
  // 1. Find the differences between the two board states.
  vector<Point> black_diff_add;
  vector<Point> white_diff_add;
  vector<Point> black_diff_remove;
  vector<Point> white_diff_remove;
  calculateSgfDiff(before_board_state, next_board_state, black_diff_add,
                   white_diff_add, black_diff_remove, white_diff_remove);
  return generateSgfFromDiff(black_diff_add, white_diff_add, black_diff_remove,
                             white_diff_remove);
}

// Function to generate SGF for the current board state
string generateSGF(const Mat &board_state,
                   const vector<Point2f> &intersections) {
  ostringstream sgf;
  sgf << "(;FF[4]GM[1]SZ[19]AP[GoBoardAnalyzer:1.0]\n"; // SGF Header

  vector<Point> black_stones;
  vector<Point> white_stones;

  if (bDebug) {
    for (int row = 0; row < 19; ++row) {
      for (int col = 0; col < 19; ++col) {
        cout << static_cast<int>(board_state.at<uchar>(row, col)) << ",";
      }
      cout << endl;
    }
  }
  for (int row = 0; row < 19; ++row) {
    for (int col = 0; col < 19; ++col) {
      int stone = board_state.at<uchar>(row, col);
      if (stone == 1) { // Black stone
        black_stones.push_back(Point(col, row));
      } else if (stone == 2) { // White stone
        white_stones.push_back(Point(col, row));
      }
      // 0 (empty) is skipped
    }
  }
  // Add black stones using AB property
  if (!black_stones.empty()) {
    sgf << ";AB";
    for (const auto &stone : black_stones) {
      char sgf_col = 'a' + stone.x;
      char sgf_row = 'a' + stone.y;
      sgf << "[" << sgf_col << sgf_row << "]";
    }
  }

  // Add white stones using AW property
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

// Function to parse SGF header from a string
SGFHeader parseSGFHeader(const string &sgf_content) {
  SGFHeader header = {0}; // Initialize all fields to default values.

  // Use a stringstream to parse the SGF content.
  stringstream ss(sgf_content);
  string token;

  // Helper function to extract property values
  auto extractValue = [&](const string &propertyName) -> string {
    size_t pos = sgf_content.find(propertyName);
    if (pos != string::npos) {
      size_t start = sgf_content.find('[', pos) + 1;
      size_t end = sgf_content.find(']', pos);
      if (end > start)
        return sgf_content.substr(start, end - start);
    }
    return "";
  };

  // Parse GM property
  string gm_value = extractValue("GM");
  if (!gm_value.empty()) {
    try {
      header.gm = stoi(gm_value);
    } catch (const invalid_argument &e) {
      cerr << "Invalid GM value: " << gm_value << endl;
      header.gm = 0; // set to default
    }
  }

  // Parse FF property
  string ff_value = extractValue("FF");
  if (!ff_value.empty()) {
    try {
      header.ff = stoi(ff_value);
    } catch (const invalid_argument &e) {
      cerr << "Invalid FF value: " << ff_value << endl;
      header.ff = 0;
    }
  }

  // Parse CA property
  header.ca = extractValue("CA");

  // Parse AP property
  header.ap = extractValue("AP");

  // Parse SZ property
  string sz_value = extractValue("SZ");
  if (!sz_value.empty()) {
    try {
      header.sz = stoi(sz_value);
    } catch (const invalid_argument &e) {
      cerr << "Invalid SZ value: " << sz_value << endl;
      header.sz = 0;
    }
  }
  return header;
}

// Function to parse SGF game moves, differentiating setup and moves, and
// handling captures
void parseSGFGame(const string &sgfContent, set<pair<int, int>> &setupBlack,
                  set<pair<int, int>> &setupWhite, vector<Move> &moves) {
  setupBlack.clear();
  setupWhite.clear();
  moves.clear();

  // Helper function to convert SGF coordinates (e.g., "ab") to row and column
  auto sgfCoordToRowCol = [](const string &coord) -> pair<int, int> {
    if (coord.size() != 2 || !islower(coord[0]) || !islower(coord[1])) {
      cout << "sgfCoordToRowCol: Invalid coordinate: " << coord << endl;
      return {-1, -1}; // Invalid coordinate
    }
    int col = coord[0] - 'a';
    int row = coord[1] - 'a';
    return {row, col};
  };

  // Regex for parsing coordinate pairs: [a-z][a-z]
  const regex coordRegex(R"(\[([a-z]{2})\])");

  // New game property regex:
  // Group 1: AB or AW (setup property type)
  // Group 2: Coordinates for AB/AW (e.g., "[ab][cd]")
  // Group 3: B or W (move player)
  // Group 4: Move coordinate for B/W (e.g., "[ab]")
  // Group 5: AE coordinates for captured stones (e.g., "[cd][ef]"), optional
  const regex gamePropertyRegex(
      R"(;(?:(AB|AW)((?:\[[a-z]{2}\])+)|(B|W)(\[[a-z]{2}\])(?:AE((?:\[[a-z]{2}\])+))?))");

  sregex_iterator it(sgfContent.begin(), sgfContent.end(), gamePropertyRegex);
  sregex_iterator end;

  for (; it != end; ++it) {
    smatch match = *it;

    string setup_prop_type = match[1].str();

    if (!setup_prop_type.empty()) { // Matched AB or AW (setup stones)
      string setup_coords_str = match[2].str();
      set<pair<int, int>> *current_setup_set = nullptr;
      if (setup_prop_type == "AB") {
        current_setup_set = &setupBlack;
      } else if (setup_prop_type == "AW") {
        current_setup_set = &setupWhite;
      }

      if (current_setup_set) {
        sregex_iterator coordIt(setup_coords_str.cbegin(),
                                setup_coords_str.cend(), coordRegex);
        sregex_iterator coordEnd;
        for (; coordIt != coordEnd; ++coordIt) {
          smatch coordMatch = *coordIt;
          string singleCoordSgf = coordMatch[1].str(); // e.g., "ab"
          pair<int, int> rc = sgfCoordToRowCol(singleCoordSgf);
          if (rc.first != -1 && rc.second != -1) {
            current_setup_set->insert(rc);
          }
        }
      }
    } else {
      string move_player_str = match[3].str();
      if (!move_player_str.empty()) { // Matched B or W (player move)
        string move_coord_str_raw = match[4].str(); // Raw, like "[ab]"
        string ae_coords_str = match[5].str(); // Raw, like "[cd][ef]" or empty

        Move current_move;
        current_move.player = (move_player_str == "B" ? BLACK : WHITE);

        // Process the main move coordinate (Group 4)
        sregex_iterator mainMoveIt(move_coord_str_raw.cbegin(),
                                   move_coord_str_raw.cend(), coordRegex);
        if (mainMoveIt != sregex_iterator()) {
          smatch mainMoveMatch = *mainMoveIt;
          string singleCoordSgf = mainMoveMatch[1].str(); // e.g., "ab"
          pair<int, int> rc = sgfCoordToRowCol(singleCoordSgf);
          if (rc.first != -1 && rc.second != -1) {
            current_move.row = rc.first;
            current_move.col = rc.second;
          } else {
            // cerr << "Invalid SGF move coordinate: " << singleCoordSgf <<
            // endl; // Optional
            continue; // Skip this malformed move
          }
        } else {
          // cerr << "Could not parse SGF move coordinate from: " <<
          // move_coord_str_raw << endl; // Optional
          continue; // Skip if main move coordinate is not parsable
        }

        // Process captured stones if AE part is present (Group 5)
        if (!ae_coords_str.empty()) {
          sregex_iterator capturedIt(ae_coords_str.cbegin(),
                                     ae_coords_str.cend(), coordRegex);
          sregex_iterator capturedEnd;
          for (; capturedIt != capturedEnd; ++capturedIt) {
            smatch capturedMatch = *capturedIt;
            string singleCapturedSgf = capturedMatch[1].str(); // e.g., "cd"
            pair<int, int> rc_captured = sgfCoordToRowCol(singleCapturedSgf);
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

// Function to visually verify SGF data on the original image
void verifySGF(const Mat &image, const string &sgf_data,
               const vector<Point2f> &intersections) {
  Mat verification_image = image.clone(); // Create a copy to draw on
  set<pair<int, int>> setupBlack;
  set<pair<int, int>> setupWhite;
  vector<Move> moves;

  parseSGFGame(sgf_data, setupBlack, setupWhite, moves);

  // Draw setup stones
  for (const auto &stone : setupBlack) {
    int row = stone.first;
    int col = stone.second;
    if (row >= 0 && row < 19 && col >= 0 && col < 19) {
      Point2f pt = intersections[row * 19 + col];
      circle(verification_image, pt, 10, Scalar(0, 0, 0), 2); // Black
      // cout << "Black stone at setup (" << col << ", " << row << ") - Pixel:
      // (" << pt.x << ", " << pt.y << ")" << endl;
    }
  }
  for (const auto &stone : setupWhite) {
    int row = stone.first;
    int col = stone.second;
    if (row >= 0 && row < 19 && col >= 0 && col < 19) {
      Point2f pt = intersections[row * 19 + col];
      circle(verification_image, pt, 10, Scalar(255, 255, 255), 2); // White
      // cout << "White stone at setup (" << col << ", " << row << ") - Pixel:
      // (" << pt.x << ", " << pt.y << ")" << endl;
    }
  }

  // Draw moves
  for (const auto &move : moves) {
    int row = move.row;
    int col = move.col;
    if (row >= 0 && row < 19 && col >= 0 && col < 19) {
      Point2f pt = intersections[row * 19 + col];
      if (move.player == 1) {
        circle(verification_image, pt, 10, Scalar(0, 0, 255), 2); // Black
        // cout << "Black stone at move (" << col << ", " << row << ") - Pixel:
        // (" << pt.x << ", " << pt.y << ")" << endl;
      } else if (move.player == 2) {
        circle(verification_image, pt, 10, Scalar(255, 0, 0), 2); // White
        // cout << "White stone at move (" << col << ", " << row << ") - Pixel:
        // (" << pt.x << ", " << pt.y << ")" << endl;
      } else if (move.player == 0) {
        putText(verification_image, "X", pt, FONT_HERSHEY_SIMPLEX, 0.7,
                Scalar(255, 255, 255), 2); // Mark with 'X'
        // cout << "Stone removed at (" << col << ", " << row << ") - Pixel: ("
        // << pt.x << ", " << pt.y << ")" << endl;
      }
      if (!move.capturedStones.empty()) {
        // cout << "  Captured: ";
        for (const auto &captured : move.capturedStones) {
          // cout << "(" << captured.second << ", " << captured.first << ") ";
        }
        // cout << endl;
      }
    }
  }

  imshow("SGF Verification", verification_image);
  waitKey(0);
}

// Function to compare two SGF strings for semantic equivalence
// (order-insensitive)
bool compareSGF(const string &sgf1, const string &sgf2) {
  // 1. Parse both SGF strings.
  set<pair<int, int>> setupBlack1, setupBlack2;
  set<pair<int, int>> setupWhite1, setupWhite2;
  vector<Move> moves1, moves2;
  SGFHeader header1 = parseSGFHeader(sgf1);
  SGFHeader header2 = parseSGFHeader(sgf2);

  parseSGFGame(sgf1, setupBlack1, setupWhite1, moves1);
  parseSGFGame(sgf2, setupBlack2, setupWhite2, moves2);

  // 2. Compare the headers, but relax the comparison for the AP property.
  bool headersMatch = (header1.gm == header2.gm) &&
                      (header1.ff == header2.ff) &&
                      //(header1.ca == header2.ca) &&  // Relaxed comparison
                      (header1.sz == header2.sz);

  if (bDebug) {
    cout << "Comparing SGF Headers:" << endl;
    cout << "  GM: " << header1.gm << " vs " << header2.gm << endl;
    cout << "  FF: " << header1.ff << " vs " << header2.ff << endl;
    cout << "  CA: " << header1.ca << " vs " << header2.ca << endl;
    cout << "  SZ: " << header1.sz << " vs " << header2.sz << endl;
    if (!headersMatch)
      cout << "Headers do not match" << endl;
  }

  // 3. Compare the setup stones.
  bool setupBlackMatch = (setupBlack1 == setupBlack2);
  if (bDebug) {
    cout << "Comparing Black Setup:" << endl;
    cout << "  Size1: " << setupBlack1.size()
         << ", Size2: " << setupBlack2.size() << endl;
    cout << "  Stones1: ";
    for (const auto &stone : setupBlack1) {
      cout << "(" << stone.first << ", " << stone.second << ") ";
    }
    cout << endl;
    cout << "  Stones2: ";
    for (const auto &stone : setupBlack2) {
      cout << "(" << stone.first << ", " << stone.second << ") ";
    }
    cout << endl;
    if (!setupBlackMatch)
      cout << "Black Setup Stones do not match" << endl;
  }

  bool setupWhiteMatch = (setupWhite1 == setupWhite2);
  if (bDebug) {
    cout << "Comparing White Setup:" << endl;
    cout << "  Size1: " << setupWhite1.size()
         << ", Size2: " << setupWhite2.size() << endl;
    cout << "  Stones1: ";
    for (const auto &stone : setupWhite1) {
      cout << "(" << stone.first << ", " << stone.second << ") ";
    }
    cout << endl;
    cout << "  Stones2: ";
    for (const auto &stone : setupWhite2) {
      cout << "(" << stone.first << ", " << stone.second << ") ";
    }
    cout << endl;
    if (!setupWhiteMatch)
      cout << "White Setup Stones do not match" << endl;
  }

  // 4. Compare moves, including captured stones (now sets).
  bool movesMatch = true;
  if (moves1.size() != moves2.size()) {
    movesMatch = false;
    if (bDebug) {
      cout << "Number of moves is different: " << moves1.size() << " vs "
           << moves2.size() << endl;
    }
  } else {
    for (size_t i = 0; i < moves1.size(); ++i) {
      const Move &m1 = moves1[i];
      const Move &m2 = moves2[i];
      if (m1.player != m2.player || m1.row != m2.row || m1.col != m2.col ||
          m1.capturedStones != m2.capturedStones) {
        movesMatch = false;
        if (bDebug) {
          cout << "Moves at index " << i << " are different:" << endl;
          cout << "  Player: " << m1.player << " vs " << m2.player << endl;
          cout << "  Row: " << m1.row << " vs " << m2.row << endl;
          cout << "  Col: " << m1.col << " vs " << m2.col << endl;
          cout << "  Captured Stones: " << endl;
          cout << "    Size1: " << m1.capturedStones.size()
               << ", Size2: " << m2.capturedStones.size() << endl;
          cout << "    Stones1: ";
          for (const auto &stone : m1.capturedStones) {
            cout << "(" << stone.first << ", " << stone.second << ") ";
          }
          cout << endl;
          cout << "    Stones2: ";
          for (const auto &stone : m2.capturedStones) {
            cout << "(" << m2.row << ", " << m2.col << ") "; // Corrected line
          }
          cout << endl;
        }
        break; // Exit the loop as soon as a difference is found.
      }
    }
  }
  bool result =
      headersMatch && setupBlackMatch && setupWhiteMatch && movesMatch;
  return result;
}

void testParseSGFGame() {
  string sgfContent = "(;GM[1]FF[4]SZ[19];AB[cb][gb][hb][ib][mb];AW[ma][jb][kb]"
                      "[nb][ob];B[ab];W[cd];AE[ef][fg])";
  set<pair<int, int>> setupBlack;
  set<pair<int, int>> setupWhite;
  vector<Move> moves;

  parseSGFGame(sgfContent, setupBlack, setupWhite, moves);

  cout << "Black Setup Stones:" << endl;
  for (const auto &stone : setupBlack) {
    cout << "(" << stone.first << ", " << stone.second << ") ";
  }
  cout << endl;

  cout << "White Setup Stones:" << endl;
  for (const auto &stone : setupWhite) {
    cout << "(" << stone.first << ", " << stone.second << ") ";
  }
  cout << endl;

  cout << "Moves:" << endl;
  for (const auto &move : moves) {
    cout << "Player: " << move.player << ", Row: " << move.row
         << ", Col: " << move.col;
    if (!move.capturedStones.empty()) {
      cout << "  Captured: ";
      for (const auto &captured : move.capturedStones) {
        cout << "(" << captured.first << ", " << captured.second << ") ";
      }
    }
    cout << endl;
  }
}
