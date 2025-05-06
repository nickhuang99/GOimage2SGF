#include <cmath>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <regex> // Include the regex library
#include <set>
#include "common.h"

using namespace std;
using namespace cv;

// Function to determine the SGF move between two board states
string determineSGFMove(const Mat &before_board_state,
                        const Mat &next_board_state) {
  // 1. Find the differences between the two board states.
  vector<Point> black_diff_add;
  vector<Point> white_diff_add;
  vector<Point> black_diff_remove;
  vector<Point> white_diff_remove;

  for (int row = 0; row < 19; ++row) {
    for (int col = 0; col < 19; ++col) {
      int before_stone = before_board_state.at<uchar>(row, col);
      int next_stone = next_board_state.at<uchar>(row, col);

      if (before_stone != next_stone) {
        if (next_stone == 1) {                       // Black stone added
          black_diff_add.push_back(Point(col, row)); // Store as (x, y)
        } else if (next_stone == 2) {                // White stone added
          white_diff_add.push_back(Point(col, row));
        } else if (before_stone == 1) { // Black stone removed
          black_diff_remove.push_back(Point(col, row));
        } else if (before_stone == 2) { // White stone removed
          white_diff_remove.push_back(Point(col, row));
        }
      }
    }
  }

  // 2. Analyze the differences to determine the move.
  string sgf_move = "";
  if (black_diff_add.size() == 1 && white_diff_add.size() == 0) {
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

  } else if (white_diff_add.size() == 1 && black_diff_add.size() == 0) {
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
    sgf_move = ";ERROR: Invalid move detected!"; // Handle as an error
  }

  return sgf_move;
}

// Function to generate SGF for the current board state
string generateSGF(const Mat &board_state,
                   const vector<Point2f> &intersections) {
  ostringstream sgf;
  sgf << "(;FF[4]GM[1]SZ[19]AP[GoBoardAnalyzer:1.0]\n"; // SGF Header

  vector<Point> black_stones;
  vector<Point> white_stones;

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

  // 1.  Iterate through the SGF content using a general regex for all
  // move/setup properties.
  const regex gamePropertyRegex(R"(;?(AB|AW|B|W|AE)((?:\[[a-z]{2}\])+))");
  sregex_iterator it(sgfContent.begin(), sgfContent.end(), gamePropertyRegex);
  sregex_iterator end;

  // Lambda function to process setup coordinates (AB, AW)
  auto processSetupCoordinates = [&](const string &propertyValue,
                                     set<pair<int, int>> &storage) {
    sregex_iterator coordIt(propertyValue.cbegin(), propertyValue.cend(),
                            coordRegex);
    sregex_iterator coordEnd;
    for (; coordIt != coordEnd; ++coordIt) {
      smatch coordMatch = *coordIt;
      string coordStr = coordMatch[1].str();
      pair<int, int> coord = sgfCoordToRowCol(coordStr);
      if (coord.first != -1 && coord.second != -1) {
        storage.insert(coord);
      }
    }
  };

  // Lambda function to process move coordinates (B, W, AE)
  auto processMoveCoordinates = [&](const string &propertyValue, int player) {
    sregex_iterator coordIt(propertyValue.cbegin(), propertyValue.cend(),
                            coordRegex);
    sregex_iterator coordEnd;
    for (; coordIt != coordEnd; ++coordIt) {
      smatch coordMatch = *coordIt;
      string coordStr = coordMatch[1].str();
      pair<int, int> coord = sgfCoordToRowCol(coordStr);
      if (coord.first != -1 && coord.second != -1) {
        Move move;
        move.player = player;
        move.row = coord.first;
        move.col = coord.second;
        if (player == 0) { // AE
          move.capturedStones.insert(coord);
        }
        moves.push_back(move);
      }
    }
  };

  for (; it != end; ++it) {
    smatch match = *it;
    string propertyName = match[1].str();  // "AB", "AW", "B", "W", or "AE"
    string propertyValue = match[2].str(); // e.g., "[ab][cd]" or "[ef]"

    if (propertyName == "AB") {
      processSetupCoordinates(propertyValue, setupBlack);
    } else if (propertyName == "AW") {
      processSetupCoordinates(propertyValue, setupWhite);
    } else if (propertyName == "B") {
      processMoveCoordinates(propertyValue, 1);
    } else if (propertyName == "W") {
      processMoveCoordinates(propertyValue, 2);
    } else if (propertyName == "AE") {
      processMoveCoordinates(propertyValue, 0);
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

// int run() {
//   // 1. Load the Go board image.
//   Mat image_bgr = imread("go_board.jpg"); // Hardcoded file name
//   if (image_bgr.empty()) {
//     cerr << "Error loading image\n";
//     return -1;
//   }

//   // 2. Process the image to get the board state.
//   Mat board_state;
//   Mat board_with_stones;
//   processGoBoard(image_bgr, board_state, board_with_stones);
//   pair<vector<double>, vector<double>> grid_lines =
//       detectUniformGrid(image_bgr);
//   vector<double> horizontal_lines = grid_lines.first;
//   vector<double> vertical_lines = grid_lines.second;
//   vector<Point2f> intersection_points =
//       findIntersections(horizontal_lines, vertical_lines);
//   // Mat previous_board_state = board_state.clone();

//   // 3. Generate the SGF for the current board state.
//   string generated_sgf = generateSGF(board_state, intersection_points);
//   cout << "Generated SGF:\n" << generated_sgf << endl;

//   // 4. Load the "correct" SGF from the file.
//   ifstream correct_sgf_file("ScreenToSGF.sgf.txt");
//   if (!correct_sgf_file.is_open()) {
//     cerr << "Error opening correct SGF file\n";
//     return -1;
//   }
//   stringstream correct_sgf_stream;
//   correct_sgf_stream << correct_sgf_file.rdbuf();
//   string correct_sgf = correct_sgf_stream.str();
//   correct_sgf_file.close();

//   cout << "\nCorrect SGF:\n" << correct_sgf << endl;

//   // 5. Compare the generated SGF with the correct SGF.
//   // if (generated_sgf == correct_sgf) {
//   //    cout << "\nSGF generation is correct!\n";
//   //} else {
//   //    cout << "\nSGF generation is incorrect.\n";
//   //}
//   bool areEquivalent = compareSGF(generated_sgf, correct_sgf);
//   if (areEquivalent)
//     cout << "The two SGF are equivalent" << endl;
//   else
//     cout << "The two SGF are NOT equivalent" << endl;

//   // 6. Verify the SGF data.
//   verifySGF(image_bgr, generated_sgf, intersection_points);
//   // testParseSGFGame();
//   return 0;
// }
