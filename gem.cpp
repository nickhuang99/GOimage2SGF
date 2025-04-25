// gem.cpp
#include "common.h"
#include <algorithm>
#include <fstream>
#include <getopt.h> // Include for getopt_long
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <opencv2/opencv.hpp>
#include <regex> // Include the regex library
#include <set>
#include <sstream>
#include <stdexcept> // Include for standard exceptions
#include <vector>

using namespace cv;
using namespace std;

// Global debug variable
bool bDebug = false;

// Custom exception class for GEM errors
class GEMError : public std::runtime_error {
public:
  GEMError(const std::string &message) : std::runtime_error(message) {}
};

void displayHelpMessage() {
  cout << "Go Environment Manager (GEM)" << endl;
  cout << "Usage: gem [options]" << endl;
  cout << "Options:" << endl;
  cout << "  -p, --process-image <image_path> : Process the Go board image."
       << endl;
  cout << "  -g, --generate-sgf <input_image> <output_sgf>"
       << " : Generate SGF from image." << endl;
  cout << "  -v, --verify <image_path> <sgf_path>"
       << " : Verify board state against SGF." << endl;
  cout << "  -c, --compare <sgf_path1> <sgf_path2>"
       << " : Compare two SGF files." << endl;
  cout << "  --parse <sgf_path>              : Parse an SGF file." << endl;
  cout << "  -h, --help                        : Display this help message."
       << endl;
  cout << "  -d, --debug                       : Enable debug output (must be "
          "at the beginning)."
       << endl;
}

void processImageWorkflow(const std::string &imagePath) {
  cout << "Processing image: " << imagePath << endl;
  cv::Mat image_bgr = imread(imagePath);
  if (image_bgr.empty()) {
    throw GEMError("Could not open or find the image: " + imagePath);
  } else {
    cv::Mat board_state, board_with_stones;
    vector<Point2f> intersection_points;
    processGoBoard(image_bgr, board_state, board_with_stones,
                   intersection_points);
    // Further processing or display (could be moved to another function if
    // needed)
    if (bDebug) {
      imshow("processGoBoard", board_with_stones);
      waitKey(0);
    }
  }
}

void generateSGFWorkflow(const std::string &inputImagePath,
                         const std::string &outputSGFPath) {
  cout << "Generating SGF from image: " << inputImagePath
       << " to: " << outputSGFPath << endl;
  cv::Mat image_bgr = imread(inputImagePath);
  if (image_bgr.empty()) {
    throw GEMError("Could not open or find the input image: " + inputImagePath);
  } else {
    cv::Mat board_state, board_with_stones;    
    std::vector<cv::Point2f> intersections;
    processGoBoard(image_bgr, board_state, board_with_stones, intersections);
    std::string sgf_content = generateSGF(board_state, intersections);
    std::ofstream outfile(outputSGFPath);
    if (!outfile.is_open()) {
      throw GEMError("Could not open SGF file for writing: " + outputSGFPath);
    }
    outfile << sgf_content << endl;
    outfile.close();
    cout << "SGF content written to: " << outputSGFPath << endl;
  }
}

void verifySGFWorkflow(const std::string &imagePath,
                       const std::string &sgfPath) {
  cout << "Verifying image: " << imagePath << " against SGF: " << sgfPath
       << endl;
  cv::Mat image_bgr = imread(imagePath);
  if (image_bgr.empty()) {
    throw GEMError("Could not open or find the image: " + imagePath);
  }
  cv::Mat board_state, board_with_stones;    
  std::vector<cv::Point2f> intersections;
  processGoBoard(image_bgr, board_state, board_with_stones, intersections);

  std::ifstream infile(sgfPath);
  if (!infile.is_open()) {
    throw GEMError("Could not open SGF file: " + sgfPath);
  }
  std::stringstream buffer;
  buffer << infile.rdbuf();
  std::string sgf_data = buffer.str();
  if (sgf_data.empty()) {
    throw GEMError("Could not read SGF data from: " + sgfPath);
  }
  verifySGF(image_bgr, sgf_data, intersections);
}

void compareSGFWorkflow(const std::string &sgfPath1,
                        const std::string &sgfPath2) {
  cout << "Comparing SGF files: " << sgfPath1 << " and " << sgfPath2 << endl;
  std::ifstream infile1(sgfPath1);
  if (!infile1.is_open()) {
    throw GEMError("Could not open the first SGF file: " + sgfPath1);
  }
  std::stringstream buffer1;
  buffer1 << infile1.rdbuf();
  std::string sgf_data1 = buffer1.str();
  if (sgf_data1.empty()) {
    throw GEMError("Could not read SGF data from: " + sgfPath1);
  }

  std::ifstream infile2(sgfPath2);
  if (!infile2.is_open()) {
    throw GEMError("Could not open the second SGF file: " + sgfPath2);
  }
  std::stringstream buffer2;
  buffer2 << infile2.rdbuf();
  std::string sgf_data2 = buffer2.str();
  if (sgf_data2.empty()) {
    throw GEMError("Could not read SGF data from: " + sgfPath2);
  }

  if (compareSGF(sgf_data1, sgf_data2)) {
    cout << "SGF files are identical." << endl;
  } else {
    cout << "SGF files are different." << endl;
  }
}

void parseSGFWorkflow(const std::string &sgfPath) {
  cout << "Parsing SGF file: " << sgfPath << endl;
  std::ifstream infile(sgfPath);
  if (!infile.is_open()) {
    throw GEMError("Could not open SGF file: " + sgfPath);
  }
  std::stringstream buffer;
  buffer << infile.rdbuf();
  std::string sgf_content = buffer.str();
  if (sgf_content.empty()) {
    throw GEMError("Could not read SGF data from: " + sgfPath);
  }

  std::set<std::pair<int, int>> setupBlack, setupWhite;
  std::vector<Move> moves;
  parseSGFGame(sgf_content, setupBlack, setupWhite, moves);
  SGFHeader header = parseSGFHeader(sgf_content);

  cout << "SGF Header:" << endl;
  cout << "  Game: " << header.gm << endl;
  cout << "  File Format: " << header.ff << endl;
  cout << "  Character Set: " << header.ca << endl;
  cout << "  Application: " << header.ap << endl;
  cout << "  Board Size: " << header.sz << endl;

  cout << "\nSetup Black: ";
  for (const auto &stone : setupBlack) {
    cout << "(" << stone.first << "," << stone.second << ") ";
  }
  cout << endl;

  cout << "Setup White: ";
  for (const auto &stone : setupWhite) {
    cout << "(" << stone.first << "," << stone.second << ") ";
  }
  cout << endl;

  cout << "\nMoves:" << endl;
  for (const auto &move : moves) {
    cout << "  Player: " << move.player << ", Row: " << move.row
         << ", Col: " << move.col;
    if (!move.capturedStones.empty()) {
      cout << ", Captured: ";
      for (const auto &captured : move.capturedStones) {
        cout << "(" << captured.first << "," << captured.second << ") ";
      }
    }
    cout << endl;
  }
}

int main(int argc, char *argv[]) {
  try {
    if (argc == 1) {
      displayHelpMessage();
      return 0;
    }
    int option_index = 0;
    struct option long_options[] = {
        {"process-image", required_argument, nullptr, 'p'},
        {"generate-sgf", required_argument, nullptr, 'g'},
        {"verify", required_argument, nullptr, 'v'},
        {"compare", required_argument, nullptr, 'c'},
        {"parse", required_argument, nullptr, 0},
        {"help", no_argument, nullptr, 'h'},
        {"debug", no_argument, nullptr, 'd'},
        {nullptr, 0, nullptr, 0}};

    int c;
    while ((c = getopt_long(argc, argv, "p:g:v:c:hd", long_options,
                            &option_index)) != -1) {
      switch (c) {
      case 'd':
        bDebug = true;
        cout << "Debug mode enabled." << endl;
        break;
      case 'p':
        processImageWorkflow(optarg);
        break;
      case 'g':
        if (optind < argc) {
          generateSGFWorkflow(optarg, argv[optind++]);
        } else {
          throw GEMError(
              "-g option requires an input image path and an output SGF path.");
        }
        break;
      case 'v':
        if (optind < argc) {
          verifySGFWorkflow(optarg, argv[optind++]);
        } else {
          throw GEMError("-v option requires an image path and an SGF path.");
        }
        break;
      case 'c':
        if (optind < argc) {
          compareSGFWorkflow(optarg, argv[optind++]);
        } else {
          throw GEMError("-c option requires two SGF paths.");
        }
        break;
      case 'h':
        displayHelpMessage();
        return 0;
      case 0: // Long-only option
        if (strcmp(long_options[option_index].name, "parse") == 0) {
          parseSGFWorkflow(optarg);
        }
        break;
      case '?':
      default:
        displayHelpMessage();
        return 1;
      }
    }
    // Handle any remaining non-option arguments here if needed

  } catch (const GEMError &e) {
    cerr << "Error: " << e.what() << endl;
    return 1;
  } catch (const std::exception &e) {
    cerr << "An unexpected error occurred: " << e.what() << endl;
    return 1;
  } catch (...) {
    cerr << "An unknown error occurred." << endl;
    return 1;
  }

  return 0;
}