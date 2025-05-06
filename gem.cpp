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
CaptureMode gCaptureMode = MODE_V4L2; // Default capture mode is V4L2
int g_capture_width = 640;  // Default capture width
int g_capture_height = 480; // Default capture height

void displayHelpMessage() {
  cout << "Go Environment Manager (GEM)" << endl;
  cout << "Usage: gem [options]" << endl;
  cout << "Options:" << endl;
  cout << "  -d, --debug                       : Enable debug output (must be "
          "at the beginning)."
       << endl;
  cout << "  -D, --device <device_path>      : Specify the video device path "
          "(default: /dev/video0). Must be at the beginning."
       << endl;
  cout << "  -b, --calibration               : Run calibration workflow "
          "(displays " // Added option
          "live webcam feed)."
       << endl;
  cout << "  -p, --process-image <image_path> : Process the Go board image."
       << endl;
  cout << "  -g, --generate-sgf <input_image> <output_sgf>"
       << " : Generate SGF from image." << endl;
  cout << "  -v, --verify <image_path> <sgf_path>"
       << " : Verify board state against SGF." << endl;
  cout << "  -c, --compare <sgf_path1> <sgf_path2>"
       << " : Compare two SGF files." << endl;
  cout << "  --parse <sgf_path>              : Parse an SGF file." << endl;
  cout << "  --probe-devices                 : List available video devices. "
          "Requires root privileges."
       << endl;
  cout << "  -s, --snapshot <output_file>    : Capture a snapshot from the "
          "webcam. Requires root privileges."
       << endl;
  cout
      << "  -r, --record-sgf <output_sgf>    : Capture a snapshot, process it, "
         "and generate an SGF file. Requires root privileges."
      << endl;
  cout << "  -h, --help                        : Display this help message."
       << endl;
  cout << "\n  Note: Snapshot and recording operations (--probe-devices, "
          "-s/--snapshot, -r/--record-sgf) often require root privileges (use "
          "sudo).\n";
  // Added option
  cout << "  -M, --mode <backend>          : Specify capture backend ('v4l2' "
          "or 'opencv', default: v4l2)."
       << endl;
  cout << "  --size <WxH>                  : Specify capture resolution (e.g., "
          "1280x720). Default: 640x480."
       << endl; // New option
}

void processImageWorkflow(const std::string &imagePath) {
  cout << "Processing image: " << imagePath << endl;
  cv::Mat image_bgr = imread(imagePath);
  if (image_bgr.empty()) {
    THROWGEMERROR("Could not open or find the image: " + imagePath);
  } else {
    try {
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
    } catch (const cv::Exception &e) {
      THROWGEMERROR("OpenCV error in processImageWorkflow: " +
                     string(e.what())); // Wrap OpenCV exceptions
    }
  }
}

void generateSGFWorkflow(const std::string &inputImagePath,
                         const std::string &outputSGFPath) {
  cout << "Generating SGF from image: " << inputImagePath
       << " to: " << outputSGFPath << endl;
  cv::Mat image_bgr = imread(inputImagePath);
  if (image_bgr.empty()) {
    THROWGEMERROR("Could not open or find the input image: " + inputImagePath);
  } else {
    try {
      cv::Mat board_state, board_with_stones;
      std::vector<cv::Point2f> intersections;
      processGoBoard(image_bgr, board_state, board_with_stones, intersections);
      std::string sgf_content = generateSGF(board_state, intersections);
      std::ofstream outfile(outputSGFPath);
      if (!outfile.is_open()) {
        THROWGEMERROR("Could not open SGF file for writing: " + outputSGFPath);
      }
      outfile << sgf_content << endl;
      outfile.close();
      cout << "SGF content written to: " << outputSGFPath << endl;
    } catch (const cv::Exception &e) {
      THROWGEMERROR("OpenCV error in generateSGFWorkflow: " +
                     string(e.what())); // Wrap OpenCV exceptions
    }
  }
}

void verifySGFWorkflow(const std::string &imagePath,
                       const std::string &sgfPath) {
  cout << "Verifying image: " << imagePath << " against SGF: " << sgfPath
       << endl;
  cv::Mat image_bgr = imread(imagePath);
  if (image_bgr.empty()) {
    THROWGEMERROR("Could not open or find the image: " + imagePath);
  } else {
    try {
      cv::Mat board_state, board_with_stones;
      std::vector<cv::Point2f> intersections;
      processGoBoard(image_bgr, board_state, board_with_stones, intersections);

      std::ifstream infile(sgfPath);
      if (!infile.is_open()) {
        THROWGEMERROR("Could not open SGF file: " + sgfPath);
      }
      std::stringstream buffer;
      buffer << infile.rdbuf();
      std::string sgf_data = buffer.str();
      if (sgf_data.empty()) {
        THROWGEMERROR("Could not read SGF data from: " + sgfPath);
      }
      verifySGF(image_bgr, sgf_data, intersections);
    } catch (const cv::Exception &e) {
      THROWGEMERROR("OpenCV error in verifySGFWorkflow: " + string(e.what()));
    }
  }
}

void compareSGFWorkflow(const std::string &sgfPath1,
                        const std::string &sgfPath2) {
  cout << "Comparing SGF files: " << sgfPath1 << " and " << sgfPath2 << endl;
  std::ifstream infile1(sgfPath1);
  if (!infile1.is_open()) {
    THROWGEMERROR("Could not open the first SGF file: " + sgfPath1);
  }
  std::stringstream buffer1;
  buffer1 << infile1.rdbuf();
  std::string sgf_data1 = buffer1.str();
  if (sgf_data1.empty()) {
    THROWGEMERROR("Could not read SGF data from: " + sgfPath1);
  }

  std::ifstream infile2(sgfPath2);
  if (!infile2.is_open()) {
    THROWGEMERROR("Could not open the second SGF file: " + sgfPath2);
  }
  std::stringstream buffer2;
  buffer2 << infile2.rdbuf();
  std::string sgf_data2 = buffer2.str();
  if (sgf_data2.empty()) {
    THROWGEMERROR("Could not read SGF data from: " + sgfPath2);
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
    THROWGEMERROR("Could not open SGF file: " + sgfPath);
  }
  std::stringstream buffer;
  buffer << infile.rdbuf();
  std::string sgf_content = buffer.str();
  if (sgf_content.empty()) {
    THROWGEMERROR("Could not read SGF data from: " + sgfPath);
  }

  std::set<std::pair<int, int>> setupBlack, setupWhite;
  std::vector<Move> moves;
  try {
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
        cout << endl;
      }
    }
  } catch (const SGFError &e) {
    THROWGEMERROR("SGF parsing error: " + string(e.what())); // Wrap SGF errors
  }
}
void probeVideoDevicesWorkflow() {
  std::vector<VideoDeviceInfo> available_devices = probeVideoDevices();

  if (available_devices.empty()) {
    std::cout << "No video capture devices found.\n";
    return;
  }
  std::cout << "Available video capture devices:\n";
  for (size_t i = 0; i < available_devices.size(); ++i) {
    std::cout << "[" << i << "] Path: " << available_devices[i].device_path
              << "\n  Driver: " << available_devices[i].driver_name
              << "\n  Card: " << available_devices[i].card_name
              << "\n  Capabilities: "
              << getCapabilityDescription(available_devices[i].capabilities)
              << " (0x" << std::hex << available_devices[i].capabilities
              << std::dec << ")"
              << "\n  Supported Formats & Sizes:";
    if (available_devices[i].supported_format_details.empty()) {
        std::cout << " None listed or error in enumeration.";
    } else {
        for (const std::string& detail : available_devices[i].supported_format_details) {
            std::cout << "\n    - " << detail;
        }
    }
    std::cout << "\n\n"; // Add an extra newline for better separation
  }
}

void captureSnapshotWorkflow(const std::string &selected_device,
                             const std::string &output) {
  cout << "\nAttempting to capture snapshot from: " << selected_device
       << " using " << (gCaptureMode == MODE_OPENCV ? "OpenCV" : "V4L2")
       << " mode.\n";
  // captureSnapshot function now respects gCaptureMode internally
  if (captureSnapshot(selected_device, output)) {
    std::cout << "Snapshot saved to " << output << std::endl;
  } else {
    // Error message printed within captureSnapshot or its called functions
  }
}

void recordSGFWorkflow(const std::string &device_path,
                       const std::string &output_sgf) {
  cout << "Capturing snapshot using "
       << (gCaptureMode == MODE_OPENCV ? "OpenCV" : "V4L2")
       << " mode, processing, and generating SGF to: " << output_sgf
       << " from device: " << device_path << endl;

  Mat captured_image; 
  try {       
    if (!captureFrame(device_path, captured_image)) {
      THROWGEMERROR("Failed to capture frame from device.");
    }

    // --- Rest of processing remains the same ---
    Mat board_state, board_with_stones;
    vector<Point2f> intersections;
    processGoBoard(captured_image, board_state, board_with_stones,
                   intersections);

    string sgf_content = generateSGF(board_state, intersections);

    ofstream outfile(output_sgf);
    if (!outfile.is_open()) {
      THROWGEMERROR("Could not open SGF file for writing: " + output_sgf);
    }
    outfile << sgf_content << endl;
    outfile.close();
    cout << "SGF content written to: " << output_sgf << endl;

  } catch (const cv::Exception &e) {
    THROWGEMERROR("OpenCV error in recordSGFWorkflow: " + string(e.what()));
  } catch (const GEMError &e) { // Catch GEMError specifically
    THROWGEMERROR(string("Error in recordSGFWorkflow: ") + e.what());
  }
}

// New function for testing perspective transform
void testPerspectiveTransform(const std::string& imagePath) {
  Mat image = imread(imagePath);
  if (image.empty()) {
    cerr << "Error: Could not open image: " << imagePath << endl;
    return;
  }
  Mat correct_image = correctPerspective(image);
}

// --- NEW Calibration Workflow Function ---
void calibrationWorkflow(const std::string &device_path) {
  cout << "Starting Calibration Workflow..." << endl;

  // Extract camera index from device path (e.g., /dev/video0 -> 0)
  int camera_index = 0; // Default
  size_t last_digit_pos = device_path.find_last_of("0123456789");
  if (last_digit_pos != string::npos) {
    size_t first_digit_pos = last_digit_pos;
    while (first_digit_pos > 0 && isdigit(device_path[first_digit_pos - 1])) {
      first_digit_pos--;
    }
    try {
      string index_str = device_path.substr(first_digit_pos);
      camera_index = std::stoi(index_str);
    } catch (const std::exception &e) {
      cerr << "Warning: Could not parse camera index from device path '"
           << device_path << "'. Using default index 0. Error: " << e.what()
           << endl;
      camera_index = 0;
    }
  } else {
    cerr << "Warning: Could not find camera index in device path '"
         << device_path << "'. Using default index 0." << endl;
    camera_index = 0;
  }

  cout << "Displaying live feed from camera index: " << camera_index
       << " (derived from " << device_path << ")" << endl;
  // Call the function (defined in snapshot.cpp)
  runInteractiveCalibration(camera_index);
  cout << "Calibration workflow finished." << endl;
}

int main(int argc, char *argv[]) {
  try {
    if (argc == 1) {
      displayHelpMessage();
      return 0;
    }
    int option_index = 0;
    std::string device_path = "/dev/video0"; // Default device
    std::string snapshot_output;
    std::string record_sgf_output;
    std::string test_image_path; // For -t option
    bool probe_only = false;
    bool run_calibration = false;    // Flag for calibration mode

    struct option long_options[] = {
        {"process-image", required_argument, nullptr, 'p'},
        {"generate-sgf", required_argument, nullptr, 'g'},
        {"verify", required_argument, nullptr, 'v'},
        {"compare", required_argument, nullptr, 'c'},
        {"parse", required_argument, nullptr, 0},
        {"help", no_argument, nullptr, 'h'},
        {"debug", no_argument, nullptr, 'd'},
        {"probe-devices", no_argument, nullptr, 1},
        {"snapshot", required_argument, nullptr, 's'},
        {"device", required_argument, nullptr, 'D'},
        {"record-sgf", required_argument, nullptr, 'r'},
        {"test-perspective", required_argument, nullptr, 't'}, // Add -t option
        {"calibration", no_argument, nullptr, 'b'}, // Added calibration long option 
        {"mode", required_argument, nullptr, 'M'}, // Added mode option 
        {"size", required_argument, nullptr, 'S'}, // Use S as a unique identifier for --size   
        {nullptr, 0, nullptr, 0}};

    int c;
    // Process all options in a single loop
    while ((c = getopt_long(argc, argv, "dp:g:v:c:h:s:r:D:t:bM:S:", long_options,
                            &option_index)) != -1) {
      switch (c) {
      case 'd':
        bDebug = true;
        cout << "Debug mode enabled." << endl;
        break;
      case 'D':
        device_path = optarg;
        break;
      case 'b': // Handle calibration option
        run_calibration = true;
        break;
      case 'M': // Handle capture mode option
        if (optarg) {
          std::string mode_str(optarg);
          if (mode_str == "opencv") {
            gCaptureMode = MODE_OPENCV;
          } else if (mode_str == "v4l2") {
            gCaptureMode = MODE_V4L2;
          } else {
            std::cerr << "Warning: Invalid capture mode '" << mode_str
                      << "'. Using default (v4l2)." << std::endl;
            gCaptureMode = MODE_V4L2;
          }
          if (bDebug)
            std::cout << "Debug: Capture mode set to "
                      << (gCaptureMode == MODE_OPENCV ? "OpenCV" : "V4L2")
                      << std::endl;
        }
        break;
      case 'p':
        processImageWorkflow(optarg);
        break;
      case 'g':
        if (optind < argc) {
          generateSGFWorkflow(optarg, argv[optind++]);
        } else {
          THROWGEMERROR(
              "-g option requires an input image path and an output SGF path.");
        }
        break;
      case 'v':
        if (optind < argc) {
          verifySGFWorkflow(optarg, argv[optind++]);
        } else {
          THROWGEMERROR("-v option requires an image path and an SGF path.");
        }
        break;
      case 'c':
        if (optind < argc) {
          compareSGFWorkflow(optarg, argv[optind++]);
        } else {
          THROWGEMERROR("-c option requires two SGF paths.");
        }
        break;
      case 'S': // Corresponds to --size
      {         // Added braces for variable scope
        std::string size_str = optarg;
        size_t delimiter_pos = size_str.find('x');
        if (delimiter_pos != std::string::npos) {
          std::string width_s = size_str.substr(0, delimiter_pos);
          std::string height_s = size_str.substr(delimiter_pos + 1);
          try {
            g_capture_width = std::stoi(width_s);
            g_capture_height = std::stoi(height_s);
            if (g_capture_width <= 0 || g_capture_height <= 0) {
              std::cerr
                  << "Error: Frame dimensions must be positive for --size."
                  << std::endl;
              return 1;
            }
            if (bDebug) {
              std::cout << "Debug: Requested capture size set to "
                        << g_capture_width << "x" << g_capture_height
                        << " by option." << std::endl;
            }
          } catch (const std::invalid_argument &ia) {
            std::cerr << "Error: Invalid number format in --size argument: "
                      << size_str << std::endl;
            return 1;
          } catch (const std::out_of_range &oor) {
            std::cerr << "Error: Number out of range in --size argument: "
                      << size_str << std::endl;
            return 1;
          }
        } else {
          std::cerr << "Error: Invalid format for --size. Expected "
                       "WIDTHxHEIGHT (e.g., 640x480)."
                    << std::endl;
          return 1;
        }
      } // End of scope for case 'S'
      break;
      case 'h':
        displayHelpMessage();
        return 0;
      case 0: // Long-only option
        if (strcmp(long_options[option_index].name, "parse") == 0) {
          parseSGFWorkflow(optarg);
        }
        break;
      case 1: // Long-only option
        if (strcmp(long_options[option_index].name, "probe-devices") == 0) {
          probe_only = true;
          probeVideoDevicesWorkflow();
        }
        break;
      case 's':
        snapshot_output = optarg;
        break;
      case 'r':
        record_sgf_output = optarg;
        break;
      case 't': // Handle -t option
        test_image_path = optarg;
        testPerspectiveTransform(test_image_path);
        return 0; // Exit after testing
      case '?':
      default:
        displayHelpMessage();
        return 1;
      }
    }

    // Calibration takes precedence if set
    if (run_calibration) {
      calibrationWorkflow(device_path);
      return 0; // Exit after calibration
    }
    // Handle any remaining non-option arguments here if needed
    if (probe_only) {
      return 0; // Exit if only probing devices
    }

    if (!snapshot_output.empty()) {
      captureSnapshotWorkflow(device_path, snapshot_output);
    }

    if (!record_sgf_output.empty()) {
      recordSGFWorkflow(device_path, record_sgf_output);
    }
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