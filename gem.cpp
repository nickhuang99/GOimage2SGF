// gem.cpp
#include "common.h"
#include <algorithm>
#include <bits/getopt_core.h>
#include <cerrno> // For errno
#include <filesystem> // For std::filesystem::remove_all and create_directory (C++17)
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
#include <stdexcept>  // Include for standard exceptions
#include <sys/stat.h> // For mkdir (though filesystem is preferred)
#include <tuple>
#include <vector>

using namespace cv;
using namespace std;
namespace fs = std::filesystem; // Alias for convenience

// Global debug variable
bool bDebug = false;
CaptureMode gCaptureMode = MODE_V4L2; // Default capture mode is V4L2
int g_capture_width = 640;            // Default capture width
int g_capture_height = 480;           // Default capture height
std::string g_default_game_name_prefix = "tournament";
std::string g_device_path = "/dev/video0";

static const std::string Default_Go_Board_Window_Title = "Simulated Go Board";
static const int canvas_size_px = 760;
void displayHelpMessage() {
  cout << "Go Environment Manager (GEM)" << endl;
  cout << "Usage: gem [options]" << endl;
  cout << "Options:" << endl;
  cout << "  -d, --debug                       : Enable debug output (must be "
          "at the beginning)."
       << endl;
  cout << "  -D, --device <device_path>      : Specify video device by number "
          "(e.g., 0, 1) "
          "(default: 0 which is equivalent to /dev/video0). Must be at the "
          "beginning."
       << endl;
  cout << "  -M, --mode <backend>              : Specify capture backend "
          "('v4l2' "
          "or 'opencv', default: v4l2)."
       << endl;
  cout << "  --size <WxH>                      : Specify capture resolution "
          "(e.g., "
          "1280x720). Default: 640x480."
       << endl;
  cout << "  -b, --calibration                 : Run capture calibration "
          "workflow."
       << endl;
  cout << "  -B, --interactive-calibration     : Run interactive calibration "
          "workflow."
       << endl;
  cout << "  --test-calibration-config         : Load calibration snapshot and "
          "config, draw corners."
       << endl;
  cout << "  -p, --process-image <image_path>   : Process the Go board image."
       << endl;
  cout << "  -g, --generate-sgf <in_img> <out_sgf> : Generate SGF from image."
       << endl;
  cout << "  -v, --verify <image_path> <sgf_path> : Verify board state against "
          "SGF."
       << endl;
  cout << "  -c, --compare <sgf1> <sgf2>         : Compare two SGF files."
       << endl;
  cout << "  --parse <sgf_path>                  : Parse an SGF file." << endl;
  cout
      << "  --probe-devices                     : List available video devices."
      << endl;
  cout << "  -s, --snapshot <output_file>        : Capture a snapshot from the "
          "webcam."
       << endl;
  cout << "  -r, --record-sgf <output_sgf>        : Capture snapshot, process, "
          "and generate SGF."
       << endl;
  cout << "  -t, --tournament             : Start tournament recording "
          "mode."
       << endl;
  cout << "      --game-name <prefix>            : Set game name prefix for "
          "tournament or study mode (default: "
       << g_default_game_name_prefix << " or 'tournament' if only -t is used)."
       << endl;
  cout << "      --draw-board <sgf_path>         : Read SGF, draw and display "
          "simulated board."
       << endl; // NEW
  cout << "  -u, --study             : Start study mode to replay a "
          "recorded game."
       << endl; // NEW
  cout << "      --test-perspective <image_path> : (Dev) Test perspective "
          "correction." // Removed -t
       << endl;
  cout << "  -h, --help                          : Display this help message."
       << endl;
  cout << "\n  Note: Webcam operations may require appropriate permissions "
          "(e.g., user in 'video' group)."
       << endl;
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
      if (bDebug || true) {
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
              << std::dec << ")" << "\n  Supported Formats & Sizes:";
    if (available_devices[i].supported_format_details.empty()) {
      std::cout << " None listed or error in enumeration.";
    } else {
      for (const std::string &detail :
           available_devices[i].supported_format_details) {
        std::cout << "\n    - " << detail;
      }
    }
    std::cout << "\n\n"; // Add an extra newline for better separation
  }
}

void captureSnapshotWorkflow(const std::string &output) {
  cout << "\nAttempting to capture snapshot from: " << g_device_path
       << " using " << (gCaptureMode == MODE_OPENCV ? "OpenCV" : "V4L2")
       << " mode.\n";
  // captureSnapshot function now respects gCaptureMode internally
  if (captureSnapshot(g_device_path, output)) {
    std::cout << "Snapshot saved to " << output << std::endl;
  } else {
    // Error message printed within captureSnapshot or its called functions
  }
}

void recordSGFWorkflow(const std::string &output_sgf) {
  cout << "Capturing snapshot using "
       << (gCaptureMode == MODE_OPENCV ? "OpenCV" : "V4L2")
       << " mode, processing, and generating SGF to: " << output_sgf
       << " from device: " << g_device_path << endl;

  Mat captured_image;
  try {
    if (!captureFrame(g_device_path, captured_image)) {
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
void testPerspectiveTransform(const std::string &imagePath) {
  Mat image = imread(imagePath);
  if (image.empty()) {
    cerr << "Error: Could not open image: " << imagePath << endl;
    return;
  }
  Mat correct_image = correctPerspective(image);
}

// --- NEW Calibration Workflow Function ---
void calibrationWorkflow(bool bInteractive) {
  cout << "Starting Calibration Workflow..." << endl;

  // Extract camera index from device path (e.g., /dev/video0 -> 0)
  int camera_index = 0; // Default
  size_t last_digit_pos = g_device_path.find_last_of("0123456789");
  if (last_digit_pos != string::npos) {
    size_t first_digit_pos = last_digit_pos;
    while (first_digit_pos > 0 && isdigit(g_device_path[first_digit_pos - 1])) {
      first_digit_pos--;
    }
    try {
      string index_str = g_device_path.substr(first_digit_pos);
      camera_index = std::stoi(index_str);
    } catch (const std::exception &e) {
      cerr << "Warning: Could not parse camera index from device path '"
           << g_device_path << "'. Using default index 0. Error: " << e.what()
           << endl;
      camera_index = 0;
    }
  } else {
    cerr << "Warning: Could not find camera index in device path '"
         << g_device_path << "'. Using default index 0." << endl;
    camera_index = 0;
  }

  cout << "Displaying live feed from camera index: " << camera_index
       << " (derived from " << g_device_path << ")" << endl;
  // Call the function (defined in snapshot.cpp)
  if (bInteractive) {
    runInteractiveCalibration(camera_index);
  } else {
    runCaptureCalibration(camera_index);
  }

  cout << "Calibration workflow finished." << endl;
}

// --- NEW Workflow Function ---
void testCalibrationConfigWorkflow() {
  std::cout << "Testing Calibration Config..." << std::endl;

  // 1. Load calibration data (corners and colors)
  std::cout << "  Loading config file: " << CALIB_CONFIG_PATH << std::endl;
  CalibrationData data =
      loadCalibrationData(CALIB_CONFIG_PATH); // Uses new function

  // Check if essential data was loaded
  if (!data.corners_loaded) {
    THROWGEMERROR("Failed to load corner data from config file: " +
                  CALIB_CONFIG_PATH);
  }
  if (!data.colors_loaded) {
    // Don't throw error here, maybe calibration was done before color sampling
    // was added
    std::cout << "  Warning: Color calibration data not found or incomplete in "
                 "config file."
              << std::endl;
    // Allow test to proceed showing only corners
  }
  if (!data.dimensions_loaded) {
    std::cout << "  Warning: Image dimensions not found in config file."
              << std::endl;
  }

  // Print loaded data
  std::cout << "  --- Loaded Calibration Data ---" << std::endl;
  if (data.dimensions_loaded) {
    std::cout << "  Dimensions: " << data.image_width << "x"
              << data.image_height << std::endl;
  }
  std::cout << "  Corners Loaded: " << (data.corners_loaded ? "Yes" : "No")
            << std::endl;
  if (data.corners_loaded) {
    std::cout << "    TL: " << data.corners[0] << std::endl;
    std::cout << "    TR: " << data.corners[1] << std::endl;
    std::cout << "    BR: " << data.corners[2] << std::endl;
    std::cout << "    BL: " << data.corners[3] << std::endl;
  }
  std::cout << "  Colors Loaded: " << (data.colors_loaded ? "Yes" : "No")
            << std::endl;
  if (data.colors_loaded) {
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "    TL Lab: [" << data.lab_tl[0] << "," << data.lab_tl[1]
              << "," << data.lab_tl[2] << "]" << std::endl;
    std::cout << "    TR Lab: [" << data.lab_tr[0] << "," << data.lab_tr[1]
              << "," << data.lab_tr[2] << "]" << std::endl;
    std::cout << "    BL Lab: [" << data.lab_bl[0] << "," << data.lab_bl[1]
              << "," << data.lab_bl[2] << "]" << std::endl;
    std::cout << "    BR Lab: [" << data.lab_br[0] << "," << data.lab_br[1]
              << "," << data.lab_br[2] << "]" << std::endl;
  }
  std::cout << "  -------------------------------" << std::endl;

  // 2. Determine snapshot file path and load image
  std::string snapshot_path =
      bDebug ? CALIB_SNAPSHOT_DEBUG_PATH : CALIB_SNAPSHOT_PATH;
  std::cout << "  Loading snapshot image: " << snapshot_path << std::endl;
  cv::Mat image = cv::imread(snapshot_path);
  if (image.empty()) {
    THROWGEMERROR("Failed to load calibration snapshot image: " +
                  snapshot_path);
  }

  // 3. Draw markers on image using loaded corners
  std::cout << "  Drawing markers on snapshot..." << std::endl;
  // Colors for markers (same as before)
  cv::Scalar marker_color_tl =
      bDebug ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
  cv::Scalar marker_color_tr =
      bDebug ? cv::Scalar(255, 255, 0) : cv::Scalar(255, 0, 0);
  cv::Scalar marker_color_br =
      bDebug ? cv::Scalar(0, 165, 255) : cv::Scalar(0, 255, 255);
  cv::Scalar marker_color_bl =
      bDebug ? cv::Scalar(255, 0, 128) : cv::Scalar(255, 0, 255);
  int marker_size = 20;
  int marker_thickness = 2;

  if (data.corners_loaded) {
    cv::drawMarker(image, data.corners[0], marker_color_tl, cv::MARKER_CROSS,
                   marker_size, marker_thickness); // TL
    cv::drawMarker(image, data.corners[1], marker_color_tr, cv::MARKER_CROSS,
                   marker_size, marker_thickness); // TR
    cv::drawMarker(image, data.corners[2], marker_color_br, cv::MARKER_CROSS,
                   marker_size, marker_thickness); // BR
    cv::drawMarker(image, data.corners[3], marker_color_bl, cv::MARKER_CROSS,
                   marker_size, marker_thickness); // BL
  } else {
    std::cout << "  Warning: Cannot draw markers as corners were not loaded."
              << std::endl;
  }

  // 4. Display the image
  std::string window_title = "Calibration Test Verification";
  cv::imshow(window_title, image);
  std::cout << "  Displaying image. Press any key or close window to exit."
            << std::endl;
  cv::waitKey(0);
  cv::destroyAllWindows();

  std::cout << "Calibration Test Finished." << std::endl;
}

void testPerspectiveTransformWorkflow(
    const std::string &imagePath) { // Renamed function
  cout << "Testing perspective transform on image: " << imagePath << endl;
  Mat image = imread(imagePath);
  if (image.empty()) {
    cerr << "Error: Could not open image for perspective test: " << imagePath
         << endl;
    return;
  }
  Mat corrected_image = correctPerspective(image);
  if (bDebug || !corrected_image.empty()) {
    imshow("Original for Perspective Test", image);
    imshow("Corrected Perspective Test", corrected_image);
    waitKey(0);
    destroyAllWindows();
  } else if (corrected_image.empty()) {
    cerr << "Error: Perspective correction resulted in an empty image." << endl;
  }
}
static bool createOrClearGameFolder(const std::string &folder_path,
                                    bool debug_flag) {
  // 1. Ensure ./share exists (as per README, user should create it with 777)
  //    This function will not attempt to create ./share itself.
  if (!fs::exists("./share")) {
    THROWGEMERROR("Base directory './share' does not exist. Please create it "
                  "with appropriate permissions (e.g., chmod 777 share).");
  }
  if (!fs::is_directory("./share")) {
    THROWGEMERROR("'./share' exists but is not a directory.");
  }

  // 2. Handle the game-specific folder
  if (fs::exists(folder_path)) {
    if (debug_flag)
      std::cout << "  Debug: Game folder " << folder_path
                << " exists. Attempting to remove and recreate..." << std::endl;
    std::error_code ec;
    fs::remove_all(folder_path,
                   ec); // C++17 way to remove directory and its contents
    if (ec) {
      // remove_all might fail if permissions are wrong or files are in use.
      std::cerr << "Warning: Could not completely remove existing game folder '"
                << folder_path << "': " << ec.message()
                << ". Attempting to proceed..." << std::endl;
      // Attempt to create anyway, it might be empty or mkdir might handle it.
    } else {
      if (debug_flag)
        std::cout << "    Debug: Successfully removed existing folder: "
                  << folder_path << std::endl;
    }
  }

  // 3. Create the new game folder
  std::error_code ec_create;
  fs::create_directory(folder_path, ec_create);
  if (ec_create) {
    THROWGEMERROR("Failed to create game folder: " + folder_path + " - " +
                  ec_create.message());
  }
  if (debug_flag)
    std::cout << "  Debug: Ensured game folder exists (created if new): "
              << folder_path << std::endl;

  return true;
}

static std::string
generateFirstGameStateSGF(const std::string &current_step_sgf_content_full) {
  size_t last_paren_pos = current_step_sgf_content_full.rfind(')');
  if (last_paren_pos != std::string::npos) {
    std::string content_for_tournament_start =
        current_step_sgf_content_full.substr(0, last_paren_pos);
    while (!content_for_tournament_start.empty() &&
           (content_for_tournament_start.back() == '\n' ||
            content_for_tournament_start.back() == '\r')) {
      content_for_tournament_start.pop_back();
    }
    std::cout << "  generateFirstGameStateSGF: " << content_for_tournament_start
              << std::endl;
    return content_for_tournament_start;
  }
  THROWGEMERROR("generateFirstGameStateSGF fails to parse game state:" +
                current_step_sgf_content_full)
}

static bool setupCalibrationFromConfig() {
  std::cout << "  Attempting to load calibration settings from: "
            << CALIB_CONFIG_PATH << std::endl;
  CalibrationData data = loadCalibrationData(
      CALIB_CONFIG_PATH); // loadCalibrationData is in image.cpp

  if (!data.device_path_loaded || data.device_path.empty()) {
    THROWGEMERROR("Tournament mode requires 'DevicePath' in " +
                  CALIB_CONFIG_PATH + ". Please run calibration first.");
  }
  if (!data.dimensions_loaded || data.image_width <= 0 ||
      data.image_height <= 0) {
    THROWGEMERROR(
        "Tournament mode requires 'ImageWidth' and 'ImageHeight' in " +
        CALIB_CONFIG_PATH + ". Please run calibration first.");
  }
  if (!data.corners_loaded) {
    THROWGEMERROR("Tournament mode requires corner data (e.g. TL_X_PX) in " +
                  CALIB_CONFIG_PATH + ". Please run calibration first.");
  }
  if (!data.colors_loaded || !data.board_color_loaded) {
    THROWGEMERROR("Tournament mode requires color calibration data (stones and "
                  "board average) in " +
                  CALIB_CONFIG_PATH + ". Please run calibration first.");
  }

  // Store original command-line specified values (or defaults)
  std::string original_g_device_path = g_device_path;
  int original_g_capture_width = g_capture_width;
  int original_g_capture_height = g_capture_height;

  // Update globals with values from config file for tournament mode
  g_device_path = data.device_path;
  g_capture_width = data.image_width;
  g_capture_height = data.image_height;

  std::cout << "  Successfully loaded calibration settings for Tournament Mode:"
            << std::endl;
  std::cout << "    Device Path: " << g_device_path << " (from config)"
            << std::endl;
  std::cout << "    Resolution: " << g_capture_width << "x" << g_capture_height
            << " (from config)" << std::endl;

  // Check if command-line options for device/size were also provided and warn
  // if they differ User might have specified -D or --size along with -t
  bool cmd_line_device_differs =
      (original_g_device_path != g_device_path) &&
      (original_g_device_path !=
       "/dev/video0"); // Check if original was not the un-touched default
  bool cmd_line_size_differs =
      (original_g_capture_width != g_capture_width ||
       original_g_capture_height != g_capture_height) &&
      (original_g_capture_width != 640 ||
       original_g_capture_height !=
           480); // Check if original was not un-touched default

  if (cmd_line_device_differs) {
    std::cout << "    WARNING: Command-line specified device '"
              << original_g_device_path
              << "' is overridden by calibrated device '" << g_device_path
              << "' from config for tournament mode." << std::endl;
  }
  if (cmd_line_size_differs) {
    std::cout << "    WARNING: Command-line specified size '"
              << original_g_capture_width << "x" << original_g_capture_height
              << "' is overridden by calibrated size '" << g_capture_width
              << "x" << g_capture_height << "' from config for tournament mode."
              << std::endl;
  }

  return true;
}
void tournamentModeWorkflow(const std::string &game_name_final_prefix) {
  // Assuming setupCalibrationFromConfig() might have been called in main() or
  // here if needed For this function, we'll use g_device_path, g_capture_width,
  // g_capture_height as they are.

  std::cout << "Starting Tournament Mode (with Last Move Highlight)..."
            << std::endl;
  std::cout << "  Game Name Prefix: " << game_name_final_prefix << std::endl;
  std::cout << "  Using Device: " << g_device_path << " at " << g_capture_width
            << "x" << g_capture_height << std::endl;

  std::string base_share_path = "./share/";
  std::string game_folder_path = base_share_path + game_name_final_prefix;

  if (!createOrClearGameFolder(
          game_folder_path,
          bDebug)) { // Ensure this function is defined in gem.cpp
    return;
  }

  std::string main_sgf_path = game_folder_path + "/tournament.sgf";

  int game_step_counter = 0;
  cv::Mat previous_board_state_matrix;
  cv::Mat current_raw_frame;
  cv::Mat simulated_board_display_image;
  cv::Mat processed_camera_capture_display_debug;

  std::string main_display_window_name =
      "Go Tournament: " + game_name_final_prefix + " (Simulated Board)";
  cv::namedWindow(main_display_window_name, cv::WINDOW_AUTOSIZE);

  std::string debug_capture_window_name = "Debug: Processed Camera View";
  if (bDebug) {
    cv::namedWindow(debug_capture_window_name, cv::WINDOW_AUTOSIZE);
    cv::moveWindow(main_display_window_name, 50, 50);
    // Ensure canvas_size_px is defined, e.g., static const int canvas_size_px =
    // 760;
    cv::moveWindow(debug_capture_window_name, 50 + canvas_size_px + 20, 50);
  }

  // Lambda to parse the primary move from an SGF move string (e.g., ";B[dp]")
  // Returns: row (0-18), col (0-18), color (BLACK/WHITE). Returns (-1,-1,EMPTY)
  // on failure or non-B/W move.
  auto parseSgfMoveNode =
      [](const std::string &sgf_move_node_str) -> std::tuple<int, int, int> {
    if (sgf_move_node_str.length() < 5 || sgf_move_node_str[0] != ';') {
      return {-1, -1, EMPTY};
    }

    char player_char = sgf_move_node_str[1];
    int stone_color = EMPTY;

    if (player_char == 'B')
      stone_color = BLACK;
    else if (player_char == 'W')
      stone_color = WHITE;
    else
      return {-1, -1, EMPTY};

    size_t open_bracket = sgf_move_node_str.find('[', 2);
    size_t close_bracket = sgf_move_node_str.find(']', open_bracket + 1);

    if (open_bracket != std::string::npos &&
        close_bracket != std::string::npos &&
        (close_bracket == open_bracket + 3)) {

      char col_sgf = sgf_move_node_str[open_bracket + 1];
      char row_sgf = sgf_move_node_str[open_bracket + 2];

      if (col_sgf >= 'a' && col_sgf <= 's' && row_sgf >= 'a' &&
          row_sgf <= 's') {
        return {row_sgf - 'a', col_sgf - 'a', stone_color};
      }
    }
    return {-1, -1, EMPTY};
  };

  std::string accumulated_tournament_sgf_content =
      ""; // To store the SGF content as it's built

  while (true) {
    std::string current_step_info_str =
        "Step " + std::to_string(game_step_counter);
    cv::setWindowTitle(main_display_window_name, main_display_window_name +
                                                     " - " +
                                                     current_step_info_str);
    if (bDebug) {
      cv::setWindowTitle(debug_capture_window_name, debug_capture_window_name +
                                                        " - " +
                                                        current_step_info_str);
    }

    if (!captureFrame(g_device_path, current_raw_frame)) {
      std::cerr << "Error: Failed to capture frame for "
                << current_step_info_str << ". Retrying in 1s..." << std::endl;
      if (simulated_board_display_image.empty())
        simulated_board_display_image =
            cv::Mat::zeros(cv::Size(canvas_size_px, canvas_size_px), CV_8UC3);
      cv::Mat error_display = simulated_board_display_image.clone();
      cv::putText(error_display, "Capture Failed. Check Camera. ESC to exit.",
                  cv::Point(50, error_display.rows / 2),
                  cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
      cv::imshow(main_display_window_name, error_display);
      if (cv::waitKey(1000) == 27)
        break;
      continue;
    }

    if (bDebug) { /* ... debug display of raw capture ... */
    }

    std::cout << "Processing " << current_step_info_str << "..." << std::endl;
    cv::Mat temp_display_processing;
    if (!simulated_board_display_image.empty() && game_step_counter > 0) {
      temp_display_processing = simulated_board_display_image.clone();
    } else {
      temp_display_processing = current_raw_frame.clone();
    }
    cv::putText(temp_display_processing,
                "Processing Step " + std::to_string(game_step_counter) + "...",
                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 255), 2);
    cv::imshow(main_display_window_name, temp_display_processing);
    cv::waitKey(1);

    cv::Mat current_board_state_matrix_local;
    std::vector<cv::Point2f> intersections;
    bool processing_ok = true;
    try {
      processGoBoard(current_raw_frame, current_board_state_matrix_local,
                     processed_camera_capture_display_debug, intersections);
      if (bDebug) {
        cv::imshow(debug_capture_window_name,
                   processed_camera_capture_display_debug);
      }
    } catch (const std::exception &e) {
      std::cerr << "Error processing board for " << current_step_info_str
                << ": " << e.what() << std::endl;
      processing_ok = false;
      if (simulated_board_display_image.empty())
        simulated_board_display_image =
            cv::Mat::zeros(cv::Size(canvas_size_px, canvas_size_px), CV_8UC3);
      cv::Mat error_display = simulated_board_display_image.clone();
      cv::putText(error_display, "Processing Error! " + current_step_info_str,
                  cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                  cv::Scalar(0, 0, 255), 2);
      cv::putText(error_display, "Press 'N' for next, 'ESC' to exit.",
                  cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  cv::Scalar(0, 0, 255), 1);
      cv::imshow(main_display_window_name, error_display);
    }

    std::string current_step_sgf_content_full =
        ""; // SGF of the current board state (AB/AW format)

    if (processing_ok) {
      std::string step_snapshot_path =
          game_folder_path + "/" + std::to_string(game_step_counter) + ".jpg";
      if (!cv::imwrite(step_snapshot_path,
                       current_raw_frame)) { /* ... cerr ... */
      } else {
        std::cout << "  Saved step snapshot: " << step_snapshot_path
                  << std::endl;
      }

      current_step_sgf_content_full =
          generateSGF(current_board_state_matrix_local, intersections);
      std::string step_sgf_path =
          game_folder_path + "/" + std::to_string(game_step_counter) + ".sgf";
      std::ofstream step_sgf_file_out(step_sgf_path);
      if (step_sgf_file_out.is_open()) {
        step_sgf_file_out << current_step_sgf_content_full;
        step_sgf_file_out.close();
        std::cout << "  Saved step SGF: " << step_sgf_path << std::endl;
      } else { /* ... cerr ... */
      }

      if (previous_board_state_matrix.empty()) {
        std::ofstream tournament_sgf_out(main_sgf_path, std::ios::trunc);
        if (!tournament_sgf_out.is_open()) {
          THROWGEMERROR("Failed to open SGF for initial write: " +
                        main_sgf_path);
        }
        size_t last_paren_pos = current_step_sgf_content_full.rfind(')');
        std::string content_for_tournament_start;
        if (last_paren_pos != std::string::npos) {
          content_for_tournament_start =
              current_step_sgf_content_full.substr(0, last_paren_pos);
          while (!content_for_tournament_start.empty() &&
                 (content_for_tournament_start.back() == '\n' ||
                  content_for_tournament_start.back() == '\r')) {
            content_for_tournament_start.pop_back();
          }
        } else {
          content_for_tournament_start =
              "(;FF[4]GM[1]SZ[19]AP[GEM:ErrorFormattingStep" +
              std::to_string(game_step_counter) + "]";
        }
        tournament_sgf_out << content_for_tournament_start;
        accumulated_tournament_sgf_content =
            content_for_tournament_start; // Store for drawing
        std::cout << "  Initialized " << main_sgf_path
                  << " with content from step " << game_step_counter
                  << std::endl;
        tournament_sgf_out.close();
      } else {
        std::ofstream main_sgf_appender(main_sgf_path, std::ios::app);
        if (!main_sgf_appender.is_open()) {
          std::cerr
              << "CRITICAL ERROR: Failed to open main SGF for appending move: "
              << main_sgf_path << std::endl;
        } else {
          std::string move_made_sgf_node = determineSGFMove(
              previous_board_state_matrix, current_board_state_matrix_local);
          if (!move_made_sgf_node.empty() &&
              move_made_sgf_node.find("ERROR") == std::string::npos) {
            main_sgf_appender << "\n" << move_made_sgf_node;
            accumulated_tournament_sgf_content +=
                "\n" + move_made_sgf_node; // Append to in-memory version
            std::cout << "  Appended move: " << move_made_sgf_node << " to "
                      << main_sgf_path << std::endl;
          } else { /* ... handle error or no move ... */
          }
          main_sgf_appender.close();
        }
      }
      previous_board_state_matrix = current_board_state_matrix_local.clone();

      // Use the accumulated SGF content for drawing, which represents the
      // tournament.sgf state The game_step_counter is the number of B/W moves
      // made after setup (0 for setup itself)
      drawSimulatedGoBoard(accumulated_tournament_sgf_content +
                               ")", // Add closing paren for valid parsing by
                                    // drawSimulatedGoBoard
                           game_step_counter, // display_up_to_move_idx
                           simulated_board_display_image,
                           game_step_counter, // highlight_this_move_idx
                           canvas_size_px);
    }

    cv::putText(simulated_board_display_image,
                current_step_info_str +
                    (processing_ok ? ". Ready. " : ". Error. ") +
                    "N: Next, ESC: Exit.",
                cv::Point(10, simulated_board_display_image.rows > 15
                                  ? simulated_board_display_image.rows - 15
                                  : 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                processing_ok ? cv::Scalar(0, 100, 0) : cv::Scalar(0, 0, 150),
                1, cv::LINE_AA);
    cv::imshow(main_display_window_name, simulated_board_display_image);

    if (bDebug && processing_ok) {
      cv::imshow(debug_capture_window_name,
                 processed_camera_capture_display_debug);
    } else if (bDebug && !processing_ok) {
      cv::imshow(debug_capture_window_name, current_raw_frame);
    }

    int key_input = cv::waitKey(0);

    if (key_input == 27) {
      std::cout << "Tournament mode finishing after " << current_step_info_str
                << "." << std::endl;
      break;
    } else if (key_input == 'n' || key_input == 'N') {
      if (processing_ok) {
        game_step_counter++;
      } else {
        std::cout << "  Processing error in current step. Press N to try "
                     "capturing again for step "
                  << game_step_counter << " or ESC to exit." << std::endl;
      }
    } else {
      if (bDebug)
        std::cout << "Ignored key: " << key_input << ". Press N or ESC."
                  << std::endl;
    }
  }

  if (!previous_board_state_matrix.empty()) {
    std::ofstream main_sgf_final_stream(main_sgf_path, std::ios::app);
    if (main_sgf_final_stream.is_open()) {
      // Check if main_sgf_path already ends with ')', if not, add it.
      // This is a bit tricky as accumulated_tournament_sgf_content doesn't have
      // the final ')' until here. The initial write for step 0 already removed
      // the ')' from current_step_sgf_content_full. So, we just need to append
      // one final ')' here.
      main_sgf_final_stream << "\n)" << std::endl;
      main_sgf_final_stream.close();
      std::cout << "  Finalized main tournament SGF: " << main_sgf_path
                << std::endl;
    } else { /* Error message */
    }
  } else { /* No successful steps message */
  }

  cv::destroyAllWindows();
  std::cout << "Tournament Mode Finished. Game data saved in: "
            << game_folder_path << std::endl;
}

void drawSimulatedBoardWorkflow(const std::string &sgf_file_path) {
  std::cout << "Starting Draw Simulated Board Workflow..." << std::endl;
  std::cout << "  SGF File: " << sgf_file_path << std::endl;

  std::ifstream sgf_file_stream(sgf_file_path);
  if (!sgf_file_stream.is_open()) {
    THROWGEMERROR("Failed to open SGF file: " + sgf_file_path);
  }

  std::stringstream buffer;
  buffer << sgf_file_stream.rdbuf();
  std::string sgf_content = buffer.str();
  sgf_file_stream.close();

  if (sgf_content.empty()) {
    THROWGEMERROR("SGF file is empty or could not be read: " + sgf_file_path);
  }

  cv::Mat board_image;
  
  // Determine total B/W moves to display the final state and highlight the last move
  std::set<std::pair<int, int>> setupB, setupW; // Not strictly needed for count, but parseSGFGame needs them
  std::vector<Move> moves;
  SGFHeader header; // Not strictly needed for count here
  try {
    header = parseSGFHeader(sgf_content); // Optional
    parseSGFGame(sgf_content, setupB, setupW, moves);
  } catch (const SGFError& e) {
      THROWGEMERROR("SGF Parsing error in drawSimulatedBoardWorkflow: " + std::string(e.what()));
  }
  
  int total_bw_moves = 0;
  for(const auto& move : moves){
      if(move.player == BLACK || move.player == WHITE){
          total_bw_moves++;
      }
  }

  // Call with corrected signature:
  // 1. SGF content string
  // 2. Display up to this move index (total_bw_moves means all B/W moves)
  // 3. Output Mat
  // 4. Highlight this move index (total_bw_moves to highlight the last B/W move)
  // 5. Canvas size (optional, uses default)
  drawSimulatedGoBoard(sgf_content, 
                       total_bw_moves, 
                       board_image, 
                       total_bw_moves, 
                       canvas_size_px); 

  if (board_image.empty()) {
    THROWGEMERROR("drawSimulatedGoBoard function returned an empty image.");
  }

  std::string window_title = Default_Go_Board_Window_Title + ": " + sgf_file_path;
  cv::imshow(window_title, board_image);
  std::cout << "  Displaying simulated board. Press any key in the OpenCV window to close." << std::endl;
  cv::waitKey(0);
  cv::destroyAllWindows();
  std::cout << "Draw Simulated Board Workflow Finished." << std::endl;
}


// Corrected studyModeWorkflow (Phase 1 structure, calls to drawSimulatedGoBoard updated for Phase 2 signature)
static int loadGameSteps(const std::string &game_folder_path,
                         std::vector<std::string> &sgf_files_out,
                         std::vector<std::string> &jpg_files_out) {
    // ... (Implementation from your latest gem.cpp)
    sgf_files_out.clear(); jpg_files_out.clear(); std::map<int, std::string> found_sgfs, found_jpgs; int max_step = -1;
    try {
        if (!fs::exists(game_folder_path) || !fs::is_directory(game_folder_path)) return -1;
        for (const auto& entry : fs::directory_iterator(game_folder_path)) {
            if (entry.is_regular_file()) {
                std::string fn = entry.path().filename().string(), ext = entry.path().extension().string(), name = fn.substr(0, fn.length()-ext.length());
                try { size_t chars_proc=0; int sn=std::stoi(name, &chars_proc); if(chars_proc==name.length()){ if(ext==".sgf")found_sgfs[sn]=entry.path().string(); else if(ext==".jpg")found_jpgs[sn]=entry.path().string(); if(sn>max_step)max_step=sn;} } catch(...){}
            }
        }
        int actual_max_step = -1;
        for(int i=0; i<=max_step; ++i) { if(found_sgfs.count(i)&&found_jpgs.count(i)){ sgf_files_out.push_back(found_sgfs[i]); jpg_files_out.push_back(found_jpgs[i]); actual_max_step=i;} else break;}
        if (sgf_files_out.empty() && max_step == -1 && bDebug) std::cout << "Debug (loadGameSteps): No numbered SGF/JPG step files found in " << game_folder_path << std::endl;
        else if (sgf_files_out.empty() && max_step > -1) return -1; // Found numbered files but no pairs from 0
        return actual_max_step;
    } catch (const fs::filesystem_error& e) { std::cerr << "FS error loadGameSteps: " << e.what() << std::endl; return -1; }
}

void studyModeWorkflow(const std::string &game_to_study_name) {
  std::cout << "Starting Study Mode for game: " << game_to_study_name << std::endl;

  std::string game_folder_path = "./share/" + game_to_study_name;
  std::vector<std::string> sgf_step_files; // Paths to N.sgf (board state snapshots)
  std::vector<std::string> jpg_step_files; 

  int max_step_idx = loadGameSteps(game_folder_path, sgf_step_files, jpg_step_files);

  if (max_step_idx < 0 || sgf_step_files.empty()) {
    THROWGEMERROR("No game steps (matching SGF/JPG pairs from 0..N) found in folder: " + game_folder_path);
  }
  std::cout << "  Loaded " << max_step_idx + 1 << " game steps (0 to " << max_step_idx << ")." << std::endl;

  std::string main_tournament_sgf_path = game_folder_path + "/tournament.sgf";
  std::ifstream tournament_sgf_stream(main_tournament_sgf_path);
  if (!tournament_sgf_stream.is_open()) {
      THROWGEMERROR("Failed to open main tournament SGF file: " + main_tournament_sgf_path);
  }
  std::stringstream sstream_tournament;
  sstream_tournament << tournament_sgf_stream.rdbuf();
  std::string full_tournament_sgf_content = sstream_tournament.str();
  tournament_sgf_stream.close();
  if (full_tournament_sgf_content.empty()) {
      THROWGEMERROR("Main tournament SGF file is empty: " + main_tournament_sgf_path);
  }

  int current_display_step_idx = 0; 
  cv::Mat simulated_board_image;
  cv::Mat snapshot_image;

  std::string sim_board_window = "Study: Simulated Board - " + game_to_study_name;
  std::string snapshot_window = "Study: Snapshot View - " + game_to_study_name;
  cv::namedWindow(sim_board_window, cv::WINDOW_AUTOSIZE);
  cv::namedWindow(snapshot_window, cv::WINDOW_AUTOSIZE);
  cv::moveWindow(sim_board_window, 50, 50);
  cv::moveWindow(snapshot_window, 50 + canvas_size_px + 20, 50); 

  auto displayCurrentStudyStep = [&]() {
    if (current_display_step_idx < 0 || static_cast<size_t>(current_display_step_idx) >= jpg_step_files.size()) { // Check jpg_files_size as it dictates pairs
        std::cerr << "Error: Invalid step index " << current_display_step_idx << " for display." << std::endl;
        return;
    }

    std::cout << "  Displaying Step: " << current_display_step_idx << std::endl;
    cv::setWindowTitle(sim_board_window, "Study: Sim Board - " + game_to_study_name + " (Step " + std::to_string(current_display_step_idx) + ")");
    cv::setWindowTitle(snapshot_window, "Study: Snapshot - " + game_to_study_name + " (Step " + std::to_string(current_display_step_idx) + ")");

    snapshot_image = cv::imread(jpg_step_files[current_display_step_idx]);
    if (snapshot_image.empty()) {
        std::cerr << "Error loading snapshot: " << jpg_step_files[current_display_step_idx] << std::endl;
        cv::Mat error_img = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::putText(error_img, "Snapshot Load Error", cv::Point(50,240), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,255),2);
        cv::imshow(snapshot_window, error_img);
    } else {
        cv::imshow(snapshot_window, snapshot_image);
    }
    
    // Call drawSimulatedGoBoard with Phase 2 signature
    // current_display_step_idx is the number of B/W moves to replay from tournament.sgf to get to this state
    // (0 for setup, 1 for after first B/W move, etc.)
    // We also highlight the move that *led* to this state.
    drawSimulatedGoBoard(full_tournament_sgf_content, 
                         current_display_step_idx, 
                         simulated_board_image, 
                         current_display_step_idx, // Highlight the current move number
                         canvas_size_px); 
    cv::imshow(sim_board_window, simulated_board_image);
  };

  displayCurrentStudyStep(); 

  while (true) {
    int key = cv::waitKey(0);
    if (key == 27) { break; } 
    else if (key == 'f' || key == 'F') { 
        if (current_display_step_idx < max_step_idx) {
            current_display_step_idx++;
            displayCurrentStudyStep();
        } else { std::cout << "  Already at the last step (" << max_step_idx << ")." << std::endl; }
    } else if (key == 'b' || key == 'B') { 
        if (current_display_step_idx > 0) {
            current_display_step_idx--;
            displayCurrentStudyStep();
        } else { std::cout << "  Already at the first step (0)." << std::endl; }
    }
  }

  cv::destroyAllWindows();
  std::cout << "Study Mode Finished for game: " << game_to_study_name << std::endl;
}


int main(int argc, char *argv[]) {
  try {
    if (argc == 1) {
      displayHelpMessage();
      return 0;
    }
    int option_index = 0;

    std::string snapshot_output;
    std::string record_sgf_output;
    std::string test_perspective_image_path; // For --test-perspective
    std::string draw_board_sgf_path_arg;     // For --draw-board

    bool run_probe_devices = false;
    bool run_calibration = false;
    bool run_interactive_calibration = false;
    bool run_test_calibration = false;
    bool run_draw_board_workflow = false; // Flag for the new workflow
    bool run_tournament_mode = false;     // Flag for new tournament mode
    bool run_study_mode = false;

    struct option long_options[] = {
        {"process-image", required_argument, nullptr, 'p'},
        {"generate-sgf", required_argument, nullptr, 'g'},
        {"verify", required_argument, nullptr, 'v'},
        {"compare", required_argument, nullptr, 'c'},
        {"parse", required_argument, nullptr, 0},
        {"help", no_argument, nullptr, 'h'},
        {"debug", no_argument, nullptr, 'd'},
        {"probe-devices", no_argument, nullptr, 0},
        {"snapshot", required_argument, nullptr, 's'},
        {"device", required_argument, nullptr, 'D'},
        {"record-sgf", required_argument, nullptr, 'r'},
        {"tournament", no_argument, nullptr, 't'},
        {"game-name", required_argument, nullptr, 0},
        {"study", no_argument, nullptr, 'u'},
        {"test-perspective", required_argument, nullptr, 0}, // Add -t option
        {"calibration", no_argument, nullptr,
         'b'}, // Added calibration long option
        {"interactive-calibration", no_argument, nullptr,
         'B'}, // Added calibration long option
        {"mode", required_argument, nullptr, 'M'}, // Added mode option
        {"size", required_argument, nullptr,
         'S'}, // Use S as a unique identifier for --size
        {"draw-board", required_argument, nullptr, 0}, // NEW OPTION
        {"test-calibration-config", no_argument, nullptr,
         'f'}, // Use f as unique value
        {nullptr, 0, nullptr, 0}};

    int c;
    // Process all options in a single loop
    while ((c = getopt_long(argc, argv, "dp:g:v:c:h:s:r:D:BbM:S:ftu",
                            long_options, &option_index)) != -1) {
      switch (c) {
      case 'd':
        bDebug = true;
        cout << "Debug mode enabled." << endl;
        break;
      case 'D': // need to handle more than one digig
      {
        Str2Num num(optarg);
        if (num && num.val() >= 0 && num.val() < 256) {
          auto sz = g_device_path.size() - 1;
          g_device_path.resize(sz);
          g_device_path += optarg;
          cout << "Device:" << g_device_path << endl;
        } else {
          cout << "invalid device number: " << optarg << endl;
        }
      } break;
      case 'b': // Handle calibration option
        run_calibration = true;
        run_interactive_calibration = false;
        break;
      case 'B':
        run_calibration = true;
        run_interactive_calibration = true;
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
      case 'f': // Corresponds to --test-calibration-config
        run_test_calibration = true;
        break;
      case 'h':
        displayHelpMessage();
        return 0;
      case 's':
        snapshot_output = optarg;
        break;
      case 'r':
        record_sgf_output = optarg;
        break;
      case 't': // Handle -t option
        run_tournament_mode = true;
        break;
      case 'u':
        run_study_mode = true;
        break;
      case 0: // Long-only options (val was 0)
        if (long_options[option_index].name == std::string("game-name")) {
          g_default_game_name_prefix = optarg;
          if (g_default_game_name_prefix.empty()) {
            THROWGEMERROR("--game-name option requires a tournament directory "
                          "name argument.");
          }
        } else if (long_options[option_index].name ==
                   std::string("test-calibration-config")) {
          run_test_calibration = true;
        } else if (long_options[option_index].name == std::string("parse")) {
          parseSGFWorkflow(optarg);
          return 0;
        } else if (long_options[option_index].name ==
                   std::string("probe-devices")) {
          run_probe_devices = true;
        } else if (long_options[option_index].name ==
                   std::string("test-perspective")) {
          test_perspective_image_path = optarg;
          if (test_perspective_image_path.empty()) {
            THROWGEMERROR("--test-perspective option requires an image file "
                          "path argument.");
          }
        } else if (long_options[option_index].name ==
                   std::string("draw-board")) { // NEW
          draw_board_sgf_path_arg = optarg;
          if (!draw_board_sgf_path_arg.empty()) {
            run_draw_board_workflow = true;
          } else {
            THROWGEMERROR(
                "--draw-board option requires an SGF file path argument.");
          }
        }
        break;
      case '?':
      default:
        displayHelpMessage();
        return 1;
      }
    }

    // Workflow execution based on flags
    if (run_probe_devices) {
      probeVideoDevicesWorkflow();
    } else if (run_calibration) {
      calibrationWorkflow(run_interactive_calibration);
    } else if (run_test_calibration) {
      testCalibrationConfigWorkflow();
    } else if (run_study_mode) {
      studyModeWorkflow(g_default_game_name_prefix);
    } else if (run_tournament_mode) {
      tournamentModeWorkflow(g_default_game_name_prefix);
    } else if (!snapshot_output.empty()) {
      captureSnapshotWorkflow(snapshot_output);
    } else if (!record_sgf_output.empty()) {
      recordSGFWorkflow(record_sgf_output);
    } else if (run_draw_board_workflow) {
      drawSimulatedBoardWorkflow(draw_board_sgf_path_arg);
    } else if (!test_perspective_image_path.empty()) {
      // Execute if no other primary workflow was triggered by a short option
      // and test-perspective was the only major action specified.
      // This check ensures it doesn't run if, e.g., -p was also given.
      // A more robust way might be to ensure only one "primary workflow" flag
      // is true.
      bool primary_action_taken =
          run_calibration || run_test_calibration || run_tournament_mode ||
          !snapshot_output.empty() || !record_sgf_output.empty() ||
          (argc > optind + 1); // Heuristic: if other positional args for
                               // p,g,v,c were processed

      // This logic for when to execute testPerspectiveTransformWorkflow might
      // need refinement based on how you want to prioritize options. For now,
      // if it's the *only* path remaining and was set:
      int non_opt_args = argc - optind;
      bool other_actions_from_getopt =
          false; // Check if any case apart from '0' (for long-only) and
                 // 'd'/'D'/'M'/'S' was hit. This would require more complex
                 // flag tracking inside the loop.

      // Simple approach for now: if test_perspective_image_path is set and no
      // other *major* workflow flag is true:
      if (!run_calibration && !run_test_calibration && !run_tournament_mode &&
          snapshot_output.empty() && record_sgf_output.empty()) {
        // And ensure no *other* action that consumes optarg was taken after
        // this (e.g. -p path_for_p) This check is tricky with getopt_long as it
        // processes in order. The most reliable way is to have a flag for
        // test_perspective_workflow and check it here.
        testPerspectiveTransformWorkflow(test_perspective_image_path);
      }
    } else if (optind == argc &&
               argc > 1) { // No primary actions left, but options were given
      // This means options like -d, -D, -M, --size might have been given
      // without a primary action. Or a long-only option like --parse was
      // processed, and that's fine. If no workflow was explicitly triggered and
      // no files remain, but options were parsed, it might be an incomplete
      // command. However, getopt handles errors for missing arguments. If we
      // reach here and nothing else was done, it's likely okay or help was
      // already shown.
    } else if (optind < argc) {
      std::cerr << "Error: Unprocessed arguments. What is '" << argv[optind]
                << "'?" << std::endl;
      displayHelpMessage();
      return 1;
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