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
#include <filesystem> // For std::filesystem::remove_all and create_directory (C++17)
#include <sys/stat.h> // For mkdir (though filesystem is preferred)
#include <cerrno>     // For errno

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
  cout << "  -M, --mode <backend>              : Specify capture backend "
          "('v4l2' "
          "or 'opencv', default: v4l2)."
       << endl;
  cout << "  --size <WxH>                      : Specify capture resolution "
          "(e.g., "
          "1280x720). Default: 640x480."
       << endl;
  cout << "  -b, --calibration                 : Run calibration workflow."
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
  cout << "  -t, --tournament-mode             : Start tournament recording "
          "mode."
       << endl;
  cout << "      --game-name <prefix>            : Set game name prefix for "
          "tournament mode (default: "
       << g_default_game_name_prefix << " or 'tournament' if only -t is used)."
       << endl;
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
void calibrationWorkflow() {
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
  runInteractiveCalibration(camera_index);
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

void testPerspectiveTransformWorkflow(const std::string& imagePath) { // Renamed function
  cout << "Testing perspective transform on image: " << imagePath << endl;
  Mat image = imread(imagePath);
  if (image.empty()) {
    cerr << "Error: Could not open image for perspective test: " << imagePath << endl;
    return;
  }
  Mat corrected_image = correctPerspective(image);
  if (bDebug || !corrected_image.empty()) {
      imshow("Original for Perspective Test", image);
      imshow("Corrected Perspective Test", corrected_image);
      waitKey(0);
      destroyAllWindows();
  } else if (corrected_image.empty()){
      cerr << "Error: Perspective correction resulted in an empty image." << endl;
  }
}
static bool createOrClearGameFolder(const std::string& folder_path, bool debug_flag) {
    // 1. Ensure ./share exists (as per README, user should create it with 777)
    //    This function will not attempt to create ./share itself.
    if (!fs::exists("./share")) {
        THROWGEMERROR("Base directory './share' does not exist. Please create it with appropriate permissions (e.g., chmod 777 share).");
    }
    if (!fs::is_directory("./share")) {
        THROWGEMERROR("'./share' exists but is not a directory.");
    }

    // 2. Handle the game-specific folder
    if (fs::exists(folder_path)) {
        if (debug_flag) std::cout << "  Debug: Game folder " << folder_path << " exists. Attempting to remove and recreate..." << std::endl;
        std::error_code ec;
        fs::remove_all(folder_path, ec); // C++17 way to remove directory and its contents
        if (ec) {
            // remove_all might fail if permissions are wrong or files are in use.
            std::cerr << "Warning: Could not completely remove existing game folder '" << folder_path 
                      << "': " << ec.message() << ". Attempting to proceed..." << std::endl;
            // Attempt to create anyway, it might be empty or mkdir might handle it.
        } else {
             if (debug_flag) std::cout << "    Debug: Successfully removed existing folder: " << folder_path << std::endl;
        }
    }

    // 3. Create the new game folder
    std::error_code ec_create;
    fs::create_directory(folder_path, ec_create);
    if (ec_create) {
         THROWGEMERROR("Failed to create game folder: " + folder_path + " - " + ec_create.message());
    }
    if (debug_flag) std::cout << "  Debug: Ensured game folder exists (created if new): " << folder_path << std::endl;
    
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

void tournamentModeWorkflow(const std::string &game_name_final_prefix) {
  std::cout << "Starting Tournament Mode..." << std::endl;
  std::cout << "  Game Name Prefix: " << game_name_final_prefix << std::endl;
  std::cout << "  Device: " << g_device_path << std::endl;

  std::string base_share_path = "./share/"; // Base path, must exist
  std::string game_folder_path = base_share_path + game_name_final_prefix;

  if (!createOrClearGameFolder(game_folder_path, bDebug)) {
    return; // Error already thrown by helper
  }

  std::string main_sgf_path = game_folder_path + "/tournament.sgf";
  std::ofstream main_sgf_file_stream(
      main_sgf_path, std::ios::trunc); // Overwrite existing
  if (!main_sgf_file_stream.is_open()) {
    THROWGEMERROR("Failed to create/overwrite main tournament SGF file: " +
                  main_sgf_path);
  }  

  int game_step_counter = 0;
  cv::Mat previous_board_state_matrix;
  cv::Mat current_raw_frame;
  cv::Mat display_image;

  std::string snapshot_window_name = "Tournament: " + game_name_final_prefix;
  cv::namedWindow(snapshot_window_name, cv::WINDOW_AUTOSIZE);
  bool canExit = false;
  while (!canExit) {
    std::string window_title_current =
        snapshot_window_name + " - Step " + std::to_string(game_step_counter);

    // 1. Capture frame for the current step
    if (!captureFrame(g_device_path, current_raw_frame)) {
      std::cerr << "Error: Failed to capture frame for step "
                << game_step_counter << ". Retrying in 1s..." << std::endl;
      display_image =
          cv::Mat::zeros(cv::Size(g_capture_width, g_capture_height), CV_8UC3);
      cv::putText(display_image, "Capture Failed. Check Camera. ESC to exit.",
                  cv::Point(10, g_capture_height / 2), cv::FONT_HERSHEY_SIMPLEX,
                  0.6, cv::Scalar(0, 0, 255), 1);
      cv::setWindowTitle(snapshot_window_name,
                         window_title_current + " - CAPTURE ERROR");
      cv::imshow(snapshot_window_name, display_image);
      if (cv::waitKey(1000) == 27)
        break;
      continue;
    }

    // 2. Immediately process this frame
    std::cout << "Processing step " << game_step_counter << "..." << std::endl;
    display_image = current_raw_frame.clone();
    cv::putText(display_image,
                "Processing Step " + std::to_string(game_step_counter) + "...",
                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 255), 1);
    cv::setWindowTitle(snapshot_window_name,
                       window_title_current + " - PROCESSING");
    cv::imshow(snapshot_window_name, display_image);
    cv::waitKey(1); // Allow OpenCV to render

    cv::Mat current_board_state_matrix_local;
    cv::Mat board_with_stones_display;
    std::vector<cv::Point2f> intersections;
    bool processing_ok = true;
    try {
      processGoBoard(current_raw_frame, current_board_state_matrix_local,
                     board_with_stones_display, intersections);
    } catch (const std::exception &e) {
      THROWGEMERROR(string("Error processing board for step ") +
                    Num2Str(game_step_counter).str() + e.what());
    }

    if (!current_board_state_matrix_local.empty()) {
      // 3. Save step snapshot (raw frame of this step)
      std::string step_snapshot_path =
          game_folder_path + "/" + std::to_string(game_step_counter) + ".jpg";
      if (!cv::imwrite(step_snapshot_path, current_raw_frame)) {
        THROWGEMERROR(string("Error Failed to save step snapshot ") +
                    step_snapshot_path);       
      } else {
        std::cout << "  Saved step snapshot: " << step_snapshot_path
                  << std::endl;
      }

      // 4. Generate and save step SGF
      std::string current_step_sgf_content_full =
          generateSGF(current_board_state_matrix_local, intersections);
      std::string step_sgf_path =
          game_folder_path + "/" + std::to_string(game_step_counter) + ".sgf";
      std::ofstream step_sgf_file_out(step_sgf_path);
      if (step_sgf_file_out.is_open()) {
        step_sgf_file_out << current_step_sgf_content_full;
        step_sgf_file_out.close();
        std::cout << "  Saved step SGF: " << step_sgf_path << std::endl;
      } else {
        THROWGEMERROR(string("Error: Failed to save step SGF: ") +
                      Num2Str(step_sgf_path).str());
      }

      // 5. Append to main tournament SGF
      // Re-open in append mode for subsequent appends, or use the initial
      // stream for the first write.

      if (game_step_counter == 0) { // First step (initial board setup)
        main_sgf_file_stream
            << generateFirstGameStateSGF(current_step_sgf_content_full);
      } else { // Subsequent steps (game_step_counter > 0)
        std::string move_made = determineSGFMove(
            previous_board_state_matrix, current_board_state_matrix_local);
        main_sgf_file_stream << move_made;
        if (bDebug) {
          std::cout << "  Appended move: " << move_made << " to "
                    << main_sgf_path << std::endl;
        }
      }  
      previous_board_state_matrix = current_board_state_matrix_local.clone();
      display_image = board_with_stones_display.clone(); // Show processed board
      cv::putText(display_image,
                  "Processed Step " + std::to_string(game_step_counter) +
                      ". N or ESC.",
                  cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  cv::Scalar(0, 255, 0), 1);
      cv::setWindowTitle(snapshot_window_name,
                         window_title_current + " - READY FOR NEXT");
      cv::imshow(snapshot_window_name, display_image);
    } // end if processing_ok

    // wait for only escape or 'n' key
    while (true) {
      int post_process_key = cv::waitKey(0);
      if (post_process_key == 27) { // ESC
        std::cout << "Tournament mode saved step " << game_step_counter
                  << " and exiting." << std::endl;
        canExit = true;
        break;
      } else if (post_process_key == 'n' || post_process_key == 'N') {
        game_step_counter++;
        break;
        // Loop will continue and capture a new frame for the new
        // game_step_counter
      } else {
        if (bDebug)
          std::cout << "Ignored key: " << post_process_key
                    << ". Press N or ESC." << std::endl;
      }
    }
  }
  main_sgf_file_stream << ")" << std::endl;
  main_sgf_file_stream.close();  
  std::cout << "  Finalized main tournament SGF: " << main_sgf_path
            << std::endl;


  cv::destroyAllWindows();
  std::cout << "Tournament Mode Finished. Game data saved in: "
            << game_folder_path << std::endl;
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
    bool run_probe_devices = false;
    bool run_calibration = false;
    bool run_test_calibration = false;
    bool run_tournament_mode = false; // Flag for new tournament mode

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
        {"tournament-mode", no_argument, nullptr, 't'},
        {"game-name", required_argument, nullptr, 0},
        {"test-perspective", required_argument, nullptr, 0}, // Add -t option
        {"calibration", no_argument, nullptr,
         'b'}, // Added calibration long option
        {"mode", required_argument, nullptr, 'M'}, // Added mode option
        {"size", required_argument, nullptr,
         'S'}, // Use S as a unique identifier for --size
        {"test-calibration-config", no_argument, nullptr,
         'f'}, // Use f as unique value
        {nullptr, 0, nullptr, 0}};

    int c;
    // Process all options in a single loop
    while ((c = getopt_long(argc, argv, "dp:g:v:c:h:s:r:D:bM:S:ft",
                            long_options, &option_index)) != -1) {
      switch (c) {
      case 'd':
        bDebug = true;
        cout << "Debug mode enabled." << endl;
        break;
      case 'D':
        g_device_path = optarg;
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
      case 0: // Long-only options (val was 0)
        if (long_options[option_index].name == std::string("game-name")) {
            g_default_game_name_prefix = optarg;            
        } else if (long_options[option_index].name == std::string("test-calibration-config")) {
            run_test_calibration = true;
        } else if (long_options[option_index].name == std::string("parse")) {
             parseSGFWorkflow(optarg); return 0;
        } else if (long_options[option_index].name == std::string("probe-devices")) {
             run_probe_devices = true;
        } else if (long_options[option_index].name == std::string("test-perspective")) {
             test_perspective_image_path = optarg;
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
      calibrationWorkflow();
    } else if (run_test_calibration) {
      testCalibrationConfigWorkflow();
    } else if (run_tournament_mode) {
      tournamentModeWorkflow(g_default_game_name_prefix);
    } else if (!snapshot_output.empty()) {
      captureSnapshotWorkflow(snapshot_output);
    } else if (!record_sgf_output.empty()) {
      recordSGFWorkflow(record_sgf_output);
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