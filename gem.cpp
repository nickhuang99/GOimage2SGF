// gem.cpp
#include "common.h"
#include "logger.h"
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
  CONSOLE_OUT << "Go Environment Manager (GEM)" << endl;
  CONSOLE_OUT << "Usage: gem [options]" << endl;
  CONSOLE_OUT << "Options:" << endl;
  CONSOLE_OUT << "  -d, --debug                       : Enable debug log "
                 "output (sets log level to DEBUG)."
              << endl;
  CONSOLE_OUT << "  -O, --log-level <level>           : Set log level (0=NONE, "
                 "1=ERROR, 2=WARNING, 3=INFO, 4=DEBUG). Default: 3 (INFO)."
              << endl; // NEW
  CONSOLE_OUT
      << "  -D, --device <device_path>      : Specify video device by number "
         "(e.g., 0) or path (e.g., /dev/video0). Default: /dev/video0."
      << endl;
  // ... (rest of help message options, using CONSOLE_OUT) ...
  CONSOLE_OUT
      << "  -M, --mode <backend>              : Specify capture backend "
         "('v4l2' "
         "or 'opencv', default: v4l2)."
      << endl;
  CONSOLE_OUT
      << "  --size <WxH>                      : Specify capture resolution "
         "(e.g., "
         "1280x720). Default: 640x480."
      << endl;
  CONSOLE_OUT
      << "  -b, --calibration                 : Run capture calibration "
         "workflow."
      << endl;
  CONSOLE_OUT
      << "  -B, --interactive-calibration     : Run interactive calibration "
         "workflow."
      << endl;
  CONSOLE_OUT
      << "  --test-calibration-config         : Load calibration snapshot and "
         "config, draw corners."
      << endl;
  CONSOLE_OUT
      << "  -p, --process-image <image_path>   : Process the Go board image."
      << endl;
  CONSOLE_OUT
      << "  -P <col> <row>                    : Detect stone at specified "
         "col/row (0-18)."
      << endl; // NEW
  CONSOLE_OUT << "      --image <image_path>          : Optional image for -P "
                 "(default: "
              << g_default_input_image_path << ")." << endl; // NEW
  CONSOLE_OUT
      << "  -g, --generate-sgf <in_img> <out_sgf> : Generate SGF from image."
      << endl;
  CONSOLE_OUT
      << "  -v, --verify <image_path> <sgf_path> : Verify board state against "
         "SGF."
      << endl;
  CONSOLE_OUT
      << "  -c, --compare <sgf1> <sgf2>         : Compare two SGF files."
      << endl;
  CONSOLE_OUT << "  --parse <sgf_path>                  : Parse an SGF file."
              << endl;
  CONSOLE_OUT
      << "  --probe-devices                     : List available video devices."
      << endl;
  CONSOLE_OUT
      << "  -s, --snapshot <output_file>        : Capture a snapshot from the "
         "webcam."
      << endl;
  CONSOLE_OUT
      << "  -r, --record-sgf <output_sgf>        : Capture snapshot, process, "
         "and generate SGF."
      << endl;
  CONSOLE_OUT << "  -t, --tournament             : Start tournament recording "
                 "mode."
              << endl;
  CONSOLE_OUT
      << "      --game-name <prefix>            : Set game name prefix for "
         "tournament or study mode (default: "
      << g_default_game_name_prefix << " or 'tournament' if only -t is used)."
      << endl;
  CONSOLE_OUT
      << "      --draw-board <sgf_path>         : Read SGF, draw and display "
         "simulated board."
      << endl; // NEW
  CONSOLE_OUT << "  -u, --study             : Start study mode to replay a "
                 "recorded game."
              << endl; // NEW
  CONSOLE_OUT
      << "      --test-perspective <image_path> : (Dev) Test perspective "
         "correction." // Removed -t
      << endl;
  CONSOLE_OUT
      << "  -h, --help                          : Display this help message."
      << endl;
  CONSOLE_OUT
      << "\n  Note: Webcam operations may require appropriate permissions "
         "(e.g., user in 'video' group)."
      << endl;
}

void processImageWorkflow(const std::string &imagePath) {
  LOG_INFO << "Processing image: " << imagePath << std::endl;
  cv::Mat image_bgr = imread(imagePath);
  if (image_bgr.empty()) {
    LOG_ERROR << "Could not open or find the image: " << imagePath << std::endl;
    THROWGEMERROR("Could not open or find the image: " + imagePath);
  } else {
    try {
      cv::Mat board_state, board_with_stones;
      std::vector<Point2f> intersection_points;
      processGoBoard(image_bgr, board_state, board_with_stones,
                     intersection_points);
      LOG_INFO << "Image processed successfully. Board state matrix size: "
               << board_state.rows << "x" << board_state.cols << std::endl;

      if (bDebug || Logger::getGlobalLogLevel() >= LogLevel::DEBUG) {
        imshow("processGoBoard Result", board_with_stones);
        LOG_DEBUG << "Displaying processed board. Press any key to continue."
                  << std::endl;
        waitKey(0);
        destroyWindow("processGoBoard Result");
      }
    } catch (const cv::Exception &e) {
      LOG_ERROR << "OpenCV error in processImageWorkflow: " << e.what()
                << std::endl;
      THROWGEMERROR("OpenCV error in processImageWorkflow: " +
                    std::string(e.what()));
    }
  }
}

void generateSGFWorkflow(const std::string &inputImagePath,
                         const std::string &outputSGFPath) {
  LOG_INFO << "Generating SGF from image: " << inputImagePath
           << " to: " << outputSGFPath << std::endl;
  cv::Mat image_bgr = imread(inputImagePath);
  if (image_bgr.empty()) {
    LOG_ERROR << "Could not open or find the input image: " << inputImagePath
              << std::endl;
    THROWGEMERROR("Could not open or find the input image: " + inputImagePath);
  } else {
    try {
      cv::Mat board_state, board_with_stones;
      std::vector<cv::Point2f> intersections;
      processGoBoard(image_bgr, board_state, board_with_stones, intersections);
      std::string sgf_content = generateSGF(board_state, intersections);
      std::ofstream outfile(outputSGFPath);
      if (!outfile.is_open()) {
        LOG_ERROR << "Could not open SGF file for writing: " << outputSGFPath
                  << std::endl;
        THROWGEMERROR("Could not open SGF file for writing: " + outputSGFPath);
      }
      outfile << sgf_content << std::endl;
      outfile.close();
      LOG_INFO << "SGF content written to: " << outputSGFPath << std::endl;
      CONSOLE_OUT << "SGF content written to: " << outputSGFPath << std::endl;
    } catch (const cv::Exception &e) {
      LOG_ERROR << "OpenCV error in generateSGFWorkflow: " << e.what()
                << std::endl;
      THROWGEMERROR("OpenCV error in generateSGFWorkflow: " +
                    std::string(e.what()));
    }
  }
}

void verifySGFWorkflow(const std::string &imagePath,
                       const std::string &sgfPath) {
  LOG_INFO << "Verifying image: " << imagePath << " against SGF: " << sgfPath
           << std::endl;
  cv::Mat image_bgr = imread(imagePath);
  if (image_bgr.empty()) {
    LOG_ERROR << "Could not open or find the image: " << imagePath << std::endl;
    THROWGEMERROR("Could not open or find the image: " + imagePath);
  } else {
    try {
      cv::Mat board_state, board_with_stones; // Not used for display here, but
                                              // processGoBoard populates them
      std::vector<cv::Point2f> intersections;
      processGoBoard(image_bgr, board_state, board_with_stones, intersections);

      std::ifstream infile(sgfPath);
      if (!infile.is_open()) {
        LOG_ERROR << "Could not open SGF file: " << sgfPath << std::endl;
        THROWGEMERROR("Could not open SGF file: " + sgfPath);
      }
      std::stringstream buffer;
      buffer << infile.rdbuf();
      std::string sgf_data = buffer.str();
      if (sgf_data.empty()) {
        LOG_ERROR << "Could not read SGF data from: " << sgfPath << std::endl;
        THROWGEMERROR("Could not read SGF data from: " + sgfPath);
      }
      // verifySGF will now use LOG_XXX for its output and imshow
      verifySGF(image_bgr, sgf_data, intersections);
    } catch (const cv::Exception &e) {
      LOG_ERROR << "OpenCV error in verifySGFWorkflow: " << e.what()
                << std::endl;
      THROWGEMERROR("OpenCV error in verifySGFWorkflow: " +
                    std::string(e.what()));
    }
  }
}

void compareSGFWorkflow(const std::string &sgfPath1,
                        const std::string &sgfPath2) {
  LOG_INFO << "Comparing SGF files: " << sgfPath1 << " and " << sgfPath2
           << std::endl;
  std::ifstream infile1(sgfPath1);
  if (!infile1.is_open()) {
    LOG_ERROR << "Could not open the first SGF file: " << sgfPath1 << std::endl;
    THROWGEMERROR("Could not open the first SGF file: " + sgfPath1);
  }
  std::stringstream buffer1;
  buffer1 << infile1.rdbuf();
  std::string sgf_data1 = buffer1.str();
  if (sgf_data1.empty()) {
    LOG_ERROR << "Could not read SGF data from: " << sgfPath1 << std::endl;
    THROWGEMERROR("Could not read SGF data from: " + sgfPath1);
  }

  std::ifstream infile2(sgfPath2);
  if (!infile2.is_open()) {
    LOG_ERROR << "Could not open the second SGF file: " << sgfPath2
              << std::endl;
    THROWGEMERROR("Could not open the second SGF file: " + sgfPath2);
  }
  std::stringstream buffer2;
  buffer2 << infile2.rdbuf();
  std::string sgf_data2 = buffer2.str();
  if (sgf_data2.empty()) {
    LOG_ERROR << "Could not read SGF data from: " << sgfPath2 << std::endl;
    THROWGEMERROR("Could not read SGF data from: " + sgfPath2);
  }

  if (compareSGF(sgf_data1, sgf_data2)) {
    LOG_INFO << "SGF files comparison result: IDENTICAL." << std::endl;
    CONSOLE_OUT << "SGF files are identical." << std::endl;
  } else {
    LOG_INFO << "SGF files comparison result: DIFFERENT." << std::endl;
    CONSOLE_OUT << "SGF files are different." << std::endl;
  }
}

void parseSGFWorkflow(const std::string &sgfPath) {
  LOG_INFO << "Parsing SGF file: " << sgfPath << std::endl;
  std::ifstream infile(sgfPath);
  if (!infile.is_open()) {
    LOG_ERROR << "Could not open SGF file: " << sgfPath << std::endl;
    THROWGEMERROR("Could not open SGF file: " + sgfPath);
  }
  std::stringstream buffer;
  buffer << infile.rdbuf();
  std::string sgf_content = buffer.str();
  if (sgf_content.empty()) {
    LOG_ERROR << "Could not read SGF data from: " << sgfPath << std::endl;
    THROWGEMERROR("Could not read SGF data from: " + sgfPath);
  }

  std::set<std::pair<int, int>> setupBlack, setupWhite;
  std::vector<Move> moves;
  try {
    parseSGFGame(sgf_content, setupBlack, setupWhite, moves);
    SGFHeader header =
        parseSGFHeader(sgf_content); // parseSGFHeader internal logs errors

    LOG_INFO << "SGF Header Parsed: GM=" << header.gm << ", FF=" << header.ff
             << ", CA=" << header.ca << ", AP=" << header.ap
             << ", SZ=" << header.sz << std::endl;
    CONSOLE_OUT << "SGF Header:" << std::endl;
    CONSOLE_OUT << "  Game: " << header.gm << std::endl;
    CONSOLE_OUT << "  File Format: " << header.ff << std::endl;
    CONSOLE_OUT << "  Character Set: " << header.ca << std::endl;
    CONSOLE_OUT << "  Application: " << header.ap << std::endl;
    CONSOLE_OUT << "  Board Size: " << header.sz << std::endl;

    std::stringstream ss_black, ss_white, ss_moves;
    ss_black << "Setup Black (" << setupBlack.size() << " stones): ";
    for (const auto &stone : setupBlack) {
      ss_black << "(" << stone.first << "," << stone.second << ") ";
    }
    LOG_INFO << ss_black.str() << std::endl;
    CONSOLE_OUT << "\n" << ss_black.str() << std::endl;

    ss_white << "Setup White (" << setupWhite.size() << " stones): ";
    for (const auto &stone : setupWhite) {
      ss_white << "(" << stone.first << "," << stone.second << ") ";
    }
    LOG_INFO << ss_white.str() << std::endl;
    CONSOLE_OUT << ss_white.str() << std::endl;

    LOG_INFO << "Parsed " << moves.size() << " moves." << std::endl;
    CONSOLE_OUT << "\nMoves:" << std::endl;
    for (size_t i = 0; i < moves.size(); ++i) {
      const auto &move = moves[i];
      ss_moves.str("");
      ss_moves.clear(); // Clear stringstream for reuse
      ss_moves << "  Move " << (i + 1) << ": Player=" << move.player
               << ", Pos=(" << move.row << "," << move.col << ")";
      if (!move.capturedStones.empty()) {
        ss_moves << ", Captured(" << move.capturedStones.size() << "): ";
        for (const auto &captured : move.capturedStones) {
          ss_moves << "(" << captured.first << "," << captured.second << ") ";
        }
      }
      LOG_INFO << ss_moves.str() << std::endl;
      CONSOLE_OUT << ss_moves.str() << std::endl;
    }
  } catch (const SGFError &e) {
    LOG_ERROR << "SGF parsing error: " << e.what() << std::endl;
    THROWGEMERROR("SGF parsing error: " + std::string(e.what()));
  }
}

void probeVideoDevicesWorkflow() {
  LOG_INFO << "Probing video devices..." << std::endl;
  std::vector<VideoDeviceInfo> available_devices =
      probeVideoDevices(); // probeVideoDevices uses LOG_XXX

  if (available_devices.empty()) {
    LOG_INFO << "No video capture devices found." << std::endl;
    CONSOLE_OUT << "No video capture devices found." << std::endl;
    return;
  }
  LOG_INFO << "Found " << available_devices.size() << " video capture devices."
           << std::endl;
  CONSOLE_OUT << "Available video capture devices:" << std::endl;
  for (size_t i = 0; i < available_devices.size(); ++i) {
    CONSOLE_OUT << "[" << i << "] Path: " << available_devices[i].device_path
                << "\n  Driver: " << available_devices[i].driver_name
                << "\n  Card: " << available_devices[i].card_name
                << "\n  Capabilities: "
                << getCapabilityDescription(available_devices[i].capabilities)
                << " (0x" << std::hex << available_devices[i].capabilities
                << std::dec << ")" << "\n  Supported Formats & Sizes:";
    if (available_devices[i].supported_format_details.empty()) {
      CONSOLE_OUT << " None listed or error in enumeration.";
    } else {
      for (const std::string &detail :
           available_devices[i].supported_format_details) {
        CONSOLE_OUT << "\n    - " << detail;
      }
    }
    CONSOLE_OUT << "\n\n";
  }
}

void captureSnapshotWorkflow(const std::string &output) {
  LOG_INFO << "Attempting to capture snapshot from: " << g_device_path
           << " using " << (gCaptureMode == MODE_OPENCV ? "OpenCV" : "V4L2")
           << " mode to: " << output << std::endl;
  if (captureSnapshot(g_device_path, output)) { // captureSnapshot uses LOG_XXX
    LOG_INFO << "Snapshot saved to " << output << std::endl;
    CONSOLE_OUT << "Snapshot saved to " << output << std::endl;
  } else {
    LOG_ERROR << "Snapshot capture failed for device " << g_device_path
              << " using mode "
              << (gCaptureMode == MODE_OPENCV ? "OpenCV" : "V4L2")
              << ". Check logs for details." << std::endl;
    // Error already logged by captureSnapshot or its callees
  }
}

void recordSGFWorkflow(const std::string &output_sgf) {
  LOG_INFO << "Capturing snapshot, processing, and generating SGF to: "
           << output_sgf << " from device: " << g_device_path << " using "
           << (gCaptureMode == MODE_OPENCV ? "OpenCV" : "V4L2") << " mode."
           << std::endl;

  Mat captured_image;
  try {
    if (!captureFrame(g_device_path, captured_image)) {
      LOG_ERROR << "Failed to capture frame from device " << g_device_path
                << std::endl;
      THROWGEMERROR("Failed to capture frame from device " + g_device_path);
    }
    LOG_DEBUG << "Frame captured successfully for recordSGFWorkflow."
              << std::endl;

    Mat board_state, board_with_stones;
    std::vector<Point2f> intersections;
    processGoBoard(captured_image, board_state, board_with_stones,
                   intersections);
    LOG_DEBUG << "Board processed for recordSGFWorkflow." << std::endl;

    std::string sgf_content = generateSGF(board_state, intersections);
    LOG_DEBUG << "SGF content generated for recordSGFWorkflow: "
              << sgf_content.substr(0, 50) << "..." << std::endl;

    std::ofstream outfile(output_sgf);
    if (!outfile.is_open()) {
      LOG_ERROR << "Could not open SGF file for writing: " << output_sgf
                << std::endl;
      THROWGEMERROR("Could not open SGF file for writing: " + output_sgf);
    }
    outfile << sgf_content << std::endl;
    outfile.close();
    LOG_INFO << "SGF content written to: " << output_sgf << std::endl;
    CONSOLE_OUT << "SGF content written to: " << output_sgf << std::endl;

  } catch (const cv::Exception &e) {
    LOG_ERROR << "OpenCV error in recordSGFWorkflow: " << e.what() << std::endl;
    THROWGEMERROR("OpenCV error in recordSGFWorkflow: " +
                  std::string(e.what()));
  } catch (const GEMError &e) {
    // Error already logged by THROWGEMERROR's source, but we log the context of
    // this workflow.
    LOG_ERROR << "GEMError in recordSGFWorkflow: " << e.what() << std::endl;
    throw;
  }
}

void calibrationWorkflow(bool bInteractive) {
  LOG_INFO << "Starting Calibration Workflow. Interactive: "
           << (bInteractive ? "Yes" : "No") << std::endl;
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
      LOG_WARN << "Warning: Could not parse camera index from device path '"
               << g_device_path
               << "'. Using default index 0. Error: " << e.what() << endl;
      camera_index = 0;
    }
  } else {
    LOG_WARN << "Warning: Could not find camera index in device path '"
             << g_device_path << "'. Using default index 0." << endl;
    camera_index = 0;
  }

  if (bInteractive) { // Assuming run_interactive_calibration is
                      // set appropriately
    LOG_INFO << "Running INTERACTIVE calibration for camera index "
             << camera_index << " (device: " << g_device_path << ")"
             << std::endl;
    runInteractiveCalibration(camera_index);
  } else {
    LOG_INFO << "Running AUTOMATED calibration using snapshot: "
             << CALIB_SNAPSHOT_PATH << std::endl;
    runCaptureCalibration();
  }
  LOG_INFO << "Calibration workflow finished." << std::endl;
}

void testCalibrationConfigWorkflow() {
  LOG_INFO << "Testing Calibration Config..." << std::endl;
  LOG_INFO << "  Loading config file: " << CALIB_CONFIG_PATH << std::endl;
  CalibrationData data = loadCalibrationData(CALIB_CONFIG_PATH);

  if (!data.corners_loaded) {
    LOG_ERROR << "Failed to load corner data from config file: " +
                     CALIB_CONFIG_PATH
              << std::endl;
    THROWGEMERROR("Failed to load corner data from config file: " +
                  CALIB_CONFIG_PATH);
  }
  if (!data.colors_loaded) {
    LOG_WARN << "Color calibration data (stones) not found or incomplete in "
                "config file."
             << std::endl;
  }
  if (!data.board_color_loaded) {
    LOG_WARN << "Average board color data not found in config file."
             << std::endl;
  }
  if (!data.dimensions_loaded) {
    LOG_WARN << "Image dimensions not found in config file." << std::endl;
  }

  LOG_INFO << "  --- Loaded Calibration Data ---" << std::endl;

  if (data.dimensions_loaded) {
    LOG_INFO << "  Dimensions: " << data.image_width << "x" << data.image_height
             << std::endl;
  }
  LOG_INFO << "  Corners Loaded: " << (data.corners_loaded ? "Yes" : "No")
           << std::endl;
  if (data.corners_loaded) {
    LOG_INFO << "    TL: " << data.corners[0] << std::endl;
    LOG_INFO << "    TR: " << data.corners[1] << std::endl;
    LOG_INFO << "    BR: " << data.corners[2] << std::endl;
    LOG_INFO << "    BL: " << data.corners[3] << std::endl;
  }
  LOG_INFO << "  Colors Loaded: " << (data.colors_loaded ? "Yes" : "No")
           << std::endl;
  if (data.colors_loaded) {
    LOG_INFO << std::fixed << std::setprecision(1);
    LOG_INFO << "    TL Lab: [" << data.lab_tl[0] << "," << data.lab_tl[1]
             << "," << data.lab_tl[2] << "]" << std::endl;
    LOG_INFO << "    TR Lab: [" << data.lab_tr[0] << "," << data.lab_tr[1]
             << "," << data.lab_tr[2] << "]" << std::endl;
    LOG_INFO << "    BL Lab: [" << data.lab_bl[0] << "," << data.lab_bl[1]
             << "," << data.lab_bl[2] << "]" << std::endl;
    LOG_INFO << "    BR Lab: [" << data.lab_br[0] << "," << data.lab_br[1]
             << "," << data.lab_br[2] << "]" << std::endl;
  }
  LOG_INFO << "  -------------------------------" << std::endl;

  // 2. Determine snapshot file path and load image
  std::string snapshot_path_to_load = CALIB_SNAPSHOT_RAW_PATH;
  LOG_INFO << "  Loading raw calibration snapshot image: "
           << snapshot_path_to_load << std::endl;
  cv::Mat image = cv::imread(snapshot_path_to_load);
  if (image.empty()) {
    LOG_ERROR << "Failed to load calibration raw snapshot image: "
              << snapshot_path_to_load << std::endl;
    THROWGEMERROR("Failed to load calibration raw snapshot image: " +
                  snapshot_path_to_load);
  }
  if (data.dimensions_loaded &&
      (image.cols != data.image_width || image.rows != data.image_height)) {
    LOG_WARN << "Snapshot image dimensions (" << image.cols << "x" << image.rows
             << ") mismatch config dimensions (" << data.image_width << "x"
             << data.image_height << "). Markers might be offset." << std::endl;
  }

  // 3. Draw markers on image using loaded corners
  LOG_INFO << "  Drawing markers on snapshot..." << std::endl;
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
    LOG_WARN << "  Warning: Cannot draw markers as corners were not loaded."
             << std::endl;
  }

  // 4. Display the image
  std::string window_title = "Calibration Test Verification";
  cv::imshow(window_title, image);
  LOG_INFO << "  Displaying image. Press any key or close window to exit."
           << std::endl;
  cv::waitKey(0);
  cv::destroyAllWindows();

  LOG_INFO << "Calibration Test Finished." << std::endl;
}

void testPerspectiveTransformWorkflow(const std::string &imagePath) {
  LOG_INFO << "Testing perspective transform on image: " << imagePath
           << std::endl;
  Mat image = imread(imagePath);
  if (image.empty()) {
    LOG_ERROR << "Could not open image for perspective test: " << imagePath
              << std::endl;
    THROWGEMERROR("Could not open image for perspective test: " + imagePath);
  }
  Mat corrected_image = correctPerspective(image);
  if (corrected_image.empty()) {
    LOG_ERROR << "Perspective correction resulted in an empty image for: "
              << imagePath << std::endl;
    THROWGEMERROR("Perspective correction resulted in an empty image for " +
                  imagePath);
  }
  LOG_INFO << "Perspective correction test completed." << std::endl;
  if (bDebug || Logger::getGlobalLogLevel() >= LogLevel::DEBUG) {
    imshow("Original for Perspective Test", image);
    imshow("Corrected Perspective Test", corrected_image);
    LOG_DEBUG << "Displaying perspective test images. Press any key."
              << std::endl;
    waitKey(0);
    destroyAllWindows();
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
      LOG_DEBUG << "  Debug: Game folder " << folder_path
                << " exists. Attempting to remove and recreate..." << std::endl;
    std::error_code ec;
    fs::remove_all(folder_path,
                   ec); // C++17 way to remove directory and its contents
    if (ec) {
      // remove_all might fail if permissions are wrong or files are in use.
      LOG_ERROR << "Warning: Could not completely remove existing game folder '"
                << folder_path << "': " << ec.message()
                << ". Attempting to proceed..." << std::endl;
      // Attempt to create anyway, it might be empty or mkdir might handle it.
    } else {
      if (debug_flag)
        LOG_DEBUG << "    Debug: Successfully removed existing folder: "
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
    LOG_DEBUG << "  Debug: Ensured game folder exists (created if new): "
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
    LOG_INFO << "  generateFirstGameStateSGF: " << content_for_tournament_start
             << std::endl;
    return content_for_tournament_start;
  }
  THROWGEMERROR("generateFirstGameStateSGF fails to parse game state:" +
                current_step_sgf_content_full)
}

static bool setupCalibrationFromConfig() {
  LOG_INFO << "Attempting to load and apply calibration settings from: "
           << CALIB_CONFIG_PATH << std::endl;
  CalibrationData data = loadCalibrationData(CALIB_CONFIG_PATH);

  if (!data.device_path_loaded || data.device_path.empty()) {
    LOG_ERROR << "Required 'DevicePath' not found in " << CALIB_CONFIG_PATH
              << ". Please run calibration first." << std::endl;
    THROWGEMERROR("Required 'DevicePath' not found in " + CALIB_CONFIG_PATH);
  }
  if (!data.dimensions_loaded || data.image_width <= 0 ||
      data.image_height <= 0) {
    LOG_ERROR << "Tournament mode requires 'ImageWidth' and 'ImageHeight' in "
              << CALIB_CONFIG_PATH << ". Please run calibration first."
              << std::endl;
    THROWGEMERROR(
        "Tournament mode requires 'ImageWidth' and 'ImageHeight' in " +
        CALIB_CONFIG_PATH + ". Please run calibration first.");
  }
  if (!data.corners_loaded) {
    LOG_ERROR << "Tournament mode requires corner data (e.g. TL_X_PX) in "
              << CALIB_CONFIG_PATH << ". Please run calibration first."
              << std::endl;
    THROWGEMERROR("Tournament mode requires corner data (e.g. TL_X_PX) in " +
                  CALIB_CONFIG_PATH + ". Please run calibration first.");
  }
  if (!data.colors_loaded || !data.board_color_loaded) {
    LOG_ERROR << "Tournament mode requires color calibration data (stones and "
                 "board average) in "
              << CALIB_CONFIG_PATH << ". Please run calibration first."
              << std::endl;
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

  LOG_INFO << "Successfully loaded and applied calibration settings."
           << std::endl;
  LOG_INFO << "  Effective Device Path: " << g_device_path << " (from config)"
           << std::endl;
  LOG_INFO << "  Effective Resolution: " << g_capture_width << "x"
           << g_capture_height << " (from config)" << std::endl;
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
    LOG_WARN << "    WARNING: Command-line specified device '"
             << original_g_device_path
             << "' is overridden by calibrated device '" << g_device_path
             << "' from config for tournament mode." << std::endl;
  }
  if (cmd_line_size_differs) {
    LOG_WARN << "    WARNING: Command-line specified size '"
             << original_g_capture_width << "x" << original_g_capture_height
             << "' is overridden by calibrated size '" << g_capture_width << "x"
             << g_capture_height << "' from config for tournament mode."
             << std::endl;
  }

  return true;
}

static void debugAndShowUserIdenticalBoardState(
    const cv::Mat &raw_frame,
    const cv::Mat
        &previous_processed_board_display, // The display of the previous state
    const cv::Mat &current_processed_board_display, // The display of the
                                                    // current (identical) state
    int step_num,
    const std::string
        &window_name // To potentially display messages or new images
) {
  LOG_INFO << "  DEBUG_INFO: Board state for step " << step_num
           << " is identical to the previous step." << std::endl;
  LOG_INFO << "  User may have forgotten to make a move, or processGoBoard "
              "failed to detect a change."
           << std::endl;
  LOG_INFO << "  Displaying current processed board. Press 'N' to try "
              "capturing for this step again, or 'ESC' to exit."
           << std::endl;

  // For now, this function will just print. You can expand it to show images in
  // new windows or overlay messages on the existing 'snapshot_window_name'.
  // Example:
  // cv::Mat combined_display;
  // if (!previous_processed_board_display.empty() &&
  // !current_processed_board_display.empty()) {
  //    cv::hconcat(previous_processed_board_display,
  //    current_processed_board_display, combined_display); cv::imshow("Debug:
  //    Prev vs Current (Identical)", combined_display);
  // } else {
  //    cv::imshow("Debug: Current Processed (Identical to Prev)",
  //    current_processed_board_display.empty() ? raw_frame :
  //    current_processed_board_display);
  // }
  // cv::waitKey(1); // Give a moment for display
}

static bool compareBoardStates(const cv::Mat &board1, const cv::Mat &board2) {
  if (board1.empty() && board2.empty())
    return true;
  if (board1.empty() || board2.empty())
    return false;
  if (board1.rows != board2.rows || board1.cols != board2.cols ||
      board1.type() != board2.type()) {
    return false; // Different dimensions or types
  }
  cv::Mat diff;
  cv::compare(board1, board2, diff,
              cv::CMP_NE); // Stores non-zero if elements are different
  return cv::countNonZero(diff) == 0;
}
// gem.cpp
// ... (other includes and global variables remain the same) ...

// Ensure these helper functions are available or defined as before
// static bool createOrClearGameFolder(const std::string &folder_path, bool
// debug_flag); static bool setupCalibrationFromConfig(); // Assumed to be
// called before tournamentModeWorkflow if needed by main() static void
// debugAndShowUserIdenticalBoardState(...); static bool
// compareBoardStates(...); processGoBoard, generateSGF, determineSGFMove,
// validateSGgfMove, drawSimulatedGoBoard etc. are external

void tournamentModeWorkflow(const std::string &game_name_final_prefix) {
  LOG_INFO << "Starting Tournament Mode (Refactored Flow)..." << std::endl;
  LOG_INFO << "  Game Name Prefix: " << game_name_final_prefix << std::endl;
  LOG_INFO << "  Using Device: " << g_device_path << " at " << g_capture_width
           << "x" << g_capture_height << std::endl;

  std::string base_share_path = "./share/";
  std::string game_folder_path = base_share_path + game_name_final_prefix;

  if (!createOrClearGameFolder(game_folder_path, bDebug)) { //
    return;
  }

  std::string main_sgf_path = game_folder_path + "/tournament.sgf";

  int game_step_counter = 0;
  int previous_move_color = EMPTY;
  cv::Mat previous_board_state_matrix;
  cv::Mat current_raw_frame;
  cv::Mat current_board_state_matrix_local; // Used for current processing
  cv::Mat simulated_board_display_image;    // Main display for simulated board
  cv::Mat processed_camera_capture_display_debug; // For debug window
  cv::Mat previous_processed_camera_capture_display_debug;

  std::string main_display_window_name =
      "Go Tournament: " + game_name_final_prefix + " (Simulated Board)";
  cv::namedWindow(main_display_window_name, cv::WINDOW_AUTOSIZE);

  std::string debug_capture_window_name = "Debug: Processed Camera View";
  if (bDebug) {
    cv::namedWindow(debug_capture_window_name, cv::WINDOW_AUTOSIZE);
    cv::moveWindow(main_display_window_name, 50, 50);
    cv::moveWindow(debug_capture_window_name, 50 + canvas_size_px + 20, 50);
  }

  auto parseSgfMoveNode =
      [](const std::string &sgf_move_node_str) -> std::tuple<int, int, int> { //
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

  std::string accumulated_tournament_sgf_content = "";
  std::vector<cv::Point2f> intersections; // Keep it accessible

  // --- Step 0: Initial Board Setup (Outside the main game loop) ---
  LOG_INFO << "Setting up initial board (Step 0)..." << std::endl;
  cv::setWindowTitle(main_display_window_name,
                     main_display_window_name + " - Step 0 (Initial Setup)");
  if (bDebug) {
    cv::setWindowTitle(debug_capture_window_name,
                       debug_capture_window_name + " - Step 0 (Initial Setup)");
  }

  bool step0_ok = false;
  while (
      !step0_ok) { // Loop until step 0 is successfully processed or user exits
    if (!captureFrame(g_device_path, current_raw_frame)) { //
      LOG_ERROR
          << "Error: Failed to capture frame for Step 0. Retrying in 1s..."
          << std::endl;
      cv::Mat error_disp =
          cv::Mat::zeros(cv::Size(canvas_size_px, canvas_size_px), CV_8UC3);
      cv::putText(error_disp,
                  "Capture Failed (Step 0). Check Camera. ESC to exit.",
                  cv::Point(50, error_disp.rows / 2), cv::FONT_HERSHEY_SIMPLEX,
                  0.8, cv::Scalar(0, 0, 255), 2);
      cv::imshow(main_display_window_name, error_disp);
      if (cv::waitKey(1000) == 27) {
        cv::destroyAllWindows();
        return;
      }
      continue; // Retry capture
    }

    if (bDebug && !current_raw_frame.empty()) {
      cv::imshow(debug_capture_window_name, current_raw_frame); // Show raw
      cv::waitKey(1);
    }

    LOG_INFO << "Processing Step 0..." << std::endl;
    cv::Mat temp_step0_display = current_raw_frame.clone();
    cv::putText(temp_step0_display, "Processing Initial Board (Step 0)...",
                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 255), 2);
    cv::imshow(main_display_window_name, temp_step0_display);
    cv::waitKey(1);

    try {
      processGoBoard(current_raw_frame, current_board_state_matrix_local,
                     processed_camera_capture_display_debug, intersections); //
      if (bDebug && !processed_camera_capture_display_debug.empty()) {
        cv::imshow(debug_capture_window_name,
                   processed_camera_capture_display_debug);
      }
      step0_ok = true; // Mark as successful processing
    } catch (const std::exception &e) {
      LOG_ERROR << "Error processing initial board (Step 0): " << e.what()
                << std::endl;
      cv::Mat error_disp =
          current_raw_frame.empty()
              ? cv::Mat::zeros(cv::Size(canvas_size_px, canvas_size_px),
                               CV_8UC3)
              : current_raw_frame.clone();
      cv::putText(error_disp,
                  "Processing Error (Step 0). ESC to exit, N to retry.",
                  cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                  cv::Scalar(0, 0, 255), 2);
      cv::imshow(main_display_window_name, error_disp);
      if (bDebug && !processed_camera_capture_display_debug.empty()) {
        cv::imshow(debug_capture_window_name,
                   processed_camera_capture_display_debug);
      } else if (bDebug && !current_raw_frame.empty()) {
        cv::imshow(debug_capture_window_name, current_raw_frame);
      }

      int key_retry = cv::waitKey(0);
      if (key_retry == 27) {
        cv::destroyAllWindows();
        return;
      }
      // If 'N' or any other key, loop will retry Step 0 processing.
      step0_ok = false; // Ensure retry
      continue;
    }
  }

  // Save snapshot and SGF for Step 0
  std::string step0_snapshot_path = game_folder_path + "/0.jpg";
  if (!cv::imwrite(step0_snapshot_path, current_raw_frame)) {
    LOG_ERROR << "Error saving snapshot for Step 0 to " << step0_snapshot_path
              << std::endl;
  } else {
    LOG_INFO << "  Saved initial snapshot: " << step0_snapshot_path
             << std::endl;
  }

  std::string step0_sgf_content_full =
      generateSGF(current_board_state_matrix_local, intersections); //
  std::string step0_sgf_path = game_folder_path + "/0.sgf";
  std::ofstream step0_sgf_file_out(step0_sgf_path);
  if (step0_sgf_file_out.is_open()) {
    step0_sgf_file_out << step0_sgf_content_full;
    step0_sgf_file_out.close();
    LOG_INFO << "  Saved initial SGF: " << step0_sgf_path << std::endl;
  } else {
    LOG_ERROR << "Error saving SGF for Step 0 to " << step0_sgf_path
              << std::endl;
  }

  // Initialize main tournament SGF
  std::ofstream tournament_sgf_out(main_sgf_path, std::ios::trunc);
  if (!tournament_sgf_out.is_open()) {
    LOG_ERROR << "Failed to open main SGF for initial write: " << main_sgf_path
              << endl;
    THROWGEMERROR("Failed to open main SGF for initial write: " +
                  main_sgf_path);
  }
  size_t last_paren_pos = step0_sgf_content_full.rfind(')');
  std::string content_for_tournament_start;
  if (last_paren_pos != std::string::npos) {
    content_for_tournament_start =
        step0_sgf_content_full.substr(0, last_paren_pos);
    while (!content_for_tournament_start.empty() &&
           (content_for_tournament_start.back() == '\n' ||
            content_for_tournament_start.back() == '\r')) {
      content_for_tournament_start.pop_back();
    }
  } else { // Fallback if SGF format is unexpected
    content_for_tournament_start =
        "(;FF[4]GM[1]SZ[19]AP[GEM:ErrorFormattingStep0]";
  }
  tournament_sgf_out << content_for_tournament_start;
  accumulated_tournament_sgf_content = content_for_tournament_start; //
  tournament_sgf_out.close();
  LOG_INFO << "  Initialized " << main_sgf_path << " with initial board state."
           << std::endl;

  previous_board_state_matrix = current_board_state_matrix_local.clone(); //
  if (bDebug) {
    previous_processed_camera_capture_display_debug =
        processed_camera_capture_display_debug.clone(); //
  }
  previous_move_color = EMPTY; // Initial state

  drawSimulatedGoBoard(accumulated_tournament_sgf_content + ")", 0,
                       simulated_board_display_image, 0, canvas_size_px); //
  cv::putText(simulated_board_display_image,
              "Step 0: Initial Board. Press 'N' for 1st move, 'ESC' to Exit.",
              cv::Point(10, simulated_board_display_image.rows - 15),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 100, 0), 1,
              cv::LINE_AA);
  cv::imshow(main_display_window_name, simulated_board_display_image);
  if (bDebug && !processed_camera_capture_display_debug.empty()) {
    cv::imshow(debug_capture_window_name,
               processed_camera_capture_display_debug);
  }

  int key_to_start_game = cv::waitKey(0);
  if (key_to_start_game == 27) { // ESC
    LOG_INFO << "Exiting tournament mode before first move." << std::endl;
    // Finalize SGF with just the initial setup
    std::ofstream main_sgf_final_stream(main_sgf_path, std::ios::app);
    if (main_sgf_final_stream.is_open()) {
      if (!accumulated_tournament_sgf_content.empty() &&
          accumulated_tournament_sgf_content.back() !=
              ')') { // Ensure it's not already closed by error fallback
        main_sgf_final_stream << "\n)" << std::endl;
      }
      main_sgf_final_stream.close();
    }
    cv::destroyAllWindows();
    return;
  }
  // Any other key apart from ESC, we assume 'N' like behavior to proceed.
  // For stricter 'N' check: if (key_to_start_game != 'n' && key_to_start_game
  // != 'N') { /* handle exit */ }

  game_step_counter = 1; // Now ready for the first actual move

  // --- Main Game Loop (Step 1 onwards) ---
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

    // Display current state (simulated board from *previous* successful step)
    // The accumulated_tournament_sgf_content is from the end of the *last
    // successful* step. We need to draw the board state up to game_step_counter
    // - 1, highlighting move game_step_counter -1. If game_step_counter is 1,
    // we display state after 0 moves (setup), highlighting nothing or the setup
    // itself.
    int display_up_to = game_step_counter > 0 ? game_step_counter - 1 : 0;
    int highlight_idx = game_step_counter > 0 ? game_step_counter - 1
                                              : -1; // Highlight previous move

    std::string sgf_for_current_display = accumulated_tournament_sgf_content;
    if (!sgf_for_current_display.empty() &&
        sgf_for_current_display.find('(') == 0 &&
        sgf_for_current_display.back() != ')') {
      sgf_for_current_display += ")";
    } else if (sgf_for_current_display.empty() && game_step_counter == 0 &&
               !step0_sgf_content_full.empty()) {
      // This case should primarily be for step 0 display if we were to include
      // it in a generic display func
      sgf_for_current_display = step0_sgf_content_full;
    }

    drawSimulatedGoBoard(sgf_for_current_display, display_up_to,
                         simulated_board_display_image, highlight_idx,
                         canvas_size_px); //
    cv::putText(simulated_board_display_image,
                "Make move for " + current_step_info_str +
                    ", then press 'N'. ESC to Exit.",
                cv::Point(10, simulated_board_display_image.rows - 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 150, 0), 1,
                cv::LINE_AA);
    cv::imshow(main_display_window_name, simulated_board_display_image);

    if (bDebug) { // Show previous step's processed capture in debug window
                  // while waiting
      if (!previous_processed_camera_capture_display_debug.empty()) {
        cv::imshow(debug_capture_window_name,
                   previous_processed_camera_capture_display_debug);
      } else if (!current_raw_frame.empty() &&
                 game_step_counter ==
                     1) { // For first move, current_raw_frame is from step 0
        cv::imshow(debug_capture_window_name, current_raw_frame);
      }
    }

    int key_input = cv::waitKey(0);

    if (key_input == 27) { // ESC
      LOG_INFO << "Tournament mode finishing after "
               << (game_step_counter > 0
                       ? "Step " + std::to_string(game_step_counter - 1)
                       : "initial setup")
               << "." << std::endl;
      break;
    } else if (key_input == 'n' || key_input == 'N') {
      // User has made their move and pressed 'N'. Now capture and process.
      LOG_INFO << "Capturing and processing " << current_step_info_str << "..."
               << std::endl;

      if (!captureFrame(g_device_path, current_raw_frame)) { //
        LOG_ERROR << "Error: Failed to capture frame for "
                  << current_step_info_str << ". Please try again."
                  << std::endl;
        // Display error on main window, keep simulated board as is
        cv::Mat temp_error_display = simulated_board_display_image.clone();
        cv::putText(temp_error_display,
                    "Capture Failed for " + current_step_info_str +
                        ". Try 'N' again.",
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 0, 255), 2);
        cv::imshow(main_display_window_name, temp_error_display);
        cv::waitKey(1); // Show message briefly
        continue;       // Go back to wait for 'N' for the same step
      }

      if (bDebug && !current_raw_frame.empty()) {
        cv::Mat temp_raw_disp = current_raw_frame.clone();
        cv::putText(temp_raw_disp, "Raw Capture: " + current_step_info_str,
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(255, 255, 0), 2);
        cv::imshow(debug_capture_window_name, temp_raw_disp);
        cv::waitKey(1);
      }

      bool processing_ok_current_attempt = true;
      try {
        processGoBoard(current_raw_frame, current_board_state_matrix_local,
                       processed_camera_capture_display_debug,
                       intersections); //
        if (bDebug && !processed_camera_capture_display_debug.empty()) {
          cv::imshow(debug_capture_window_name,
                     processed_camera_capture_display_debug);
        }

        if (!validateSGgfMove(previous_board_state_matrix,
                              current_board_state_matrix_local,
                              previous_move_color)) { //
          LOG_WARN << "  WARNING: Invalid move or no change detected for "
                   << current_step_info_str << "." << std::endl;
          processing_ok_current_attempt = false;
          debugAndShowUserIdenticalBoardState(
              current_raw_frame,
              previous_processed_camera_capture_display_debug,
              processed_camera_capture_display_debug, game_step_counter,
              main_display_window_name); //
        }
      } catch (const std::exception &e) {
        LOG_ERROR << "Error processing board for " << current_step_info_str
                  << ": " << e.what() << std::endl;
        processing_ok_current_attempt = false;
        cv::Mat error_display_on_main =
            simulated_board_display_image
                .clone(); // Show on top of last good simulated board
        cv::putText(error_display_on_main,
                    "Processing Error! " + current_step_info_str,
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 0, 255), 2);
        cv::imshow(main_display_window_name, error_display_on_main);
        if (bDebug && !processed_camera_capture_display_debug
                           .empty()) { // Show the problematic processed view
          cv::imshow(debug_capture_window_name,
                     processed_camera_capture_display_debug);
        } else if (bDebug &&
                   !current_raw_frame
                        .empty()) { // Or raw if processing failed early
          cv::imshow(debug_capture_window_name, current_raw_frame);
        }
      }

      if (processing_ok_current_attempt) {
        std::string step_snapshot_path =
            game_folder_path + "/" + std::to_string(game_step_counter) + ".jpg";
        if (!cv::imwrite(step_snapshot_path, current_raw_frame)) { /* cerr */
        } else {
          LOG_INFO << "  Saved step snapshot: " << step_snapshot_path
                   << std::endl;
        }

        std::string current_step_sgf_data =
            generateSGF(current_board_state_matrix_local, intersections); //
        std::string step_sgf_path =
            game_folder_path + "/" + std::to_string(game_step_counter) + ".sgf";
        std::ofstream step_sgf_file(step_sgf_path);
        if (step_sgf_file.is_open()) {
          step_sgf_file << current_step_sgf_data;
          step_sgf_file.close();
          LOG_INFO << "  Saved step SGF: " << step_sgf_path << std::endl;
        } else { /* cerr */
        }

        std::string move_made_sgf_node = "";
        try {
          move_made_sgf_node = determineSGFMove(
              previous_board_state_matrix, current_board_state_matrix_local); //
        } catch (const GEMError &e_sgf) {                                     //
          std::string error_what = e_sgf.what();
          if (error_what.find("ERROR: Invalid move detected!") !=
              std::string::npos) { //
            LOG_ERROR << "  No change detected (determineSGFMove error). No "
                         "SGF move appended."
                      << std::endl; //
          } else {
            throw;
          } //
        }

        int last_move_r_to_highlight = -1, last_move_c_to_highlight = -1,
            last_move_clr_to_highlight = EMPTY; //

        if (!move_made_sgf_node.empty()) {
          std::ofstream main_sgf_appender(main_sgf_path, std::ios::app);
          if (!main_sgf_appender.is_open()) { /* cerr */
          } else {
            main_sgf_appender << "\n" << move_made_sgf_node;                 //
            accumulated_tournament_sgf_content += "\n" + move_made_sgf_node; //
            LOG_INFO << "  Appended move: " << move_made_sgf_node << " to "
                     << main_sgf_path << std::endl; //
            std::tie(last_move_r_to_highlight, last_move_c_to_highlight,
                     last_move_clr_to_highlight) =
                parseSgfMoveNode(move_made_sgf_node);         //
            previous_move_color = last_move_clr_to_highlight; //
            main_sgf_appender.close();
          }
        } else {
          LOG_INFO << "  No valid move node by determineSGFMove. Nothing "
                      "appended to main SGF."
                   << std::endl; //
          // previous_move_color remains unchanged from the last valid move
        }

        previous_board_state_matrix =
            current_board_state_matrix_local.clone(); //
        if (bDebug) {
          previous_processed_camera_capture_display_debug =
              processed_camera_capture_display_debug.clone(); //
        }

        // Draw the board *after* this successful step to show for next prompt
        std::string sgf_content_for_draw = accumulated_tournament_sgf_content;
        if (!sgf_content_for_draw.empty() &&
            sgf_content_for_draw.find('(') == 0 &&
            sgf_content_for_draw.back() != ')') {
          sgf_content_for_draw += ")";
        }
        drawSimulatedGoBoard(sgf_content_for_draw, game_step_counter,
                             simulated_board_display_image, game_step_counter,
                             canvas_size_px); //
        cv::imshow(main_display_window_name, simulated_board_display_image);
        if (bDebug && !processed_camera_capture_display_debug.empty()) {
          cv::imshow(debug_capture_window_name,
                     processed_camera_capture_display_debug);
        }

        game_step_counter++; // Advance to the next step
      } else {               // processing_ok_current_attempt was false
        LOG_INFO << "  " << current_step_info_str
                 << " had an issue. Please correct the board and press 'N' to "
                    "retry, or 'ESC' to exit."
                 << std::endl;
        // Do not increment game_step_counter. The loop will re-prompt for the
        // same step. Show the simulated board from the *previous good state*
        std::string sgf_content_for_error_display =
            accumulated_tournament_sgf_content;
        if (!sgf_content_for_error_display.empty() &&
            sgf_content_for_error_display.find('(') == 0 &&
            sgf_content_for_error_display.back() != ')') {
          sgf_content_for_error_display += ")";
        }
        drawSimulatedGoBoard(sgf_content_for_error_display,
                             game_step_counter - 1,
                             simulated_board_display_image,
                             game_step_counter - 1, canvas_size_px);

        cv::putText(simulated_board_display_image,
                    "Error processing " + current_step_info_str +
                        ". Retry 'N' or 'ESC'.",
                    cv::Point(10, simulated_board_display_image.rows - 15),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 150), 1,
                    cv::LINE_AA);
        cv::imshow(main_display_window_name, simulated_board_display_image);
        if (bDebug && !processed_camera_capture_display_debug
                           .empty()) { // Show the problematic processed view
          cv::imshow(debug_capture_window_name,
                     processed_camera_capture_display_debug);
        } else if (bDebug &&
                   !current_raw_frame
                        .empty()) { // Or raw if processing failed earlier
          cv::imshow(debug_capture_window_name, current_raw_frame);
        }
      }
    } else { // Ignored key
      LOG_DEBUG << "Ignored key: " << key_input << ". Press N or ESC."
                << std::endl;
    }

    // Finalize SGF by adding closing parenthesis
    if (!accumulated_tournament_sgf_content.empty()) {
      std::ofstream main_sgf_final_stream(main_sgf_path, std::ios::app);
      if (main_sgf_final_stream.is_open()) {
        if (accumulated_tournament_sgf_content.find('(') == 0 &&
            accumulated_tournament_sgf_content.back() != ')') {
          main_sgf_final_stream << "\n)" << std::endl; //
        } else if (accumulated_tournament_sgf_content.find('(') != 0) {
          // This case implies the SGF was not even initialized properly.
          // It might be better to ensure accumulated_tournament_sgf_content
          // always starts with "(". For now, let's assume if it's not empty and
          // doesn't start with '(', something else went wrong.
        }
        main_sgf_final_stream.close();
        LOG_INFO << "  Finalized main tournament SGF: " << main_sgf_path
                 << std::endl;
      } else {
        LOG_ERROR
            << "CRITICAL ERROR: Failed to open main SGF for final closing: "
            << main_sgf_path << std::endl;
      }
    }

    cv::destroyAllWindows();
    LOG_INFO << "Tournament Mode Finished. Game data saved in: "
             << game_folder_path << std::endl;
  }
}
// Corrected drawSimulatedBoardWorkflow
void drawSimulatedBoardWorkflow(const std::string &sgf_file_path) {
  LOG_INFO << "Starting Draw Simulated Board Workflow..." << std::endl;
  LOG_INFO << "  SGF File: " << sgf_file_path << std::endl;

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

  std::set<std::pair<int, int>> setupB, setupW;
  std::vector<Move> moves;
  SGFHeader header;
  try {
    header = parseSGFHeader(sgf_content);
    parseSGFGame(sgf_content, setupB, setupW, moves);
  } catch (const SGFError &e) {
    THROWGEMERROR("SGF Parsing error in drawSimulatedBoardWorkflow: " +
                  std::string(e.what()));
  }

  int total_bw_moves = 0;
  for (const auto &move : moves) {
    if (move.player == BLACK || move.player == WHITE) {
      total_bw_moves++;
    }
  }

  // Corrected call to drawSimulatedGoBoard:
  // Arg1: sgf_content (string)
  // Arg2: display_up_to_move_idx (int) - show all B/W moves
  // Arg3: board_image (Mat&)
  // Arg4: highlight_this_move_idx (int) - highlight the last B/W move
  // Arg5: canvas_size_px (int)
  drawSimulatedGoBoard(sgf_content, total_bw_moves, board_image, total_bw_moves,
                       canvas_size_px);

  if (board_image.empty()) {
    THROWGEMERROR("drawSimulatedGoBoard function returned an empty image.");
  }

  std::string window_title =
      Default_Go_Board_Window_Title + ": " + sgf_file_path;
  cv::imshow(window_title, board_image);
  LOG_INFO << "  Displaying simulated board. Press any key in the OpenCV "
              "window to close."
           << std::endl;
  cv::waitKey(0);
  cv::destroyAllWindows();
  LOG_INFO << "Draw Simulated Board Workflow Finished." << std::endl;
}

// Corrected studyModeWorkflow (Phase 1 structure, calls to
// drawSimulatedGoBoard updated for Phase 2 signature)
static int loadGameSteps(const std::string &game_folder_path,
                         std::vector<std::string> &sgf_files_out,
                         std::vector<std::string> &jpg_files_out) {
  // ... (Implementation from your latest gem.cpp)
  sgf_files_out.clear();
  jpg_files_out.clear();
  std::map<int, std::string> found_sgfs, found_jpgs;
  int max_step = -1;
  try {
    if (!fs::exists(game_folder_path) || !fs::is_directory(game_folder_path))
      return -1;
    for (const auto &entry : fs::directory_iterator(game_folder_path)) {
      if (entry.is_regular_file()) {
        std::string fn = entry.path().filename().string(),
                    ext = entry.path().extension().string(),
                    name = fn.substr(0, fn.length() - ext.length());
        try {
          size_t chars_proc = 0;
          int sn = std::stoi(name, &chars_proc);
          if (chars_proc == name.length()) {
            if (ext == ".sgf")
              found_sgfs[sn] = entry.path().string();
            else if (ext == ".jpg")
              found_jpgs[sn] = entry.path().string();
            if (sn > max_step)
              max_step = sn;
          }
        } catch (...) {
        }
      }
    }
    int actual_max_step = -1;
    for (int i = 0; i <= max_step; ++i) {
      if (found_sgfs.count(i) && found_jpgs.count(i)) {
        sgf_files_out.push_back(found_sgfs[i]);
        jpg_files_out.push_back(found_jpgs[i]);
        actual_max_step = i;
      } else
        break;
    }
    if (sgf_files_out.empty() && max_step == -1 && bDebug)
      LOG_DEBUG
          << "Debug (loadGameSteps): No numbered SGF/JPG step files found in "
          << game_folder_path << std::endl;
    else if (sgf_files_out.empty() && max_step > -1)
      return -1;
    return actual_max_step;
  } catch (const fs::filesystem_error &e) {
    LOG_ERROR << "FS error loadGameSteps: " << e.what() << std::endl;
    return -1;
  }
}

void studyModeWorkflow(const std::string &game_to_study_name) {
  LOG_INFO << "Starting Study Mode for game: " << game_to_study_name
           << std::endl;

  std::string game_folder_path = "./share/" + game_to_study_name;
  std::vector<std::string> sgf_step_files;
  std::vector<std::string> jpg_step_files;

  int max_step_idx =
      loadGameSteps(game_folder_path, sgf_step_files, jpg_step_files);

  if (max_step_idx < 0 || sgf_step_files.empty()) {
    THROWGEMERROR(
        "No game steps (matching SGF/JPG pairs from 0..N) found in folder: " +
        game_folder_path);
  }
  LOG_INFO << "  Loaded " << max_step_idx + 1 << " game steps (0 to "
           << max_step_idx << ")." << std::endl;

  std::string main_tournament_sgf_path = game_folder_path + "/tournament.sgf";
  std::ifstream tournament_sgf_stream(main_tournament_sgf_path);
  if (!tournament_sgf_stream.is_open()) {
    THROWGEMERROR("Failed to open main tournament SGF file: " +
                  main_tournament_sgf_path);
  }
  std::stringstream sstream_tournament;
  sstream_tournament << tournament_sgf_stream.rdbuf();
  std::string full_tournament_sgf_content = sstream_tournament.str();
  tournament_sgf_stream.close();
  if (full_tournament_sgf_content.empty()) {
    THROWGEMERROR("Main tournament SGF file is empty: " +
                  main_tournament_sgf_path);
  }

  int current_display_step_idx = 0;
  cv::Mat simulated_board_image;
  cv::Mat snapshot_image;

  std::string sim_board_window =
      "Study: Simulated Board - " + game_to_study_name;
  std::string snapshot_window = "Study: Snapshot View - " + game_to_study_name;
  cv::namedWindow(sim_board_window, cv::WINDOW_AUTOSIZE);
  cv::namedWindow(snapshot_window, cv::WINDOW_AUTOSIZE);
  cv::moveWindow(sim_board_window, 50, 50);
  cv::moveWindow(snapshot_window, 50 + canvas_size_px + 20, 50);

  auto displayCurrentStudyStep = [&]() {
    if (current_display_step_idx < 0 ||
        static_cast<size_t>(current_display_step_idx) >=
            jpg_step_files.size()) {
      LOG_ERROR << "Error: Invalid step index " << current_display_step_idx
                << " for display." << std::endl;
      return;
    }

    LOG_INFO << "  Displaying Step: " << current_display_step_idx << std::endl;
    cv::setWindowTitle(sim_board_window,
                       "Study: Sim Board - " + game_to_study_name + " (Step " +
                           std::to_string(current_display_step_idx) + ")");
    cv::setWindowTitle(snapshot_window,
                       "Study: Snapshot - " + game_to_study_name + " (Step " +
                           std::to_string(current_display_step_idx) + ")");

    snapshot_image = cv::imread(jpg_step_files[current_display_step_idx]);
    if (snapshot_image.empty()) {
      LOG_ERROR << "Error loading snapshot: "
                << jpg_step_files[current_display_step_idx] << std::endl;
      cv::Mat error_img = cv::Mat::zeros(480, 640, CV_8UC3);
      cv::putText(error_img, "Snapshot Load Error", cv::Point(50, 240),
                  cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
      cv::imshow(snapshot_window, error_img);
    } else {
      cv::imshow(snapshot_window, snapshot_image);
    }

    // Corrected call to drawSimulatedGoBoard with Phase 2 signature
    // Arg1: full_tournament_sgf_content (string)
    // Arg2: current_display_step_idx (int) - this is 'display_up_to_move_idx'
    // Arg3: simulated_board_image (Mat&)
    // Arg4: current_display_step_idx (int) - this is
    // 'highlight_this_move_idx' Arg5: canvas_size_px (int)
    drawSimulatedGoBoard(full_tournament_sgf_content, current_display_step_idx,
                         simulated_board_image, current_display_step_idx,
                         canvas_size_px);
    cv::imshow(sim_board_window, simulated_board_image);
  };

  displayCurrentStudyStep();

  while (true) {
    int key = cv::waitKey(0);
    if (key == 27) {
      break;
    } else if (key == 'f' || key == 'F') {
      if (current_display_step_idx < max_step_idx) {
        current_display_step_idx++;
        displayCurrentStudyStep();
      } else {
        LOG_INFO << "  Already at the last step (" << max_step_idx << ")."
                 << std::endl;
      }
    } else if (key == 'b' || key == 'B') {
      if (current_display_step_idx > 0) {
        current_display_step_idx--;
        displayCurrentStudyStep();
      } else {
        LOG_INFO << "  Already at the first step (0)." << std::endl;
      }
    }
  }

  cv::destroyAllWindows();
  LOG_INFO << "Study Mode Finished for game: " << game_to_study_name
           << std::endl;
}

void detectStonePositionWorkflow(int target_col, int target_row,
                                 const std::string &image_path_from_arg) {
  LOG_INFO << "Starting Stone Detection at Position Workflow..." << std::endl;
  if (target_col < 0 || target_col > 18 || target_row < 0 || target_row > 18) {
    THROWGEMERROR("Invalid target column/row. Must be between 0 and 18.");
  }

  std::string image_path_to_use = image_path_from_arg;
  if (image_path_to_use.empty()) {
    image_path_to_use = g_default_input_image_path;
    LOG_INFO << "  No --image specified, using default: " << image_path_to_use
             << std::endl;
  } else {
    LOG_INFO << "  Using image: " << image_path_to_use << std::endl;
  }
  LOG_INFO << "  Target position: Col=" << target_col << ", Row=" << target_row
           << std::endl;

  cv::Mat raw_image = cv::imread(image_path_to_use);
  if (raw_image.empty()) {
    THROWGEMERROR("Could not load image: " + image_path_to_use);
  }

  // 1. Correct Perspective
  // Ensure calibration is set up if correctPerspective relies on it
  // implicitly or explicitly needs it. For this workflow, we'll assume
  // correctPerspective uses CALIB_CONFIG_PATH.
  if (!setupCalibrationFromConfig()) { // Ensure globals like
                                       // g_capture_width/height are set if
                                       // needed by correctPerspective
    // setupCalibrationFromConfig might throw, or return false.
    // If it returns false without throwing, we should handle it.
    LOG_ERROR << "Warning: Could not setup calibration from config for "
                 "perspective correction. Results may be inaccurate."
              << std::endl;
    // Depending on strictness, you might THROWGEMERROR here.
  }
  cv::Mat corrected_image = correctPerspective(raw_image);
  if (corrected_image.empty()) {
    THROWGEMERROR(
        "Perspective correction failed or resulted in an empty image.");
  }

  // 2. Load Calibration Data (needed for detectStoneAtPosition)
  CalibrationData calib_data = loadCalibrationData(CALIB_CONFIG_PATH);
  if (!calib_data.colors_loaded || !calib_data.corners_loaded ||
      !calib_data.dimensions_loaded) {
    // detectStoneAtPosition might rely on dimensions from calib_data if not
    // using corrected_image.cols/rows directly For now, detectStoneAtPosition
    // as implemented primarily needs calib_data.lab_** for color references.
    THROWGEMERROR("Essential calibration data (colors/corners/dimensions) not "
                  "loaded from " +
                  CALIB_CONFIG_PATH + ". Cannot proceed with stone detection.");
  }

  // 3. Detect Stone
  int stone_color = detectStoneAtPosition(corrected_image, target_col,
                                          target_row, calib_data);

  // 4. Prepare for Drawing: Calculate pixel center of the target intersection
  // This logic is similar to the start of calculateGridIntersectionROI
  std::vector<cv::Point2f> ideal_board_corners =
      getBoardCornersCorrected(corrected_image.cols, corrected_image.rows);
  if (ideal_board_corners.size() != 4) {
    THROWGEMERROR(
        "getBoardCornersCorrected did not return 4 points for drawing.");
  }
  cv::Point2f board_top_left_px = ideal_board_corners[0];
  float grid_area_width_px =
      ideal_board_corners[1].x - ideal_board_corners[0].x;
  float grid_area_height_px =
      ideal_board_corners[3].y - ideal_board_corners[0].y;
  int grid_lines = 19;
  float avg_grid_spacing_x =
      grid_area_width_px / static_cast<float>(grid_lines - 1);
  float avg_grid_spacing_y =
      grid_area_height_px / static_cast<float>(grid_lines - 1);

  cv::Point2f intersection_center_px(
      board_top_left_px.x + static_cast<float>(target_col) * avg_grid_spacing_x,
      board_top_left_px.y +
          static_cast<float>(target_row) * avg_grid_spacing_y);

  // Determine a suitable radius for drawing the circle (e.g., ~45% of average
  // grid spacing)
  int draw_radius = static_cast<int>(
      std::min(avg_grid_spacing_x, avg_grid_spacing_y) * 0.45f);
  draw_radius = std::max(draw_radius, 5); // Minimum radius
  int circle_thickness = 2;

  // 5. Display and Draw Result
  cv::Mat display_image = corrected_image.clone();
  std::string result_text;
  cv::Scalar circle_color;

  switch (stone_color) {
  case BLACK:
    result_text = "Detected: BLACK Stone";
    circle_color = cv::Scalar(0, 0, 0); // Black
    cv::circle(display_image, intersection_center_px, draw_radius, circle_color,
               circle_thickness);
    break;
  case WHITE:
    result_text = "Detected: WHITE Stone";
    circle_color = cv::Scalar(255, 255, 255); // White
    cv::circle(display_image, intersection_center_px, draw_radius, circle_color,
               circle_thickness);
    break;
  case EMPTY:
    result_text = "Detected: EMPTY";
    // Optionally draw a subtle marker for empty if desired, e.g., a small
    // green dot or cross For now, just text.
    break;
  default:
    result_text = "Detection Error or Unknown State";
    break;
  }

  cv::putText(display_image, result_text, cv::Point(10, 30),
              cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
  cv::putText(display_image,
              "Col: " + std::to_string(target_col) +
                  " Row: " + std::to_string(target_row),
              cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5,
              cv::Scalar(0, 255, 0), 1);

  std::string window_title = "Stone Detection Result";
  cv::imshow(window_title, display_image);
  if (bDebug) {
    cv::Rect roi_for_debug = calculateGridIntersectionROI(
        target_col, target_row, corrected_image.cols, corrected_image.rows);
    if (roi_for_debug.width > 0 && roi_for_debug.height > 0) {
      roi_for_debug &=
          cv::Rect(0, 0, corrected_image.cols, corrected_image.rows); // clamp
      if (roi_for_debug.width > 0 && roi_for_debug.height > 0) {
        cv::Mat roi_debug_img = corrected_image(roi_for_debug).clone();
        cv::imshow("Debug ROI for Detection", roi_debug_img);
      }
    }
  }
  LOG_INFO << "  " << result_text << " at (" << target_col << "," << target_row
           << ")" << std::endl;
  LOG_INFO << "  Displaying image. Press any key in the OpenCV window to close."
           << std::endl;
  cv::waitKey(0);
  cv::destroyAllWindows();
  LOG_INFO << "Stone Detection Workflow Finished." << std::endl;
}

int main(int argc, char *argv[]) {
  // Initialize logger with default path and level.
  // This allows logging from the very start, even during option parsing if
  // needed. The log level can be overridden by the command-line argument
  // later.
  Logger::init(); // Uses default "share/log.txt" and LogLevel::INFO

  // Initial log message indicating program start
  LOG_INFO << "GEM application started." << std::endl;

  try {
    if (argc == 1) {
      displayHelpMessage();
      return 0;
    }
    int option_index = 0;
    int parsed_log_level_int =
        -1; // To store numeric log level from command line

    std::string snapshot_output;
    std::string record_sgf_output;
    std::string test_perspective_image_path; // For --test-perspective
    std::string draw_board_sgf_path_arg;     // For --draw-board
    std::string detect_stone_image_path_arg; // For --image with -P

    bool run_probe_devices = false;
    bool run_calibration = false;
    bool run_interactive_calibration = false;
    bool run_test_calibration = false;
    bool run_draw_board_workflow = false; // Flag for the new workflow
    bool run_tournament_mode = false;     // Flag for new tournament mode
    bool run_study_mode = false;
    bool run_detect_stone_position_workflow = false;  // NEW Flag
    int detect_stone_col = -1, detect_stone_row = -1; // NEW Args for -P

    auto isWorkflowSelected = [&]() -> bool {
      return run_probe_devices || run_calibration ||
             run_detect_stone_position_workflow || run_test_calibration ||
             run_study_mode || run_tournament_mode ||
             !snapshot_output.empty() || !record_sgf_output.empty() ||
             run_draw_board_workflow || !test_perspective_image_path.empty();
    };
    struct option long_options[] = {
        {"image", required_argument, nullptr,
         0}, // For -P's optional image path
        {"process-image", required_argument, nullptr, 'p'},
        {"generate-sgf", required_argument, nullptr, 'g'},
        {"verify", required_argument, nullptr, 'v'},
        {"compare", required_argument, nullptr, 'c'},
        {"parse", required_argument, nullptr, 0},
        {"help", no_argument, nullptr, 'h'},
        {"debug", no_argument, nullptr, 'd'},
        {"log-level", required_argument, nullptr, 'O'}, // NEW
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
    while ((c = getopt_long(argc, argv, "dp:g:v:c:h:s:r:D:BbM:S:ftuPO",
                            long_options, &option_index)) != -1) {
      switch (c) {
      case 'd':
        bDebug = true;
        LOG_INFO << "Debug mode enabled." << endl;
        break;
      case 'O': // NEW: Log Level Option
        try {
          parsed_log_level_int = std::stoi(optarg);
          if (parsed_log_level_int >= static_cast<int>(LogLevel::NONE) &&
              parsed_log_level_int <= static_cast<int>(LogLevel::DEBUG)) {
            Logger::setGlobalLogLevel(
                static_cast<LogLevel>(parsed_log_level_int));
            // The setGlobalLogLevel method will log this change itself if
            // INFO level is active
          } else {
            CONSOLE_ERR << "Invalid log level: " << optarg
                        << ". Must be between 0 (NONE) and 4 (DEBUG)."
                        << std::endl;
            // Logger might not be fully set up or at a level to log this
            // error yet. LOG_ERROR << "Invalid log level specified: " <<
            // optarg << ". Using current level." << std::endl; No change to
            // log level if invalid
          }
        } catch (const std::invalid_argument &ia) {
          CONSOLE_ERR << "Invalid argument for log level: " << optarg
                      << ". Expected an integer (0-4)." << std::endl;
          // LOG_ERROR << "Invalid argument for log level: " << optarg << ".
          // Not an integer." << std::endl;
        } catch (const std::out_of_range &oor) {
          CONSOLE_ERR << "Log level argument out of range: " << optarg << "."
                      << std::endl;
          // LOG_ERROR << "Log level argument out of range: " << optarg << "."
          // << std::endl;
        }
        break;
      case 'D': // need to handle more than one digig
      {
        Str2Num num(optarg);
        if (num && num.val() >= 0 && num.val() < 256) {
          auto sz = g_device_path.size() - 1;
          g_device_path.resize(sz);
          g_device_path += optarg;
          LOG_INFO << "Device:" << g_device_path << endl;
        } else {
          LOG_INFO << "invalid device number: " << optarg << endl;
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
            LOG_ERROR << "Warning: Invalid capture mode '" << mode_str
                      << "'. Using default (v4l2)." << std::endl;
            gCaptureMode = MODE_V4L2;
          }
          if (bDebug)
            LOG_DEBUG << "Debug: Capture mode set to "
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
          THROWGEMERROR("-g option requires an input image path and an "
                        "output SGF path.");
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
              LOG_ERROR
                  << "Error: Frame dimensions must be positive for --size."
                  << std::endl;
              return 1;
            }

            LOG_DEBUG << "Debug: Requested capture size set to "
                      << g_capture_width << "x" << g_capture_height
                      << " by option." << std::endl;

          } catch (const std::invalid_argument &ia) {
            LOG_ERROR << "Error: Invalid number format in --size argument: "
                      << size_str << std::endl;
            return 1;
          } catch (const std::out_of_range &oor) {
            LOG_ERROR << "Error: Number out of range in --size argument: "
                      << size_str << std::endl;
            return 1;
          }
        } else {
          LOG_ERROR << "Error: Invalid format for --size. Expected "
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
      case 'P': // NEW case for detecting stone position
        run_detect_stone_position_workflow = true;
        break;
      case 0: // Long-only options (val was 0)
        if (long_options[option_index].name == std::string("image")) {
          detect_stone_image_path_arg = optarg; // Store path for -P
        } else if (long_options[option_index].name ==
                   std::string("game-name")) {
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

    // Log effective settings AFTER parsing all options
    LOG_INFO << "Effective settings: Debug Mode=" << (bDebug ? "ON" : "OFF")
             << ", LogLevel=" << static_cast<int>(Logger::getGlobalLogLevel())
             << ", Device=" << g_device_path << ", CaptureMode="
             << (gCaptureMode == MODE_OPENCV ? "OpenCV" : "V4L2")
             << ", Size=" << g_capture_width << "x" << g_capture_height
             << std::endl;

    // Add a helper isWorkflowSelected if needed to simplify the final else if
    // condition bool isWorkflowSelected(bool probe, bool calib, bool detect,
    // /*...all workflow flags...*/) { return probe || calib || detect ...; }

    // --- Workflow Execution Logic ---
    // (Prioritize more specific/terminal workflows first)

    if (run_probe_devices) {
      probeVideoDevicesWorkflow();
    } else if (run_calibration) {
      // Calibration might implicitly use setupCalibrationFromConfig or handle
      // its own config needs
      calibrationWorkflow(run_interactive_calibration);
    } else if (
        run_detect_stone_position_workflow) { // Check this before generic
                                              // workflows that might also use
                                              // setupCalibrationFromConfig
      if (optind + 1 < argc) { // Need two positional args: col and row
        try {
          detect_stone_col = std::stoi(argv[optind]);
          detect_stone_row = std::stoi(argv[optind + 1]);
          if (detect_stone_col < 0 || detect_stone_col > 18 ||
              detect_stone_row < 0 || detect_stone_row > 18) {
            THROWGEMERROR("-P option col/row values must be between 0 and 18.");
          }
          // Use detect_stone_image_path_arg if set, otherwise it's empty and
          // workflow will use default
          detectStonePositionWorkflow(detect_stone_col, detect_stone_row,
                                      detect_stone_image_path_arg);
        } catch (const std::invalid_argument &ia) {
          THROWGEMERROR("Invalid column/row arguments for -P option. "
                        "Expected integers.");
        } catch (const std::out_of_range &oor) {
          THROWGEMERROR("Column/row arguments for -P option are out of range.");
        }
      } else {
        THROWGEMERROR("-P option requires <col> and <row> arguments after "
                      "all options.");
      }
    } else if (run_test_calibration) { // This might also need
                                       // setupCalibrationFromConfig
      if (!setupCalibrationFromConfig()) {
        THROWGEMERROR(
            "Calibration setup failed for --test-calibration-config.");
      }
      testCalibrationConfigWorkflow();
    } else if (run_study_mode) {
      if (!setupCalibrationFromConfig()) {
        THROWGEMERROR("Calibration setup failed for study mode.");
      }
      studyModeWorkflow(g_default_game_name_prefix);
    } else if (run_tournament_mode) {
      if (!setupCalibrationFromConfig()) {
        THROWGEMERROR("Calibration setup failed for tournament mode.");
      }
      tournamentModeWorkflow(g_default_game_name_prefix);
    } else if (!snapshot_output.empty()) {
      // Snapshot might or might not need full calib setup, depends on if it
      // uses g_device_path/size from config. Assuming for now it can use
      // command-line specified or defaults if CALIB_CONFIG_PATH is not
      // strictly enforced for it. If CALIB_CONFIG_PATH values are *required*
      // for snapshot, then setupCalibrationFromConfig() should be called.
      captureSnapshotWorkflow(snapshot_output);
    } else if (!record_sgf_output.empty()) {
      if (!setupCalibrationFromConfig()) {
        THROWGEMERROR("Calibration setup failed for record SGF mode.");
      }
      recordSGFWorkflow(record_sgf_output);
    } else if (run_draw_board_workflow) {
      // Draw board likely doesn't need camera calibration, just SGF parsing.
      drawSimulatedBoardWorkflow(draw_board_sgf_path_arg);
    } else if (!test_perspective_image_path.empty()) {
      // This is a dev tool, assume it might need calibration if
      // correctPerspective() relies on it.
      if (!setupCalibrationFromConfig()) {
        THROWGEMERROR("Calibration setup failed for test perspective mode.");
      }
      testPerspectiveTransformWorkflow(test_perspective_image_path);
    } else if (optind < argc) { // Unprocessed positional arguments without a
                                // specific flag like -P
      // This indicates an error if no workflow like -p, -g, -v, -c was hit
      // which would consume them. However, those cases already return 0
      // above. This path is more for "gem some_unknown_arg"
      LOG_ERROR << "Error: Unprocessed arguments. What is '" << argv[optind]
                << "'?" << std::endl;
      displayHelpMessage();
      return 1;
    } else if (argc > 1 && optind == argc && !isWorkflowSelected()) {
      // If only options like -d, -D, -M, -S, --game-name, -O were provided
      // without a primary action workflow
      LOG_INFO << "Configuration options processed. No primary action workflow "
                  "specified. Displaying help."
               << std::endl;
      displayHelpMessage(); // Uses CONSOLE_OUT
    } else if (optind < argc) {
      LOG_ERROR << "Unprocessed arguments. What is '" << argv[optind] << "'?"
                << std::endl;
      displayHelpMessage(); // Uses CONSOLE_OUT
      return 1;
    }
  } catch (const GEMError &e) {
    LOG_ERROR << "Error: " << e.what() << endl;
    return 1;
  } catch (const std::exception &e) {
    LOG_ERROR << "An unexpected error occurred: " << e.what() << endl;
    return 1;
  } catch (...) {
    LOG_ERROR << "An unknown error occurred." << endl;
    return 1;
  }
  return 0;
}