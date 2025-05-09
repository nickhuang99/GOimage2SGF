#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/imgproc.hpp> // For cvtColor
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "common.h"

// Enum to represent the active corner for adjustment
enum class ActiveCorner {
  NONE,
  TOP_LEFT,
  TOP_RIGHT,
  BOTTOM_LEFT,
  BOTTOM_RIGHT
};
ActiveCorner currentActiveCorner = ActiveCorner::TOP_LEFT; // Default to TL

// --- NEW: Define calibration output paths ---
const std::string CALIB_CONFIG_PATH = "./share/config.txt";
const std::string CALIB_SNAPSHOT_PATH = "./share/snapshot.jpg";
const std::string CALIB_SNAPSHOT_DEBUG_PATH = "./share/snapshot_osd.jpg";

// Function to modify a single point based on direction keys
void movePoint(cv::Point2f &point, int move_key, int step, int frame_width,
               int frame_height) {
  switch (move_key) {
  case 'i':
    point.y -= step;
    break; // Up
  case 'k':
    point.y += step;
    break; // Down
  case 'j':
    point.x -= step;
    break; // Left
  case 'l':
    point.x += step;
    break; // Right
  }
  // Boundary checks
  point.x = std::max(0.0f, std::min((float)frame_width - 1, point.x));
  point.y = std::max(0.0f, std::min((float)frame_height - 1, point.y));
}

// This function will be called by runInteractiveCalibration's main loop
// It replaces the old big switch in handleCalibrationInput for movement.
// It returns 's' for save, 27 for ESC, or 0 for other handled keys.
int processCalibrationKeyPress(
    int key, cv::Point2f &tl, cv::Point2f &tr, cv::Point2f &bl, cv::Point2f &br,
    int frame_width, int frame_height,
    ActiveCorner &activeCorner) { // Pass activeCorner by reference
  const int step = 5;

  switch (key) {
  // Corner Selection
  case '1':
    activeCorner = ActiveCorner::TOP_LEFT;
    break;
  case '2':
    activeCorner = ActiveCorner::TOP_RIGHT;
    break;
  case '3':
    activeCorner = ActiveCorner::BOTTOM_LEFT;
    break;
  case '4':
    activeCorner = ActiveCorner::BOTTOM_RIGHT;
    break;

  // Movement Keys for the active corner
  case 'i': // Up
  case 'k': // Down
  case 'j': // Left
  case 'l': // Right
    if (activeCorner == ActiveCorner::TOP_LEFT)
      movePoint(tl, key, step, frame_width, frame_height);
    else if (activeCorner == ActiveCorner::TOP_RIGHT)
      movePoint(tr, key, step, frame_width, frame_height);
    else if (activeCorner == ActiveCorner::BOTTOM_LEFT)
      movePoint(bl, key, step, frame_width, frame_height);
    else if (activeCorner == ActiveCorner::BOTTOM_RIGHT)
      movePoint(br, key, step, frame_width, frame_height);
    break;

  case 's':
    return 's'; // Signal save
  case 27:
    return 27; // Signal exit (ESC)
  default:
    return 0; // Or key if you want to pass unhandled ones
  }

  // Anti-crossing logic can be more complex with free individual corner
  // movement. For now, we'll rely on boundary checks within movePoint. A more
  // robust solution might check if the quadrilateral becomes self-intersecting.
  // Simple horizontal anti-crossing (less relevant now but can be adapted if
  // needed): if (tr.x < tl.x + 10.0f) tr.x = tl.x + 10.0f; if (br.x < bl.x
  // + 10.0f) br.x = bl.x + 10.0f; Simple vertical anti-crossing: if (bl.y <
  // tl.y + 10.0f) bl.y = tl.y + 10.0f; if (br.y < tr.y + 10.0f) br.y = tr.y
  // + 10.0f; This basic anti-crossing might need refinement depending on
  // desired behavior.

  return 0; // Default: key was handled for movement/selection
}

// --- Helper Function to Handle Keyboard Input for Calibration ---
// Takes the pressed key, modifies corner points (by reference).
// Update function signature:
int handleCalibrationInput(int key, cv::Point2f &topLeft, cv::Point2f &topRight,
                           cv::Point2f &bottomLeft,
                           cv::Point2f &bottomRight, // Add these
                           int frame_width, int frame_height) {
  const int step = 5;
  int return_signal = 0;

  switch (key) {
  // Top edge controls
  case 'u':
    topLeft.y -= step;
    topRight.y -= step;
    break;
  case 'd':
    topLeft.y += step;
    topRight.y += step;
    break;
  case 'w':
    topLeft.x -= step;
    topRight.x += step;
    break;
  case 'n':
    topLeft.x += step;
    topRight.x -= step;
    break;

  // Bottom edge controls
  case 'k':
    bottomLeft.y -= step;
    bottomRight.y -= step;
    break; // bottom edge up
  case 'j':
    bottomLeft.y += step;
    bottomRight.y += step;
    break; // bottom edge down
  case 'l':
    bottomLeft.x -= step;
    bottomRight.x += step;
    break; // bottom edge wider (NEW: 'l' instead of ',')
  case 'm':
    bottomLeft.x += step;
    bottomRight.x -= step;
    break; // bottom edge narrower

  case 's':
    return_signal = 's';
    break;
  case 27:
    return_signal = 27;
    break; // ESC
  default:
    if (key != -1)
      return_signal = key;
    break;
  }

  // Boundary checks for all four corners
  topLeft.x = std::max(0.0f, std::min((float)frame_width - 1, topLeft.x));
  topLeft.y = std::max(0.0f, std::min((float)frame_height - 1, topLeft.y));
  topRight.x = std::max(0.0f, std::min((float)frame_width - 1, topRight.x));
  topRight.y = std::max(0.0f, std::min((float)frame_height - 1, topRight.y));

  bottomLeft.x = std::max(0.0f, std::min((float)frame_width - 1, bottomLeft.x));
  bottomLeft.y =
      std::max(0.0f, std::min((float)frame_height - 1, bottomLeft.y));
  bottomRight.x =
      std::max(0.0f, std::min((float)frame_width - 1, bottomRight.x));
  bottomRight.y =
      std::max(0.0f, std::min((float)frame_height - 1, bottomRight.y));

  // Prevent crossing over for top edge (existing logic)
  if (topRight.x < topLeft.x + 10.0f) { // Minimum 10px separation
    if (key == 'n') {
      topLeft.x -= step;
      topRight.x += step;
    } // Revert if crossed by 'n'
    else if (key == 'w') {
      topLeft.x += step;
      topRight.x -= step;
    } // Revert if crossed by 'w'
    else {
      topRight.x = topLeft.x + 10.0f;
    } // Generic fix if already crossed
  }

  // Prevent crossing over for bottom edge (NEW logic)
  if (bottomRight.x < bottomLeft.x + 10.0f) { // Minimum 10px separation
    if (key == 'm') {
      bottomLeft.x -= step;
      bottomRight.x += step;
    } // Revert if crossed by 'm'
    else if (key == ',') {
      bottomLeft.x += step;
      bottomRight.x -= step;
    } // Revert if crossed by ',' (comma)
    else {
      bottomRight.x = bottomLeft.x + 10.0f;
    } // Generic fix
  }

  // Optional: Prevent vertical crossing (e.g., TL.y > BL.y)
  // For simplicity, this is not added here, but could be considered.
  // Users can visually manage this. The perspective transform can handle
  // non-convex quads but it's generally better if the quadrilateral is
  // well-behaved.

  return return_signal;
}

// --- Helper Function to Draw Calibration OSD (Unchanged for drawing
// points/lines) ---
void drawCalibrationOSD(
    cv::Mat &display_frame, const cv::Point2f &tl, const cv::Point2f &tr,
    const cv::Point2f &bl, const cv::Point2f &br,
    ActiveCorner activeCorner) { // Add activeCorner parameter
  // --- Draw Corner Circles (highlight active corner) ---
  int circle_radius = 5;
  int active_circle_radius = 8;
  cv::Scalar inactive_color(150, 150, 150); // Dim color for inactive

  cv::circle(display_frame, tl,
             (activeCorner == ActiveCorner::TOP_LEFT ? active_circle_radius
                                                     : circle_radius),
             (activeCorner == ActiveCorner::TOP_LEFT ? cv::Scalar(0, 0, 255)
                                                     : inactive_color),
             -1); // Red if active
  cv::circle(display_frame, tr,
             (activeCorner == ActiveCorner::TOP_RIGHT ? active_circle_radius
                                                      : circle_radius),
             (activeCorner == ActiveCorner::TOP_RIGHT ? cv::Scalar(255, 0, 0)
                                                      : inactive_color),
             -1); // Blue if active
  cv::circle(display_frame, bl,
             (activeCorner == ActiveCorner::BOTTOM_LEFT ? active_circle_radius
                                                        : circle_radius),
             (activeCorner == ActiveCorner::BOTTOM_LEFT
                  ? cv::Scalar(255, 0, 255)
                  : inactive_color),
             -1); // Magenta if active
  cv::circle(display_frame, br,
             (activeCorner == ActiveCorner::BOTTOM_RIGHT ? active_circle_radius
                                                         : circle_radius),
             (activeCorner == ActiveCorner::BOTTOM_RIGHT
                  ? cv::Scalar(0, 255, 255)
                  : inactive_color),
             -1); // Yellow if active

  // --- Draw Connecting Lines ---
  cv::Scalar line_color(0, 255, 0);
  int line_thickness = 1;
  cv::line(display_frame, tl, tr, line_color, line_thickness);
  cv::line(display_frame, tr, br, line_color, line_thickness);
  cv::line(display_frame, br, bl, line_color, line_thickness);
  cv::line(display_frame, bl, tl, line_color, line_thickness);

  // --- OSD Text Settings ---
  int font_face = cv::FONT_HERSHEY_SIMPLEX;
  cv::Scalar help_text_color(0, 0, 255);
  cv::Scalar coord_text_color(255, 200, 0); // Cyan for coordinates
  int text_thickness = 1;
  double help_font_scale = 0.45; // Slightly smaller to fit more text
  cv::Point help_text_origin(10, 20);
  cv::Point help_text_origin_line2(10, 35);
  cv::Point help_text_origin_line3(10, 50);

  // --- Draw Help Text OSD ---
  std::string active_corner_str = "Active: ";
  switch (activeCorner) {
  case ActiveCorner::TOP_LEFT:
    active_corner_str += "TL (1)";
    break;
  case ActiveCorner::TOP_RIGHT:
    active_corner_str += "TR (2)";
    break;
  case ActiveCorner::BOTTOM_LEFT:
    active_corner_str += "BL (3)";
    break;
  case ActiveCorner::BOTTOM_RIGHT:
    active_corner_str += "BR (4)";
    break;
  default:
    active_corner_str += "None";
    break;
  }

  std::string help_text_line1 = "Select: 1(TL) 2(TR) 3(BL) 4(BR)";
  std::string help_text_line2 =
      "Move (" + active_corner_str + "): i(up) k(down) j(left) l(right)";
  std::string help_text_line3 = "s: save, esc: exit";

  cv::putText(display_frame, help_text_line1, help_text_origin, font_face,
              help_font_scale, help_text_color, 1, cv::LINE_AA);
  cv::putText(display_frame, help_text_line2, help_text_origin_line2, font_face,
              help_font_scale, help_text_color, 1, cv::LINE_AA);
  cv::putText(display_frame, help_text_line3, help_text_origin_line3, font_face,
              help_font_scale, help_text_color, 1, cv::LINE_AA);

  // --- Draw Coordinate Text OSD (no change here) ---
  double coord_font_scale = 0.4;
  std::stringstream ss;
  ss << std::fixed << std::setprecision(0);

  ss.str("");
  ss << "TL(" << tl.x << "," << tl.y << ")";
  cv::putText(display_frame, ss.str(), tl + cv::Point2f(10, -10), font_face,
              coord_font_scale, coord_text_color, text_thickness, cv::LINE_AA);
  ss.str("");
  ss << "TR(" << tr.x << "," << tr.y << ")";
  cv::putText(display_frame, ss.str(), tr + cv::Point2f(-60, -10), font_face,
              coord_font_scale, coord_text_color, text_thickness, cv::LINE_AA);
  ss.str("");
  ss << "BL(" << bl.x << "," << bl.y << ")";
  cv::putText(display_frame, ss.str(), bl + cv::Point2f(10, 20), font_face,
              coord_font_scale, coord_text_color, text_thickness, cv::LINE_AA);
  ss.str("");
  ss << "BR(" << br.x << "," << br.y << ")";
  cv::putText(display_frame, ss.str(), br + cv::Point2f(-60, 20), font_face,
              coord_font_scale, coord_text_color, text_thickness, cv::LINE_AA);
}

// --- NEW Function to Save Corner Configuration (Unchanged) ---
bool saveCornerConfig(const std::string &filename, const cv::Point2f &tl,
                      const cv::Point2f &tr, const cv::Point2f &bl,
                      const cv::Point2f &br, int frame_width, int frame_height,
                      const cv::Vec3f &lab_tl, const cv::Vec3f &lab_tr,
                      const cv::Vec3f &lab_bl, const cv::Vec3f &lab_br,
                      const cv::Vec3f &avg_lab_board) // NEW parameter
{
  std::ofstream outFile(filename);
  if (!outFile.is_open()) {
    std::cerr << "Error: Could not open config file for writing: " << filename
              << std::endl;
    return false;
  }
  outFile << "# Go Board Calibration Configuration" << std::endl;
  // ... (ImageWidth, ImageHeight, Pixel Coordinates, Corner Lab Colors as in
  // your existing code)
  outFile << "ImageWidth=" << frame_width << std::endl;
  outFile << "ImageHeight=" << frame_height << std::endl;
  outFile << std::fixed << std::setprecision(1);

  outFile << "\n# Corner Pixel Coordinates (TL, TR, BL, BR)" << std::endl;
  outFile << "TL_X_PX=" << tl.x << std::endl;
  outFile << "TL_Y_PX=" << tl.y << std::endl;
  outFile << "TR_X_PX=" << tr.x << std::endl;
  outFile << "TR_Y_PX=" << tr.y << std::endl;
  outFile << "BL_X_PX=" << bl.x << std::endl;
  outFile << "BL_Y_PX=" << bl.y << std::endl;
  outFile << "BR_X_PX=" << br.x << std::endl;
  outFile << "BR_Y_PX=" << br.y << std::endl;

  outFile << "\n# Sampled Corner Lab Colors (L: 0-255, a/b: 0-255 approx)"
          << std::endl;
  outFile << "# Black stones expected at TL, BL. White stones at TR, BR."
          << std::endl;
  outFile << "TL_L=" << lab_tl[0] << std::endl;
  outFile << "TL_A=" << lab_tl[1] << std::endl;
  outFile << "TL_B=" << lab_tl[2] << std::endl;
  outFile << "TR_L=" << lab_tr[0] << std::endl;
  outFile << "TR_A=" << lab_tr[1] << std::endl;
  outFile << "TR_B=" << lab_tr[2] << std::endl;
  outFile << "BL_L=" << lab_bl[0] << std::endl;
  outFile << "BL_A=" << lab_bl[1] << std::endl;
  outFile << "BL_B=" << lab_bl[2] << std::endl;
  outFile << "BR_L=" << lab_br[0] << std::endl;
  outFile << "BR_A=" << lab_br[1] << std::endl;
  outFile << "BR_B=" << lab_br[2] << std::endl;

  // --- NEW: Save Average Board Color ---
  outFile << "\n# Sampled Average Empty Board Lab Color" << std::endl;
  outFile << "BOARD_L_AVG=" << avg_lab_board[0] << std::endl;
  outFile << "BOARD_A_AVG=" << avg_lab_board[1] << std::endl;
  outFile << "BOARD_B_AVG=" << avg_lab_board[2] << std::endl;

  // Percentage Coordinates (as in your existing code)
  if (frame_width > 0 && frame_height > 0) {
    outFile << "\n# Corner Percentage Coordinates (%)" << std::endl;
    outFile << "TL_X_PC=" << (tl.x / frame_width * 100.0f) << std::endl;
    outFile << "TL_Y_PC=" << (tl.y / frame_height * 100.0f) << std::endl;
    outFile << "TR_X_PC=" << (tr.x / frame_width * 100.0f) << std::endl;
    outFile << "TR_Y_PC=" << (tr.y / frame_height * 100.0f) << std::endl;
    outFile << "BL_X_PC=" << (bl.x / frame_width * 100.0f) << std::endl;
    outFile << "BL_Y_PC=" << (bl.y / frame_height * 100.0f) << std::endl;
    outFile << "BR_X_PC=" << (br.x / frame_width * 100.0f) << std::endl;
    outFile << "BR_Y_PC=" << (br.y / frame_height * 100.0f) << std::endl;
  }

  outFile.close();
  if (!outFile) {
    std::cerr << "Error: Failed to write data correctly to config file: "
              << filename << std::endl;
    return false;
  }
  std::cout << "Configuration (corners, stone colors, board color) saved to "
            << filename << std::endl;
  return true;
}

// Definition of the new utility function
bool trySetCameraResolution(cv::VideoCapture &cap, int desired_width,
                            int desired_height, bool attempt_fallback_format) {
  if (!cap.isOpened()) {
    if (bDebug)
      std::cerr << "Debug: trySetCameraResolution - VideoCapture not open."
                << std::endl;
    return false;
  }

  bool success = false;
  int initial_fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
  std::string initial_format_name = "MJPEG";

  if (bDebug) {
    std::cout << "Debug: trySetCameraResolution - Attempting "
              << initial_format_name << " at " << desired_width << "x"
              << desired_height << std::endl;
  }

  // Attempt 1: MJPEG (or a primary preferred format)
  cap.set(cv::CAP_PROP_FOURCC, initial_fourcc);
  cap.set(cv::CAP_PROP_FRAME_WIDTH, static_cast<double>(desired_width));
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<double>(desired_height));

  // Allow some time for settings to apply on some cameras/backends
  // cap.grab(); // You might need a short delay or a grab here for settings to
  // stick before get() cv::waitKey(50); // Small delay

  int actual_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int actual_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  int actual_fourcc_int = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
  char fourcc_str[5] = {0};
  memcpy(fourcc_str, &actual_fourcc_int, 4);

  if (bDebug) {
    std::cout << "Debug: trySetCameraResolution - After 1st attempt (MJPEG): "
              << "Actual Size: " << actual_width << "x" << actual_height
              << ", Actual FOURCC: " << fourcc_str
              << " (Requested: " << desired_width << "x" << desired_height
              << ")" << std::endl;
  }

  if (actual_width == desired_width && actual_height == desired_height) {
    success = true;
  }

  // Attempt 2: Fallback format (e.g., YUYV) if first failed and fallback is
  // enabled
  if (!success && attempt_fallback_format) {
    if (bDebug)
      std::cout << "Debug: trySetCameraResolution - Initial attempt failed. "
                   "Trying fallback YUYV."
                << std::endl;

    int fallback_fourcc = cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V');
    std::string fallback_format_name = "YUYV";

    cap.set(cv::CAP_PROP_FOURCC, fallback_fourcc);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, static_cast<double>(desired_width));
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<double>(desired_height));

    // cap.grab();
    // cv::waitKey(50);

    actual_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    actual_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    actual_fourcc_int = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
    memcpy(fourcc_str, &actual_fourcc_int, 4); // Update fourcc_str

    if (bDebug) {
      std::cout << "Debug: trySetCameraResolution - After 2nd attempt (YUYV): "
                << "Actual Size: " << actual_width << "x" << actual_height
                << ", Actual FOURCC: " << fourcc_str
                << " (Requested: " << desired_width << "x" << desired_height
                << ")" << std::endl;
    }

    if (actual_width == desired_width && actual_height == desired_height) {
      success = true;
    }
  }

  if (success && bDebug) {
    std::cout << "Debug: trySetCameraResolution - Successfully set to "
              << actual_width << "x" << actual_height
              << " with FOURCC: " << fourcc_str << std::endl;
  } else if (!success && bDebug) {
    std::cout
        << "Debug: trySetCameraResolution - Failed to set desired resolution "
        << desired_width << "x" << desired_height
        << ". Final actual size: " << actual_width << "x" << actual_height
        << ", FOURCC: " << fourcc_str << std::endl;
  }
  return success;
}

// --- SIMPLIFIED Main Calibration Function ---
void runInteractiveCalibration(int camera_index) {
  cv::VideoCapture cap;
  cap.open(camera_index, cv::CAP_V4L2);
  if (!cap.isOpened()) {
    if (bDebug)
      std::cout
          << "Debug: Opening with CAP_V4L2 failed, trying default backend."
          << std::endl;
    cap.open(camera_index);
    if (!cap.isOpened()) {
      std::string error_message = "OpenCV failed to open camera index " +
                                  Num2Str(camera_index).str() +
                                  ". Tried CAP_V4L2 and default.";
      THROWGEMERROR(error_message);
    }
  }

  std::cout << "Opened Camera Index: " << camera_index
            << " for Interactive Calibration..." << std::endl;

  if (!trySetCameraResolution(cap, g_capture_width, g_capture_height, true)) {
    int final_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int final_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::stringstream ss;
    ss << "Calibration: Failed to set desired resolution " << g_capture_width
       << "x" << g_capture_height << ". Actual resolution is " << final_width
       << "x" << final_height << ".";
    THROWGEMERROR(ss.str());
  }

  int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  if (bDebug) {
    std::cout
        << "Debug: Calibration - Proceeding with actual frame dimensions: "
        << frame_width << "x" << frame_height << std::endl;
  }

  std::string window_name = "Calibration - Adjust Corners (ESC: exit, S: save)";
  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

  cv::Mat frame;
  cv::Mat clean_frame_to_save;

  float init_percent_x = 15.0f;
  float init_percent_y = 15.0f;
  cv::Point2f topLeft(frame_width * init_percent_x / 100.0f,
                      frame_height * init_percent_y / 100.0f);
  cv::Point2f topRight(frame_width * (100.0f - init_percent_x) / 100.0f,
                       frame_height * init_percent_y / 100.0f);
  cv::Point2f bottomLeft(frame_width * init_percent_x / 100.0f,
                         frame_height * (100.0f - init_percent_y) / 100.0f);
  cv::Point2f bottomRight(frame_width * (100.0f - init_percent_x) / 100.0f,
                          frame_height * (100.0f - init_percent_y) / 100.0f);

  ActiveCorner currentActiveCorner = ActiveCorner::TOP_LEFT;

  // --- UPDATE Console Help Text ---
  std::cout << "\n--- Interactive Calibration ---" << std::endl;
  std::cout << "INSTRUCTIONS: Place BLACK stones near TL(1) & BL(3) markers."
            << std::endl;
  std::cout << "              Place WHITE stones near TR(2) & BR(4) markers."
            << std::endl;
  std::cout << "              Ensure the center area of the board is EMPTY."
            << std::endl; // Updated instruction
  std::cout << "ADJUSTMENT KEYS:" << std::endl;
  std::cout << "  SELECT CORNER: 1(TL), 2(TR), 3(BL), 4(BR)" << std::endl;
  std::cout << "  MOVE ACTIVE:   i(up), k(down), j(left), l(right)"
            << std::endl;
  std::cout << "ACTIONS:" << std::endl;
  std::cout << "  s: Save config & snapshot (with stones/empty center placed!)"
            << std::endl; // Updated instruction
  std::cout << "  esc: Exit without saving" << std::endl;
  std::cout << "------------------------------------" << std::endl;
  std::cout << "Currently Active Corner: TL (Top-Left)" << std::endl;

  while (true) {
    bool success = cap.read(frame);
    if (!success || frame.empty()) { /* ... error handling ... */
      continue;
    }

    clean_frame_to_save = frame.clone();
    cv::Mat display_frame = frame.clone();

    drawCalibrationOSD(display_frame, topLeft, topRight, bottomLeft,
                       bottomRight, currentActiveCorner);
    cv::imshow(window_name, display_frame);

    int key = cv::waitKey(30);
    int key_result = processCalibrationKeyPress(
        key, topLeft, topRight, bottomLeft, bottomRight, frame_width,
        frame_height, currentActiveCorner);

    if (key_result == 27) {
      break;
    } else if (key_result == 's') {
      std::cout << "Processing 'save' command..." << std::endl;

      cv::Mat frame_lab;
      cv::cvtColor(clean_frame_to_save, frame_lab, cv::COLOR_BGR2Lab);
      int sample_radius = 3;

      // Sample corner colors
      std::cout << "  Sampling corner stone colors..." << std::endl;
      cv::Vec3f lab_tl = getAverageLab(frame_lab, topLeft, sample_radius);
      cv::Vec3f lab_tr = getAverageLab(frame_lab, topRight, sample_radius);
      cv::Vec3f lab_bl = getAverageLab(frame_lab, bottomLeft, sample_radius);
      cv::Vec3f lab_br = getAverageLab(frame_lab, bottomRight, sample_radius);

      // --- NEW: Sample Board Color at Multiple Points and Average ---
      std::cout << "  Sampling empty board colors..." << std::endl;
      std::vector<cv::Point2f> board_sample_points;
      board_sample_points.push_back((topLeft + topRight) * 0.5f); // Mid-top
      board_sample_points.push_back((bottomLeft + bottomRight) *
                                    0.5f); // Mid-bottom
      board_sample_points.push_back((topLeft + bottomLeft) * 0.5f); // Mid-left
      board_sample_points.push_back((topRight + bottomRight) *
                                    0.5f); // Mid-right
      board_sample_points.push_back(
          (topLeft + topRight + bottomLeft + bottomRight) * 0.25f); // Centroid

      cv::Vec3f sum_lab_board(0, 0, 0);
      int valid_board_samples = 0;
      for (const auto &pt : board_sample_points) {
        if (pt.x >= 0 && pt.x < frame_width && pt.y >= 0 &&
            pt.y < frame_height) {
          cv::Vec3f sample = getAverageLab(frame_lab, pt, sample_radius);
          // Optional: Add a check here to discard sample if it's too close to
          // known stone colors, in case a stone was accidentally left at a
          // sample point. For now, we assume user followed instructions for
          // empty board areas.
          sum_lab_board += sample;
          valid_board_samples++;
          if (bDebug)
            std::cout << "    Sampled board at (" << pt.x << "," << pt.y
                      << "): Lab " << sample << std::endl;
        } else {
          if (bDebug)
            std::cout << "    Skipping board sample point (" << pt.x << ","
                      << pt.y << ") - out of bounds." << std::endl;
        }
      }

      cv::Vec3f avg_lab_board(-1, -1, -1); // Default if no valid samples
      if (valid_board_samples > 0) {
        avg_lab_board = sum_lab_board / static_cast<float>(valid_board_samples);
      } else {
        std::cerr
            << "Warning: Could not get any valid board color samples from the "
               "center/mid-edge points. Board color in config might be invalid."
            << std::endl;
      }
      // --- End NEW Board Sampling ---

      if (bDebug) {
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "  Debug: Sampled TL Lab: " << lab_tl << std::endl;
        std::cout << "  Debug: Sampled TR Lab: " << lab_tr << std::endl;
        std::cout << "  Debug: Sampled BL Lab: " << lab_bl << std::endl;
        std::cout << "  Debug: Sampled BR Lab: " << lab_br << std::endl;
        std::cout << "  Debug: Averaged Board Lab (" << valid_board_samples
                  << " samples): " << avg_lab_board << std::endl;
      }

      // --- Save Snapshot (as before) ---
      std::cout << "  Saving snapshot..." << std::endl;
      // ... (your existing snapshot saving logic for CALIB_SNAPSHOT_PATH /
      // CALIB_SNAPSHOT_DEBUG_PATH) ...
      bool saved_image_flag = false;
      std::string snapshot_path_to_use =
          bDebug ? CALIB_SNAPSHOT_DEBUG_PATH : CALIB_SNAPSHOT_PATH;
      saved_image_flag = cv::imwrite(
          snapshot_path_to_use, (bDebug ? display_frame : clean_frame_to_save));
      if (!saved_image_flag) {
        std::cerr << "Error: Failed to save snapshot to "
                  << snapshot_path_to_use << std::endl;
      } else {
        std::cout << "  Snapshot saved to " << snapshot_path_to_use
                  << std::endl;
      }

      // --- Save Config (pass new avg_lab_board) ---
      std::cout << "  Saving config file..." << std::endl;
      if (!saveCornerConfig(CALIB_CONFIG_PATH, topLeft, topRight, bottomLeft,
                            bottomRight, frame_width, frame_height, lab_tl,
                            lab_tr, lab_bl, lab_br,
                            avg_lab_board)) // Pass averaged board color
      {
        std::cerr << "Error: Failed to save config to " << CALIB_CONFIG_PATH
                  << std::endl;
      }
      // saveCornerConfig prints its own success message
      std::cout << "Save complete." << std::endl;
    }
  }

  cap.release();
  cv::destroyAllWindows();
  std::cout << "Calibration window closed." << std::endl;
}