#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
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
                      const cv::Point2f &br, int frame_width,
                      int frame_height) {
  // ... (code from previous response to save key=value) ...
  std::ofstream outFile(filename);
  if (!outFile.is_open()) { /* ... error handling ... */
    return false;
  }
  outFile << "# Go Board Corner Configuration..." << std::endl;
  outFile << "ImageWidth=" << frame_width << std::endl;
  outFile << "ImageHeight=" << frame_height << std::endl;
  outFile << std::fixed << std::setprecision(1);
  outFile << "\n# Pixel Coordinates" << std::endl;
  outFile << "TL_X_PX=" << tl.x << std::endl;
  outFile << "TL_Y_PX=" << tl.y << std::endl;
  outFile << "TR_X_PX=" << tr.x << std::endl;
  outFile << "TR_Y_PX=" << tr.y << std::endl;
  outFile << "BL_X_PX=" << bl.x << std::endl;
  outFile << "BL_Y_PX=" << bl.y << std::endl;
  outFile << "BR_X_PX=" << br.x << std::endl;
  outFile << "BR_Y_PX=" << br.y << std::endl;
  if (frame_width > 0 && frame_height > 0) {
    outFile << "\n# Percentage Coordinates (%)" << std::endl;
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
  std::cout << "Corner configuration saved to " << filename << std::endl;
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
  // Instead of cap.open(camera_index);
  cap.open(camera_index, cv::CAP_V4L2);
  if (!cap.isOpened()) {
    std::string error_message =
        "OpenCV (CAP_V4L2) failed to open video capture device with index " +
        Num2Str(camera_index).str() + ". " + "Device path hint: /dev/videoX " +
        "Possible reasons: \n"
        "1. Camera not connected or powered.\n"
        "2. Invalid camera index or device path.\n"
        "3. Camera is already in use by another application.\n"
        "4. Insufficient permissions (check /dev/videoX ownership/group, add "
        "user to 'video' group, or try sudo if appropriate).\n"
        "5. V4L2 backend issues or camera driver problems.\n"
        "6. OpenCV was not built with V4L2 support for this camera.";
    THROWGEMERROR(error_message);
  }
  if (bDebug)
    std::cout << "Debug: Calibration - Requesting frame size "
              << g_capture_width << "x" << g_capture_height << "." << std::endl;
  // Use the new utility function
  if (!trySetCameraResolution(cap, g_capture_width, g_capture_height, true)) {
    // Get current actuals for the error message
    int final_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int final_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::stringstream ss;
    ss << "Calibration: Failed to set desired resolution " << g_capture_width
       << "x" << g_capture_height
       << " even after fallback. Actual resolution is " << final_width << "x"
       << final_height << ".";
    THROWGEMERROR(ss.str());
  }
  // Proceed with these dimensions, as trySetCameraResolution confirmed them or
  // failed trying.
  int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  if (bDebug) {
    std::cout << "Debug: Calibration - Actual frame dimensions from camera: "
              << frame_width << "x" << frame_height << std::endl;
  }
  std::cout << "Opened Camera Index: " << camera_index << std::endl;
  std::cout << "Starting Interactive Calibration..." << std::endl;

  std::string window_name = "Calibration - Adjust Corners (ESC: exit, S: save)";
  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

  cv::Mat frame;
  cv::Mat clean_frame_to_save;

  // Initialize corners (as before)
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

  ActiveCorner currentActiveCorner =
      ActiveCorner::TOP_LEFT; // Default active corner

  std::cout << "Initial Corners Set. Use Keys to Adjust:" << std::endl;
  std::cout << "  SELECT CORNER: 1 (Top-Left), 2 (Top-Right), 3 (Bottom-Left), "
               "4 (Bottom-Right)"
            << std::endl;
  std::cout << "  MOVE ACTIVE CORNER:" << std::endl;
  std::cout << "    i: move UP" << std::endl;
  std::cout << "    k: move DOWN" << std::endl;
  std::cout << "    j: move LEFT" << std::endl;
  std::cout << "    l: move RIGHT" << std::endl;
  std::cout << "------------------------------------" << std::endl;
  std::cout << "  s: save current snapshot and config to ./share/" << std::endl;
  std::cout << "  esc: exit" << std::endl;
  std::cout << "Currently Active Corner: TL (Top-Left)" << std::endl;

  while (true) {
    bool success = cap.read(frame);
    if (!success || frame.empty()) { /* ... error handling ... */
      continue;
    }

    clean_frame_to_save = frame.clone();
    cv::Mat display_frame = frame.clone();

    // Pass currentActiveCorner to OSD
    drawCalibrationOSD(display_frame, topLeft, topRight, bottomLeft,
                       bottomRight, currentActiveCorner);
    cv::imshow(window_name, display_frame);

    int key = cv::waitKey(30);
    // Call the new key processing function
    int key_result = processCalibrationKeyPress(
        key, topLeft, topRight, bottomLeft, bottomRight, frame_width,
        frame_height, currentActiveCorner);

    if (key_result == 27)
      break;
    else if (key_result == 's') { // 's' pressed     
      std::cout << "Saving snapshot and config..." << std::endl;

      bool saved_image = false;   

      if (bDebug) {        
        saved_image = cv::imwrite(CALIB_SNAPSHOT_DEBUG_PATH, display_frame);
      } else {        
        saved_image = cv::imwrite(CALIB_SNAPSHOT_PATH, clean_frame_to_save);
      }
      if (!saved_image)
        std::cerr << "Error: Failed to save snapshot!" << std::endl;

      if (!saveCornerConfig(CALIB_CONFIG_PATH, topLeft, topRight, bottomLeft,
                            bottomRight, frame_width, frame_height)) {
        std::cerr << "Error: Failed to save config!" << std::endl;
      }
      std::cout << "Save complete." << std::endl;
    }
    // --- REMOVED the explicit getWindowProperty check ---
  }

  // --- Output Final Coordinates/Percentages (Same as before) ---
  // ...

  cap.release();
  cv::destroyAllWindows();
  std::cout << "Calibration window closed." << std::endl;
}