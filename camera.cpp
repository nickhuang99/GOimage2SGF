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
using namespace std;

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
const std::string CALIB_DEBUG_CONFIG_PATH = "./share/config_debug.txt";
const std::string CALIB_SNAPSHOT_PATH = "./share/snapshot.jpg";
const std::string CALIB_SNAPSHOT_RAW_PATH =
    "./share/snapshot_raw_calibration.jpg";
const std::string CALIB_SNAPSHOT_DEBUG_PATH = "./share/snapshot_osd.jpg";

const std::string WINDOW_RAW_FEED =
    "Raw Camera - Adjust Corners (1-4, ijkl, s, esc)";
const std::string WINDOW_CORRECTED_PREVIEW =
    "Corrected Preview - Align Stones/Grid";

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
    return key; // Or key if you want to pass unhandled ones
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

// --- NEW OSD for Corrected Preview Window ---
static void
drawCorrectedPreviewOSD(cv::Mat &corrected_frame, int preview_width,
                        int preview_height,
                        const std::vector<cv::Point2f> &corrected_dest_points,
                        int adaptive_radius) {
  cv::Point2f sample_pt_TL_stone = corrected_dest_points[0];
  cv::Point2f sample_pt_TR_stone = corrected_dest_points[1];
  cv::Point2f sample_pt_BR_stone = corrected_dest_points[2];
  cv::Point2f sample_pt_BL_stone = corrected_dest_points[3];

  int board_width = sample_pt_TR_stone.x - sample_pt_TL_stone.x;
  int board_height = sample_pt_BL_stone.y - sample_pt_TL_stone.y;
  cv::Scalar grid_color(100, 100, 100);
  float x_step = static_cast<float>(board_width - 1) / 18.0f;
  float y_step = static_cast<float>(board_height - 1) / 18.0f;
  float x_start = static_cast<float>(sample_pt_TL_stone.x);
  float y_start = static_cast<float>(sample_pt_TL_stone.y);
  float x_end = static_cast<float>(sample_pt_BR_stone.x);
  float y_end = static_cast<float>(sample_pt_BR_stone.y);

  for (int i = 0; i < 19; ++i) {
    cv::line(corrected_frame, cv::Point2f(x_start + i * x_step, y_start),
             cv::Point2f(x_start + i * x_step, y_end), grid_color, 1);
    cv::line(corrected_frame, cv::Point2f(x_start, y_start + i * y_step),
             cv::Point2f(x_end, y_start + i * y_step), grid_color, 1);
  }

  cv::Scalar white_marker_outline(255, 0, 0);
  cv::Scalar black_marker_outline(0, 0, 255);

  // TL (Black Stone Target)
  cv::circle(corrected_frame, sample_pt_TL_stone, adaptive_radius,
             black_marker_outline, 1);
  // TR (White Stone Target)
  cv::circle(corrected_frame, sample_pt_TR_stone, adaptive_radius,
             white_marker_outline, 1);
  // BR (White Stone Target)
  cv::circle(corrected_frame, sample_pt_BR_stone, adaptive_radius,
             white_marker_outline, 1);
  // BL (Black Stone Target)
  cv::circle(corrected_frame, sample_pt_BL_stone, adaptive_radius,
             black_marker_outline, 1);

  std::string help_text =
      "Align physical stones to CORNER markers. Grid should be straight.";
  cv::putText(corrected_frame, help_text, cv::Point(10, preview_height - 10),
              cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1,
              cv::LINE_AA);
}

// saveCornerConfig needs to take all Lab values
bool saveCornerConfig(
    const std::string &filename,
    const std::string &device_path_for_config,
    int frame_width, int frame_height, const cv::Point2f &tl_raw,
    const cv::Point2f &tr_raw, const cv::Point2f &bl_raw,
    const cv::Point2f &br_raw,
    const cv::Vec3f &standard_lab_tl_corrected,     // Data from sampleDataForConfig
    const cv::Vec3f &standard_lab_tr_corrected,
    const cv::Vec3f &standard_lab_bl_corrected,
    const cv::Vec3f &standard_lab_br_corrected,
    const cv::Vec3f &standard_avg_lab_board_corrected, // Data from sampleDataForConfig
    bool enhanced_data_is_available, // Flag
    const std::vector<cv::Vec3f>* enhanced_lab_colors_corrected_ptr, // Pointer to enhanced data
    float avg_radius_from_enhanced_detection_corrected_px // Enhanced radius
) {
  std::ofstream outFile(filename);
  if (!outFile.is_open()) {
    std::cerr << "Error: Could not open config file for writing: " << filename << std::endl;
    return false;
  }
  outFile << "# Go Board Calibration Configuration (vGEM2.1)" << std::endl; // Update version
  outFile << "# Generated by GEM" << std::endl;

  outFile << "\n# Device and Resolution at Calibration Time" << std::endl;
  outFile << "DevicePath=" << device_path_for_config << std::endl;
  outFile << "ImageWidth=" << frame_width << std::endl;
  outFile << "ImageHeight=" << frame_height << std::endl;

  outFile << std::fixed << std::setprecision(1);

  outFile << "\n# Corner Pixel Coordinates (Raw Image - TL, TR, BL, BR)" << std::endl;
  outFile << "TL_X_PX=" << tl_raw.x << std::endl;
  outFile << "TL_Y_PX=" << tl_raw.y << std::endl;
  outFile << "TR_X_PX=" << tr_raw.x << std::endl;
  outFile << "TR_Y_PX=" << tr_raw.y << std::endl;
  outFile << "BL_X_PX=" << bl_raw.x << std::endl;
  outFile << "BL_Y_PX=" << bl_raw.y << std::endl;
  outFile << "BR_X_PX=" << br_raw.x << std::endl;
  outFile << "BR_Y_PX=" << br_raw.y << std::endl;

  // Write STANDARD sampled colors with "STD_" prefix
  outFile << "\n# Standard Sampled Lab Colors from Corrected Image (Ideal Grid Points)" << std::endl;
  outFile << "STD_TL_L=" << standard_lab_tl_corrected[0] << std::endl;
  outFile << "STD_TL_A=" << standard_lab_tl_corrected[1] << std::endl;
  outFile << "STD_TL_B=" << standard_lab_tl_corrected[2] << std::endl;
  outFile << "STD_TR_L=" << standard_lab_tr_corrected[0] << std::endl;
  outFile << "STD_TR_A=" << standard_lab_tr_corrected[1] << std::endl;
  outFile << "STD_TR_B=" << standard_lab_tr_corrected[2] << std::endl;
  outFile << "STD_BL_L=" << standard_lab_bl_corrected[0] << std::endl;
  outFile << "STD_BL_A=" << standard_lab_bl_corrected[1] << std::endl;
  outFile << "STD_BL_B=" << standard_lab_bl_corrected[2] << std::endl;
  outFile << "STD_BR_L=" << standard_lab_br_corrected[0] << std::endl;
  outFile << "STD_BR_A=" << standard_lab_br_corrected[1] << std::endl;
  outFile << "STD_BR_B=" << standard_lab_br_corrected[2] << std::endl;
  outFile << "STD_BOARD_L_AVG=" << standard_avg_lab_board_corrected[0] << std::endl;
  outFile << "STD_BOARD_A_AVG=" << standard_avg_lab_board_corrected[1] << std::endl;
  outFile << "STD_BOARD_B_AVG=" << standard_avg_lab_board_corrected[2] << std::endl;

  if (frame_width > 0 && frame_height > 0) {
    outFile << "\n# Corner Percentage Coordinates (Raw Image %)" << std::endl;
    outFile << "TL_X_PC=" << (tl_raw.x / frame_width * 100.0f) << std::endl;
    outFile << "TL_Y_PC=" << (tl_raw.y / frame_height * 100.0f) << std::endl;
    outFile << "TR_X_PC=" << (tr_raw.x / frame_width * 100.0f) << std::endl;
    outFile << "TR_Y_PC=" << (tr_raw.y / frame_height * 100.0f) << std::endl;
    outFile << "BL_X_PC=" << (bl_raw.x / frame_width * 100.0f) << std::endl;
    outFile << "BL_Y_PC=" << (bl_raw.y / frame_height * 100.0f) << std::endl;
    outFile << "BR_X_PC=" << (br_raw.x / frame_width * 100.0f) << std::endl;
    outFile << "BR_Y_PC=" << (br_raw.y / frame_height * 100.0f) << std::endl;
  }

  // Conditionally write ENHANCED detection data
  if (enhanced_data_is_available &&
      avg_radius_from_enhanced_detection_corrected_px > 0 &&
      enhanced_lab_colors_corrected_ptr &&
      enhanced_lab_colors_corrected_ptr->size() == 4) {
    const std::vector<cv::Vec3f>& enhanced_colors = *enhanced_lab_colors_corrected_ptr;
    outFile << "\n# Enhanced Stone Detection Data (from Corrected Image)" << std::endl;
    outFile << "DETECTED_AVG_STONE_RADIUS_CORRECTED_PX=" << avg_radius_from_enhanced_detection_corrected_px << std::endl;
    outFile << "DETECTED_TL_L=" << enhanced_colors[0][0] << std::endl;
    outFile << "DETECTED_TL_A=" << enhanced_colors[0][1] << std::endl;
    outFile << "DETECTED_TL_B=" << enhanced_colors[0][2] << std::endl;
    outFile << "DETECTED_TR_L=" << enhanced_colors[1][0] << std::endl;
    outFile << "DETECTED_TR_A=" << enhanced_colors[1][1] << std::endl;
    outFile << "DETECTED_TR_B=" << enhanced_colors[1][2] << std::endl;
    outFile << "DETECTED_BL_L=" << enhanced_colors[2][0] << std::endl;
    outFile << "DETECTED_BL_A=" << enhanced_colors[2][1] << std::endl;
    outFile << "DETECTED_BL_B=" << enhanced_colors[2][2] << std::endl;
    outFile << "DETECTED_BR_L=" << enhanced_colors[3][0] << std::endl;
    outFile << "DETECTED_BR_A=" << enhanced_colors[3][1] << std::endl;
    outFile << "DETECTED_BR_B=" << enhanced_colors[3][2] << std::endl;
  } else if (enhanced_data_is_available) {
    outFile << "\n# Enhanced Stone Detection was attempted but data was incomplete/invalid." << std::endl;
  }
  outFile.close();
  if (!outFile) {
    std::cerr << "Error: Failed to properly close config file after writing: "
              << filename << std::endl;
    return false;
  }
  std::cout << "Configuration saved to " << filename << std::endl;
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

void sampleDataForConfig(const cv::Mat &final_corrected_lab,
                         const std::vector<cv::Point2f> &corrected_dest_points,
                         int adaptive_radius, std::vector<cv::Vec3f> &output) {
  // Sample points are the corners of the `corrected_dest_points`
  cv::Point2f sample_pt_TL_stone = corrected_dest_points[0];
  cv::Point2f sample_pt_TR_stone = corrected_dest_points[1];
  cv::Point2f sample_pt_BR_stone = corrected_dest_points[2];
  cv::Point2f sample_pt_BL_stone = corrected_dest_points[3];

  std::vector<cv::Point2f> board_sample_pts_corrected_final;
  board_sample_pts_corrected_final.clear(); // Start with an empty list

  // Assuming corrected_dest_points are [0]=TL, [1]=TR, [2]=BR, [3]=BL
  cv::Point2f c_tl = corrected_dest_points[0];
  cv::Point2f c_tr = corrected_dest_points[1];
  cv::Point2f c_br = corrected_dest_points[2];
  cv::Point2f c_bl = corrected_dest_points[3];

  // Calculate grid spacing in the corrected image
  // Use the full span of the board defined by these corners.
  // Note: This assumes a rectangular shape defined by these 4 points.
  // Width from TL.x to TR.x; Height from TL.y to BL.y
  float board_pixel_width = c_tr.x - c_tl.x;
  float board_pixel_height = c_bl.y - c_tl.y;

  if (board_pixel_width < 18.0f || board_pixel_height < 18.0f) {
    // Handle degenerate case, perhaps revert to old 5-point sampling or error
    THROWGEMERROR(
        "Corrected board dimensions too small for full empty space sampling.");
  }

  float x_step = board_pixel_width / 18.0f;
  float y_step = board_pixel_height / 18.0f;

  for (int r = 0; r < 19; ++r) {
    for (int c = 0; c < 19; ++c) {
      // Check if this is one of the four stone corners
      bool is_stone_corner = (r == 0 && c == 0) ||  // TL
                             (r == 0 && c == 18) || // TR
                             (r == 18 && c == 0) || // BL
                             (r == 18 && c == 18);  // BR

      if (!is_stone_corner) {
        cv::Point2f current_intersection_pt;
        current_intersection_pt.x = c_tl.x + c * x_step;
        current_intersection_pt.y = c_tl.y + r * y_step;
        board_sample_pts_corrected_final.push_back(current_intersection_pt);
      }
    }
  }

  // Stone colors based on their positions in the corrected standard view
  cv::Vec3f lab_tl_sampled = getAverageLab(
      final_corrected_lab, sample_pt_TL_stone, adaptive_radius); // Black
  cv::Vec3f lab_tr_sampled = getAverageLab(
      final_corrected_lab, sample_pt_TR_stone, adaptive_radius); // White
  cv::Vec3f lab_bl_sampled = getAverageLab(
      final_corrected_lab, sample_pt_BL_stone, adaptive_radius); // Black
  cv::Vec3f lab_br_sampled = getAverageLab(
      final_corrected_lab, sample_pt_BR_stone, adaptive_radius); // White

  cv::Vec3f sum_lab_board(0, 0, 0);
  int valid_board_s = 0;
  for (const auto &pt : board_sample_pts_corrected_final) {
    cv::Vec3f s = getAverageLab(final_corrected_lab, pt, adaptive_radius);
    sum_lab_board += s;
    valid_board_s++;
  }
  cv::Vec3f avg_lab_board_sampled =
      (valid_board_s > 0) ? (sum_lab_board / static_cast<float>(valid_board_s))
                          : cv::Vec3f(-1, -1, -1);
  output.push_back(lab_tl_sampled);
  output.push_back(lab_tr_sampled);
  output.push_back(lab_bl_sampled);
  output.push_back(lab_br_sampled);
  output.push_back(avg_lab_board_sampled);
}

static bool
verifyCalibrationAfterSave(const cv::Mat &raw_image_for_verification) {
  std::cout << "  Verifying calibration (strict check)..." << std::endl;

  if (raw_image_for_verification.empty()) {
    std::cerr
        << "    Verification Error: Input image for verification is empty."
        << std::endl;
    return false;
  }

  cv::Mat board_state_matrix;
  cv::Mat board_with_stones_display;
  std::vector<cv::Point2f> intersection_points;

  try {
    processGoBoard(raw_image_for_verification, board_state_matrix,
                   board_with_stones_display, intersection_points);

    if (board_state_matrix.empty() || board_state_matrix.rows != 19 ||
        board_state_matrix.cols != 19) {
      std::cerr << "    Verification Error: processGoBoard did not return a "
                   "valid 19x19 board state."
                << std::endl;
      return false;
    }

    for (int r = 0; r < 19; ++r) {
      for (int c = 0; c < 19; ++c) {
        // corner cases
        if ((r == 0 || r == 18) && (c == 0 || c == 18)) {
          int color = board_state_matrix.at<uchar>(r, c);
          bool black_correct =
              color == BLACK && (r == 0 && c == 0 || r == 18 && c == 0);
          bool white_correct =
              color == WHITE && (r == 0 && c == 18 || r == 18 && c == 18);
          if (!black_correct && !white_correct) {
            cout << "stone detection failed at [" << r << "," << c << "]"
                 << color << endl;
            if (bDebug) {
              cv::imshow("Calibration Verification stones detection failed",
                         board_with_stones_display);
              cv::waitKey(0);
            }
            return false;
          }
        } else { // non-corner cases, must be empty
          if (board_state_matrix.at<uchar>(r, c) != EMPTY) {
            cout << "verifyCalibrationAfterSave: not empty at[" << r << "," << c
                 << "]" << endl;
            if (bDebug) {
              cv::imshow("Calibration Verification empty detection failed",
                         board_with_stones_display);
              cv::waitKey(0);
            }
            return false;
          }
        }
      }
    }
    return true;
  } catch (const GEMError &ge) { // Catch specific GEMError from processGoBoard
    std::cerr << "    Verification GEMError: " << ge.what() << std::endl;
    return false;
  } catch (const std::exception &e) { // Catch other standard exceptions
    std::cerr << "    Verification Exception: " << e.what() << std::endl;
    return false;
  }
}

bool processAndSaveCalibration(
    const cv::Mat &final_raw_bgr_for_snapshot,
    const std::vector<cv::Point2f> &current_source_points,
    // --- New parameters for enhanced data ---
    bool enhanced_detection_was_successful,
    const std::vector<cv::Vec3f> *enhanced_lab_colors, // Pointer
    float enhanced_avg_radius_px) {

  int frame_width = final_raw_bgr_for_snapshot.cols;
  int frame_height = final_raw_bgr_for_snapshot.rows;

  std::cout << "  processAndSaveCalibration: Starting..." << std::endl;
  if (enhanced_detection_was_successful) {
    std::cout << "    Enhanced detection data IS available." << std::endl;
    if (enhanced_lab_colors) {
      std::cout << "    Enhanced Avg Radius: " << enhanced_avg_radius_px
                << std::endl;
      // Optionally print the enhanced colors here if bDebug
    }
  } else {
    std::cout << "    Enhanced detection data IS NOT available. Using standard "
                 "sampling."
              << std::endl;
  }

  std::vector<cv::Point2f> corrected_dest_points =
      getBoardCornersCorrected(frame_width, frame_height);

  cv::Mat transform_matrix =
      cv::getPerspectiveTransform(current_source_points, corrected_dest_points);

  cv::Mat final_corrected_bgr_for_sampling;
  cv::warpPerspective(final_raw_bgr_for_snapshot,
                      final_corrected_bgr_for_sampling, transform_matrix,
                      cv::Size(frame_width, frame_height));

  cv::imwrite(CALIB_SNAPSHOT_RAW_PATH, final_raw_bgr_for_snapshot);
  cv::imwrite(CALIB_SNAPSHOT_PATH, final_corrected_bgr_for_sampling);
  std::cout << "  Raw calibration snapshot saved to " << CALIB_SNAPSHOT_RAW_PATH
            << std::endl;
  std::cout << "  Corrected calibration snapshot (used for standard color "
               "sampling) saved to "
            << CALIB_SNAPSHOT_PATH << std::endl;

  cv::Mat final_corrected_lab;
  cv::cvtColor(final_corrected_bgr_for_sampling, final_corrected_lab,
               cv::COLOR_BGR2Lab);

  int correct_board_width_px =
      corrected_dest_points[1].x - corrected_dest_points[0].x;
  int correct_board_height_px =
      corrected_dest_points[3].y - corrected_dest_points[0].y;
  int adaptive_radius_corrected = calculateAdaptiveSampleRadius(
      correct_board_width_px, correct_board_height_px);

  std::vector<cv::Vec3f>
      standard_sample_data; // TL, TR, BL, BR, BoardAvg from corrected
  sampleDataForConfig(final_corrected_lab, corrected_dest_points,
                      adaptive_radius_corrected, standard_sample_data);

  // Now, call saveCornerConfig, providing both standard and (if available)
  // enhanced data
  bool save_success = saveCornerConfig(
      CALIB_CONFIG_PATH, g_device_path, frame_width, frame_height,
      current_source_points[0], current_source_points[1], // TL_raw, TR_raw
      current_source_points[3],
      current_source_points[2], // BL_raw, BR_raw (order for saving)
      standard_sample_data[0],
      standard_sample_data[1], // TL_corrected_lab, TR_corrected_lab
      standard_sample_data[2],
      standard_sample_data[3], // BL_corrected_lab, BR_corrected_lab
      standard_sample_data[4], // BoardAvg_corrected_lab
      // --- Enhanced data part ---
      enhanced_detection_was_successful, // Flag indicating if enhanced data is
                                         // valid
      enhanced_detection_was_successful
          ? enhanced_lab_colors
          : nullptr, // Pass pointer to enhanced colors or nullptr
      enhanced_detection_was_successful ? enhanced_avg_radius_px
                                        : -1.0f // Pass enhanced radius or -1
  );

  if (!save_success) {
    std::cerr << "  Error: saveCornerConfig failed." << std::endl;
    return false; // Propagate save failure
  }

  std::cout << "  Calibration data saved to " << CALIB_CONFIG_PATH << std::endl;

  // Perform verification (uses the newly saved config.txt)
  bool verification_passed =
      verifyCalibrationAfterSave(final_raw_bgr_for_snapshot);
  if (verification_passed) {
    std::cout
        << "  Calibration VERIFIED successfully! Corner stones match expected."
        << std::endl;
  } else {
    std::cout << "  Calibration VERIFICATION FAILED after saving!" << std::endl;
  }
  return verification_passed; // Return status of verification
}

// Helper function to detect a single corner stone for enhanced calibration
// Operates on the PERSPECTIVE-CORRECTED image.
static bool detectAndSampleCornerStone(
    const cv::Mat
        &corrected_bgr_image, // Input: Perspective-corrected BGR image
    cv::Mat &display_image_for_debug_drawing, // Input/Output: Image for drawing
                                              // debug markers (can be a clone
                                              // of corrected_bgr_image)
    const cv::Point2f
        &target_corner_coord_corrected, // Input: Expected corner coordinate in
                                        // the corrected image (e.g.,
                                        // corrected_dest_points[0])
    int expected_stone_color,           // Input: BLACK or WHITE
    float
        approx_grid_spacing_corrected, // Input: Approximate grid spacing in the
                                       // corrected image to define ROI size
    cv::Point2f
        &out_detected_center_corrected,   // Output: Detected center in
                                          // corrected_bgr_image coordinates
    float &out_detected_radius_corrected, // Output: Detected radius in
                                          // corrected_bgr_image pixels
    cv::Vec3f &out_sampled_lab_color_corrected, // Output: Sampled Lab color
                                                // from this stone
    bool enhanced_detection_debug_flag) {       // Input: Flag to control debug
                                                // messages and drawing

  cv::Point2f detected_center_in_roi; // Center relative to the ROI within
                                      // corrected_bgr_image
  float detected_radius_in_roi;       // Radius relative to the ROI

  // Define ROI around the target_corner_coord_corrected
  float roi_half_width = approx_grid_spacing_corrected *
                         1.25f; // ROI slightly larger than one grid cell
  cv::Rect roi_on_corrected_image(
      std::max(0, static_cast<int>(target_corner_coord_corrected.x -
                                   roi_half_width)),
      std::max(0, static_cast<int>(target_corner_coord_corrected.y -
                                   roi_half_width)),
      static_cast<int>(roi_half_width * 2),
      static_cast<int>(roi_half_width * 2));
  // Ensure ROI is within the corrected image bounds
  roi_on_corrected_image &=
      cv::Rect(0, 0, corrected_bgr_image.cols, corrected_bgr_image.rows);

  if (roi_on_corrected_image.width <= 0 || roi_on_corrected_image.height <= 0) {
    if (bDebug || enhanced_detection_debug_flag) {
      std::cerr << "  Warning (detectAndSampleCornerStone): ROI for "
                << (expected_stone_color == BLACK ? "Black" : "White")
                << " stone at corrected " << target_corner_coord_corrected
                << " is invalid or out of bounds." << std::endl;
    }
    return false;
  }

  // detectColoredRoundShape takes BGR, applies its own Lab conversion and
  // returns center/radius relative to the input image (which is
  // corrected_bgr_image here)
  if (detectColoredRoundShape(corrected_bgr_image, roi_on_corrected_image,
                              expected_stone_color,
                              out_detected_center_corrected,
                              out_detected_radius_corrected, nullptr)) {

    // Re-sample color from the detected stone in the corrected LAB image
    cv::Mat corrected_lab_image;
    cv::cvtColor(corrected_bgr_image, corrected_lab_image, cv::COLOR_BGR2Lab);
    out_sampled_lab_color_corrected =
        getAverageLab(corrected_lab_image, out_detected_center_corrected,
                      static_cast<int>(out_detected_radius_corrected));

    if (bDebug || enhanced_detection_debug_flag) {
      // Draw on the provided display image
      cv::circle(display_image_for_debug_drawing, out_detected_center_corrected,
                 static_cast<int>(out_detected_radius_corrected),
                 cv::Scalar(0, 255, 0), 2); // Green circle
      cv::rectangle(display_image_for_debug_drawing, roi_on_corrected_image,
                    cv::Scalar(0, 255, 0), 1); // Green ROI
      std::cout << "  Successfully detected "
                << (expected_stone_color == BLACK ? "Black" : "White")
                << " stone at/near corrected " << target_corner_coord_corrected
                << ". Detected center (corrected): "
                << out_detected_center_corrected
                << ", radius (corrected): " << out_detected_radius_corrected
                << ", Sampled Lab: " << out_sampled_lab_color_corrected
                << std::endl;
    }
    return true;
  } else {
    if (bDebug || enhanced_detection_debug_flag) {
      cv::rectangle(display_image_for_debug_drawing, roi_on_corrected_image,
                    cv::Scalar(0, 0, 255), 1); // Red ROI
      std::cout << "  Warning: Failed to detect "
                << (expected_stone_color == BLACK ? "Black" : "White")
                << " stone via detectColoredRoundShape at/near corrected "
                << target_corner_coord_corrected << std::endl;
    }
    return false;
  }
}

// Helper function to detect a single corner stone for enhanced calibration
static bool detectAndProcessSingleCornerStone(
    const cv::Mat &corrected_bgr_image, // The perspective-corrected image
    cv::Mat
        &display_image_for_debug_drawing, // Image to draw debug markers on (can
                                          // be corrected_bgr_image or a clone)
    const cv::Point2f &corner_coord_corrected, // Expected corner coordinate in
                                               // the corrected image
    int expected_stone_color,                  // BLACK or WHITE
    float grid_spacing_corrected, // Approx grid spacing in corrected image for
                                  // ROI
    cv::Point2f &out_detected_center, // Output: detected center in original
                                      // corrected_bgr_image coordinates
    float &out_detected_radius,       // Output: detected radius
    cv::Vec3f
        &out_sampled_lab_color, // Output: sampled Lab color from this stone
    bool enhanced_detection_debug_flag) { // To control debug prints/drawings

  cv::Point2f detected_center_roi; // Center relative to ROI
  float detected_radius_roi;

  // Define ROI around the corrected corner coordinate
  float roi_half_width =
      grid_spacing_corrected * 1.5f; // ROI 1.5x grid cell size
  cv::Rect roi_corrected(
      std::max(0, static_cast<int>(corner_coord_corrected.x - roi_half_width)),
      std::max(0, static_cast<int>(corner_coord_corrected.y - roi_half_width)),
      static_cast<int>(roi_half_width * 2),
      static_cast<int>(roi_half_width * 2));
  // Ensure ROI is within the corrected image bounds
  roi_corrected &=
      cv::Rect(0, 0, corrected_bgr_image.cols, corrected_bgr_image.rows);

  if (roi_corrected.width <= 0 || roi_corrected.height <= 0) {
    if (bDebug || enhanced_detection_debug_flag) {
      std::cerr << "  Warning (detectAndProcessSingleCornerStone): ROI for "
                << (expected_stone_color == BLACK ? "Black" : "White")
                << " stone at " << corner_coord_corrected
                << " is invalid or out of bounds." << std::endl;
    }
    return false;
  }

  // Pass nullptr for calibData, as we're using predefined color ranges or want
  // to sample fresh. detectColoredRoundShape operates on BGR, then converts to
  // Lab internally.
  if (detectColoredRoundShape(corrected_bgr_image, roi_corrected,
                              expected_stone_color, detected_center_roi,
                              detected_radius_roi, nullptr)) {

    out_detected_center =
        detected_center_roi; // Already in corrected_bgr_image coordinates
                             // because detectColoredRoundShape adjusts it
    out_detected_radius = detected_radius_roi;

    // Re-sample color from the detected stone in the corrected LAB image
    cv::Mat corrected_lab_image;
    cv::cvtColor(corrected_bgr_image, corrected_lab_image, cv::COLOR_BGR2Lab);
    out_sampled_lab_color =
        getAverageLab(corrected_lab_image, out_detected_center,
                      static_cast<int>(out_detected_radius));

    if (bDebug || enhanced_detection_debug_flag) {
      cv::circle(display_image_for_debug_drawing, out_detected_center,
                 static_cast<int>(out_detected_radius), cv::Scalar(0, 255, 0),
                 2); // Green
      cv::rectangle(display_image_for_debug_drawing, roi_corrected,
                    cv::Scalar(0, 255, 0), 1); // Green ROI
      std::cout << "  Successfully detected "
                << (expected_stone_color == BLACK ? "Black" : "White")
                << " stone at corrected " << corner_coord_corrected
                << ". Detected center: " << out_detected_center
                << ", radius: " << out_detected_radius
                << ", Lab: " << out_sampled_lab_color << std::endl;
    }
    return true;
  } else {
    if (bDebug || enhanced_detection_debug_flag) {
      cv::rectangle(display_image_for_debug_drawing, roi_corrected,
                    cv::Scalar(0, 0, 255), 1); // Red ROI
      std::cout << "  Warning: Failed to detect "
                << (expected_stone_color == BLACK ? "Black" : "White")
                << " stone via detectColoredRoundShape at corrected "
                << corner_coord_corrected << std::endl;
    }
    return false;
  }
}

// --- SIMPLIFIED Main Calibration Function ---
void runInteractiveCalibration(int camera_index) {
  cv::VideoCapture cap;
  cap.open(camera_index, cv::CAP_V4L2); // Try V4L2 first
  if (!cap.isOpened()) {
    if (bDebug)
      std::cout
          << "Debug: Opening with CAP_V4L2 failed, trying default backend."
          << std::endl;
    cap.open(camera_index); // Fallback to default
    if (!cap.isOpened()) {
      THROWGEMERROR("OpenCV failed to open camera index " +
                    Num2Str(camera_index).str());
    }
  }
  std::cout << "Opened Camera Index: " << camera_index
            << " for Interactive Calibration." << std::endl;

  if (!trySetCameraResolution(cap, g_capture_width, g_capture_height, true)) {
    // Error/warning already printed by trySetCameraResolution if bDebug
    // Allow to proceed with whatever resolution was set, or throw if
    // strictness is needed
  }
  int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  if (bDebug)
    std::cout << "Debug: Calibration - Using frame dimensions: " << frame_width
              << "x" << frame_height << std::endl;

  cv::namedWindow(WINDOW_RAW_FEED, cv::WINDOW_AUTOSIZE);
  cv::namedWindow(WINDOW_CORRECTED_PREVIEW, cv::WINDOW_AUTOSIZE);
  cv::moveWindow(WINDOW_RAW_FEED, 0, 0);
  // Position windows side-by-side
  cv::moveWindow(WINDOW_CORRECTED_PREVIEW, frame_width, 0);

  cv::Mat raw_frame, display_raw, display_corrected_bgr;

  // Initialize corners for the raw feed
  cv::Point2f topLeft_raw(frame_width * 0.15f, frame_height * 0.15f);
  cv::Point2f topRight_raw(frame_width * 0.85f, frame_height * 0.15f);
  cv::Point2f bottomLeft_raw(frame_width * 0.15f, frame_height * 0.85f);
  cv::Point2f bottomRight_raw(frame_width * 0.85f, frame_height * 0.85f);
  ActiveCorner currentActiveCorner = ActiveCorner::TOP_LEFT;

  // Define destination points for the corrected preview
  std::vector<cv::Point2f> corrected_dest_points =
      getBoardCornersCorrected(frame_width, frame_height);

  std::cout << "\n--- Interactive Calibration (Live Corrected Preview) ---"
            << std::endl;
  // ... (Print full instructions as per your previous good example) ...
  std::cout << "INSTRUCTIONS:" << std::endl;
  std::cout << "1. Place BLACK stones near physical Top-Left & Bottom-Left "
               "board corners."
            << std::endl;
  std::cout << "2. Place WHITE stones near physical Top-Right & Bottom-Right "
               "board corners."
            << std::endl;
  std::cout << "3. Ensure board mid-points & center (marked orange in preview) "
               "are EMPTY."
            << std::endl;
  std::cout << "4. Adjust markers in '" << WINDOW_RAW_FEED
            << "' (using 1-4, ijkl keys)" << std::endl;
  std::cout << "   until physical stones in your camera view appear at the"
            << std::endl;
  std::cout << "   CORNER MARKERS in the '" << WINDOW_CORRECTED_PREVIEW
            << "' window," << std::endl;
  std::cout << "   and the grid there looks straight and uniform." << std::endl;
  std::cout << "ACTIONS: 's' to save, 'esc' to exit." << std::endl;
  std::cout << "------------------------------------" << std::endl;

  int correct_board_width =
      corrected_dest_points[1].x - corrected_dest_points[0].x;
  int correct_board_height =
      corrected_dest_points[3].y - corrected_dest_points[0].y;
  int adaptive_radius =
      calculateAdaptiveSampleRadius(correct_board_width, correct_board_height);
  if (bDebug)
    std::cout << "  Sampling Lab colors from CORRECTED image using radius: "
              << adaptive_radius << std::endl;

  while (true) {
    if (!cap.read(raw_frame) || raw_frame.empty()) {
      std::cerr << "Error: Could not read frame from camera." << std::endl;
      cv::waitKey(100);
      continue; // Prevent tight loop on error
    }

    display_raw = raw_frame.clone();
    drawCalibrationOSD(display_raw, topLeft_raw, topRight_raw, bottomLeft_raw,
                       bottomRight_raw, currentActiveCorner);
    cv::imshow(WINDOW_RAW_FEED, display_raw);

    std::vector<cv::Point2f> current_source_points = {
        topLeft_raw, topRight_raw, bottomRight_raw, bottomLeft_raw};

    cv::Mat transform_matrix = cv::getPerspectiveTransform(
        current_source_points, corrected_dest_points);
    cv::warpPerspective(display_raw, display_corrected_bgr, transform_matrix,
                        cv::Size(frame_width, frame_height));

    drawCorrectedPreviewOSD(display_corrected_bgr, frame_width, frame_height,
                            corrected_dest_points, adaptive_radius);
    cv::imshow(WINDOW_CORRECTED_PREVIEW, display_corrected_bgr);

    int key = cv::waitKey(30);
    int key_action_type = processCalibrationKeyPress(
        key, topLeft_raw, topRight_raw, bottomLeft_raw, bottomRight_raw,
        frame_width, frame_height, currentActiveCorner);

    if (key_action_type == 27) { // ESC
      std::cout << "Calibration cancelled by user." << std::endl;
      break;
    } else if (key_action_type == 's') {
      std::cout << "Processing 'save' command with enhanced stone detection..."
                << std::endl;
      cv::Mat final_raw_bgr_for_snapshot =
          raw_frame.clone(); // Keep the raw frame for standard saving

      // Generate the corrected image based on current_source_points for
      // enhanced detection
      std::vector<cv::Point2f> current_source_points = {
          topLeft_raw, topRight_raw, bottomRight_raw, bottomLeft_raw};
      // corrected_dest_points is already defined (TL, TR, BR, BL order)
      cv::Mat current_transform_matrix = cv::getPerspectiveTransform(
          current_source_points, corrected_dest_points);
      cv::Mat final_corrected_bgr_for_enhanced_detection;
      cv::warpPerspective(final_raw_bgr_for_snapshot,
                          final_corrected_bgr_for_enhanced_detection,
                          current_transform_matrix,
                          cv::Size(frame_width, frame_height));

      cv::Mat display_enhanced_detection_on_corrected =
          final_corrected_bgr_for_enhanced_detection
              .clone(); // For drawing detection results

      bool enhanced_detection_overall_success = true;
      std::vector<cv::Point2f> detected_stone_centers_corrected(
          4); // Centers in corrected image pixels
      std::vector<float> detected_stone_radii_corrected(
          4); // Radii in corrected image pixels
      std::vector<cv::Vec3f> new_sampled_lab_colors_from_corrected(
          4); // Lab colors from corrected image stones
      float sum_detected_radii_corrected = 0.0f;
      int detected_stones_count = 0;

      // Estimate grid spacing in the corrected image (used for ROI definition)
      // corrected_dest_points order is TL, TR, BR, BL
      float corrected_grid_spacing_x =
          (corrected_dest_points[1].x - corrected_dest_points[0].x) / 18.0f;
      float corrected_grid_spacing_y =
          (corrected_dest_points[3].y - corrected_dest_points[0].y) /
          18.0f; // BL.y - TL.y
      float avg_corrected_grid_spacing =
          (corrected_grid_spacing_x + corrected_grid_spacing_y) / 2.0f;

      bool enhanced_detection_debug_output = true; // Or use bDebug

      // Corner order for processing: TL, TR, BL, BR (to match typical UI/mental
      // model) Target points in corrected_dest_points: TL=[0], TR=[1], BR=[2],
      // BL=[3]
      const std::vector<cv::Point2f> &targets = corrected_dest_points;
      const int corner_indices[] = {
          0, 1, 3,
          2}; // TL, TR, BL, BR in corrected_dest_points order for processing
      const int stone_colors[] = {BLACK, WHITE, BLACK, WHITE};
      const std::string corner_names[] = {
          "Top-Left (Black)", "Top-Right (White)", "Bottom-Left (Black)",
          "Bottom-Right (White)"};

      for (int i = 0; i < 4; ++i) {
        int corner_idx_in_corrected_dest =
            corner_indices[i]; // e.g. 0 for TL, 1 for TR, 3 for BL, 2 for BR
        int expected_color = stone_colors[i];

        std::cout << "  Attempting detection for: " << corner_names[i]
                  << " at corrected coord: "
                  << targets[corner_idx_in_corrected_dest] << std::endl;

        if (detectAndSampleCornerStone(
                final_corrected_bgr_for_enhanced_detection,
                display_enhanced_detection_on_corrected,
                targets[corner_idx_in_corrected_dest], expected_color,
                avg_corrected_grid_spacing,
                detected_stone_centers_corrected
                    [i], // Store based on 0=TL, 1=TR, 2=BL, 3=BR loop order
                detected_stone_radii_corrected[i],
                new_sampled_lab_colors_from_corrected[i],
                enhanced_detection_debug_output)) {
          sum_detected_radii_corrected += detected_stone_radii_corrected[i];
          detected_stones_count++;
        } else {
          enhanced_detection_overall_success = false;
        }
      }

      float final_average_detected_radius_corrected = -1.0f;
      if (enhanced_detection_overall_success && detected_stones_count == 4) {
        final_average_detected_radius_corrected =
            sum_detected_radii_corrected / 4.0f;
        std::cout << "  SUCCESS: Enhanced stone detection for all 4 corners on "
                     "corrected image."
                  << std::endl;
        std::cout
            << "  Average DETECTED STONE RADIUS (corrected image pixels): "
            << final_average_detected_radius_corrected << std::endl;
        // Debug print the sampled colors
        if (bDebug || enhanced_detection_debug_output) {
          for (int i = 0; i < 4; ++i) {
            std::cout << "    " << corner_names[i] << " - Detected Center: "
                      << detected_stone_centers_corrected[i]
                      << ", Radius: " << detected_stone_radii_corrected[i]
                      << ", Sampled Lab: "
                      << new_sampled_lab_colors_from_corrected[i] << std::endl;
          }
        }
      } else {
        std::cout
            << "  WARNING: Enhanced stone detection FAILED or was incomplete ("
            << detected_stones_count
            << "/4) on corrected image. Standard calibration data will "
               "primarily be used."
            << std::endl;
        enhanced_detection_overall_success = false; // Ensure this is false
      }

      if (bDebug || enhanced_detection_debug_output) {
        cv::imshow("Enhanced Detection on Corrected Image (Debug)",
                   display_enhanced_detection_on_corrected);
        // cv::waitKey(0); // Optional: pause to inspect
      }

      // Call processAndSaveCalibration
      // It uses `current_source_points` (raw coords) and
      // `final_raw_bgr_for_snapshot` for standard calib. It will now also
      // receive the enhanced data (radius and colors from *corrected* image).
      if (processAndSaveCalibration(
              final_raw_bgr_for_snapshot, current_source_points,
              enhanced_detection_overall_success, // True only if all 4 stones
                                                  // were found by new method
              (enhanced_detection_overall_success
                   ? &new_sampled_lab_colors_from_corrected
                   : nullptr),
              final_average_detected_radius_corrected // This is avg radius in
                                                      // corrected image pixels,
                                                      // or -1
              )) {
        std::cout
            << "Interactive calibration finished and VERIFIED successfully."
            << std::endl;
        break;
      } else {
        std::cout << "  Interactive calibration saved, but VERIFICATION FAILED "
                     "or save process had issues."
                  << std::endl;
        std::cout << "  ADVICE: Please check stone placement, lighting, and "
                     "corner markers, then try saving again."
                  << std::endl;
      }
      // --- END NEW INTEGRATION ---
    }
  }
  cap.release();
  cv::destroyAllWindows();
}

void runCaptureCalibration(int camera_index) {
  cv::VideoCapture cap;
  cap.open(camera_index, cv::CAP_V4L2); // Try V4L2 first
  if (!cap.isOpened()) {
    if (bDebug)
      std::cout
          << "Debug: Opening with CAP_V4L2 failed, trying default backend."
          << std::endl;
    cap.open(camera_index); // Fallback to default
    if (!cap.isOpened()) {
      THROWGEMERROR("OpenCV failed to open camera index " +
                    Num2Str(camera_index).str());
    }
  }
  std::cout << "Opened Camera Index: " << camera_index
            << " for Interactive Calibration." << std::endl;

  if (!trySetCameraResolution(cap, g_capture_width, g_capture_height, true)) {
    std::string err = "cannot set capture frame size of ";
    err +=
        Num2Str(g_capture_width).str() + "x" + Num2Str(g_capture_height).str();
    THROWGEMERROR(err);
  }

  cv::namedWindow(WINDOW_RAW_FEED, cv::WINDOW_AUTOSIZE);
  cv::moveWindow(WINDOW_RAW_FEED, 0, 0);

  cv::Mat raw_frame;

  // Define destination points for the corrected preview

  while (true) {
    if (!cap.read(raw_frame) || raw_frame.empty()) {
      std::cerr << "Error: Could not read frame from camera." << std::endl;
      cv::waitKey(100);
      continue; // Prevent tight loop on error
    }

    cv::Mat input_image = raw_frame.clone();

    cv::imshow(WINDOW_RAW_FEED, input_image);
    int key_action_type = cv::waitKey(30);
    if (key_action_type == 27) { // ESC
      std::cout << "Calibration cancelled by user." << std::endl;
      break;
    } else if (key_action_type == 's') {
      std::cout << "Processing 'save' command..." << std::endl;
      std::vector<cv::Point2f> detected_corners;
      if (detectFourCornersGoBoard(input_image, detected_corners)) {
        // if (processAndSaveCalibration(
        //         input_image, detected_corners,
        //         enhanced_detection_overall_success, // True only if all 4 stones
        //                                             // were found by new method
        //         (enhanced_detection_overall_success
        //              ? &new_sampled_lab_colors_from_corrected
        //              : nullptr),
        //         final_average_detected_radius_corrected // This is avg radius in
        //                                                 // corrected image
        //                                                 // pixels, or -1
        //         )) {
        //   cout << "calibration succeed!" << endl;
        //   break;
        // } else {
        //   std::cout << "  Calibration VERIFICATION FAILED! Detected corner "
        //                "stones do not match expected."
        //             << std::endl;
        //   std::cout << "  ADVICE: Please check stone placement (Black at TL & "
        //                "BL, White at TR & BR),"
        //             << std::endl;
        //   std::cout << "          lighting conditions, and ensure corners are "
        //                "accurately marked."
        //             << std::endl;
        //   std::cout << "          Consider re-adjusting and saving again, or "
        //                "exiting (ESC) to retry later."
        //             << std::endl;
        // }
      } else {
        cout << "detectFourCornersGoBoard failed please place two black stone "
                "at corner of TL BL "
             << "and two white stones at TR BR" << endl;
      }
    }
  }
  cap.release();
  cv::destroyAllWindows();
}