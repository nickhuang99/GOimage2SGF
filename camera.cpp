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
bool saveCornerConfig(const std::string &filename, const cv::Point2f &tl_raw,
                      const cv::Point2f &tr_raw, const cv::Point2f &bl_raw,
                      const cv::Point2f &br_raw, int frame_width,
                      int frame_height, const cv::Vec3f &lab_tl_sampled,
                      const cv::Vec3f &lab_tr_sampled,
                      const cv::Vec3f &lab_bl_sampled,
                      const cv::Vec3f &lab_br_sampled,
                      const cv::Vec3f &avg_lab_board_sampled) {
  std::ofstream outFile(filename);
  if (!outFile.is_open()) { /* error handling */
    return false;
  }
  outFile << "# Go Board Calibration Configuration" << std::endl;
  outFile << "ImageWidth=" << frame_width << std::endl;
  outFile << "ImageHeight=" << frame_height << std::endl;
  outFile << std::fixed << std::setprecision(1);

  outFile << "\n# Corner Pixel Coordinates (Raw Image - TL, TR, BL, BR)"
          << std::endl;
  outFile << "TL_X_PX=" << tl_raw.x << std::endl;
  outFile << "TL_Y_PX=" << tl_raw.y << std::endl;
  outFile << "TR_X_PX=" << tr_raw.x << std::endl;
  outFile << "TR_Y_PX=" << tr_raw.y << std::endl;
  outFile << "BL_X_PX=" << bl_raw.x << std::endl;
  outFile << "BL_Y_PX=" << bl_raw.y << std::endl;
  outFile << "BR_X_PX=" << br_raw.x << std::endl;
  outFile << "BR_Y_PX=" << br_raw.y << std::endl;

  outFile << "\n# Sampled Lab Colors from Corrected Image (L:0-255, A:0-255, "
             "B:0-255)"
          << std::endl;
  outFile << "# Black stones expected at TL, BL physical locations."
          << std::endl;
  outFile << "# White stones expected at TR, BR physical locations."
          << std::endl;
  outFile << "TL_L=" << lab_tl_sampled[0] << std::endl;
  outFile << "TL_A=" << lab_tl_sampled[1] << std::endl;
  outFile << "TL_B=" << lab_tl_sampled[2] << std::endl;
  outFile << "TR_L=" << lab_tr_sampled[0] << std::endl;
  outFile << "TR_A=" << lab_tr_sampled[1] << std::endl;
  outFile << "TR_B=" << lab_tr_sampled[2] << std::endl;
  outFile << "BL_L=" << lab_bl_sampled[0] << std::endl;
  outFile << "BL_A=" << lab_bl_sampled[1] << std::endl;
  outFile << "BL_B=" << lab_bl_sampled[2] << std::endl;
  outFile << "BR_L=" << lab_br_sampled[0] << std::endl;
  outFile << "BR_A=" << lab_br_sampled[1] << std::endl;
  outFile << "BR_B=" << lab_br_sampled[2] << std::endl;

  outFile << "\n# Sampled Average Empty Board Lab Color (from Corrected Image)"
          << std::endl;
  outFile << "BOARD_L_AVG=" << avg_lab_board_sampled[0] << std::endl;
  outFile << "BOARD_A_AVG=" << avg_lab_board_sampled[1] << std::endl;
  outFile << "BOARD_B_AVG=" << avg_lab_board_sampled[2] << std::endl;

  // Percentage coordinates are of the RAW corners
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
  outFile.close();
  if (!outFile) { /* error handling */
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

void sampleLabColorsAndSaveConfig(
    const cv::Mat
        &final_corrected_lab, // Input: Perspective-corrected LAB image
    const std::vector<cv::Point2f>
        &corrected_dest_points, // Input: TL, TR, BR, BL coordinates in the
                                // corrected image
    int adaptive_radius,        // Input: Pre-calculated sampling radius
    const std::string &output_config_filename) {
  // Sample points are the corners of the `corrected_dest_points`
  cv::Point2f sample_pt_TL_stone = corrected_dest_points[0];
  cv::Point2f sample_pt_TR_stone = corrected_dest_points[1];
  cv::Point2f sample_pt_BR_stone = corrected_dest_points[2];
  cv::Point2f sample_pt_BL_stone = corrected_dest_points[3];

  std::vector<cv::Point2f> board_sample_pts_corrected_final;
  board_sample_pts_corrected_final.push_back(
      (sample_pt_TL_stone + sample_pt_TR_stone) * 0.5f);
  board_sample_pts_corrected_final.push_back(
      (sample_pt_BL_stone + sample_pt_BR_stone) * 0.5f);
  board_sample_pts_corrected_final.push_back(
      (sample_pt_TL_stone + sample_pt_BL_stone) * 0.5f);
  board_sample_pts_corrected_final.push_back(
      (sample_pt_TR_stone + sample_pt_BR_stone) * 0.5f);
  board_sample_pts_corrected_final.push_back(
      (sample_pt_TL_stone + sample_pt_TR_stone + sample_pt_BL_stone +
       sample_pt_BR_stone) *
      0.25f);

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
    if (s[0] >= 0) {
      sum_lab_board += s;
      valid_board_s++;
    }
  }
  cv::Vec3f avg_lab_board_sampled =
      (valid_board_s > 0) ? (sum_lab_board / static_cast<float>(valid_board_s))
                          : cv::Vec3f(-1, -1, -1);

  // Save to config.txt: RAW corners, but LAB values from CORRECTED image.
  saveCornerConfig(output_config_filename, sample_pt_TL_stone,
                   sample_pt_TR_stone, sample_pt_BL_stone, sample_pt_BR_stone,
                   final_corrected_lab.cols, final_corrected_lab.rows,
                   lab_tl_sampled, lab_tr_sampled, lab_bl_sampled,
                   lab_br_sampled, avg_lab_board_sampled);
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
    // Allow to proceed with whatever resolution was set, or throw if strictness
    // is needed
  }
  int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  if (bDebug)
    std::cout << "Debug: Calibration - Using frame dimensions: " << frame_width
              << "x" << frame_height << std::endl;

  cv::namedWindow(WINDOW_RAW_FEED, cv::WINDOW_AUTOSIZE);
  cv::namedWindow(WINDOW_CORRECTED_PREVIEW, cv::WINDOW_AUTOSIZE);
  cv::moveWindow(WINDOW_CORRECTED_PREVIEW, frame_width + 20,
                 0); // Position windows side-by-side

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
      std::cout << "Processing 'save' command..." << std::endl;
      cv::Mat final_raw_bgr_for_snapshot =
          raw_frame.clone(); // The raw frame corresponding to the current good
                             // preview
      if (bDebug) {
        cv::Mat debug_sample_lab;
        cv::cvtColor(final_raw_bgr_for_snapshot, debug_sample_lab,
                     cv::COLOR_BGR2Lab);
        sampleLabColorsAndSaveConfig(debug_sample_lab, current_source_points,
                                     adaptive_radius, CALIB_DEBUG_CONFIG_PATH);
        std::cout << "saving debug config at: " << CALIB_DEBUG_CONFIG_PATH
                  << std::endl;
      }
      // Perspective correct this final raw frame for sampling
      cv::Mat final_corrected_bgr_for_sampling;
      cv::warpPerspective(final_raw_bgr_for_snapshot,
                          final_corrected_bgr_for_sampling, transform_matrix,
                          cv::Size(frame_width, frame_height));

      // Save snapshots
      cv::imwrite(CALIB_SNAPSHOT_RAW_PATH, final_raw_bgr_for_snapshot);
      cv::imwrite(CALIB_SNAPSHOT_PATH,
                  final_corrected_bgr_for_sampling); // This is the one whose
                                                     // colors are sampled
      if (bDebug)
        cv::imwrite(CALIB_SNAPSHOT_DEBUG_PATH, display_raw); // OSD on raw
      std::cout << "  Raw calibration snapshot saved to "
                << CALIB_SNAPSHOT_RAW_PATH << std::endl;
      std::cout << "  Corrected calibration snapshot (used for color sampling) "
                   "saved to "
                << CALIB_SNAPSHOT_PATH << std::endl;

      cv::Mat final_corrected_lab;
      cv::cvtColor(final_corrected_bgr_for_sampling, final_corrected_lab,
                   cv::COLOR_BGR2Lab);

      sampleLabColorsAndSaveConfig(final_corrected_lab, corrected_dest_points,
                                   adaptive_radius, CALIB_CONFIG_PATH);

      std::cout << "Calibration complete. Configuration saved." << std::endl;
      break;
    }
  }
  cap.release();
  cv::destroyAllWindows();
}