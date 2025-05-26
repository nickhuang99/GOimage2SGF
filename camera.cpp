#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
// #include <iostream> // Replaced by logger for most output
#include <opencv2/imgproc.hpp> // For cvtColor
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "common.h" // Includes logger.h

// Using declarations for clarity if not using 'using namespace std;' globally
// using std::string; // Prefer qualifying with std::
// using std::vector;
// using cv::Point2f;
// using cv::Mat;
// using cv::Scalar;

// Enum to represent the active corner for adjustment
enum class ActiveCorner {
  NONE,
  TOP_LEFT,
  TOP_RIGHT,
  BOTTOM_LEFT,
  BOTTOM_RIGHT
};
// ActiveCorner currentActiveCorner = ActiveCorner::TOP_LEFT; // This should be
// local to runInteractiveCalibration

// --- Define calibration output paths (declarations are extern in common.h) ---
const std::string CALIB_CONFIG_PATH = "./share/config.txt";
const std::string CALIB_SNAPSHOT_PATH =
    "./share/snapshot.jpg"; // Corrected snapshot for general use
const std::string CALIB_SNAPSHOT_RAW_PATH =
    "./share/snapshot_raw_calibration.jpg"; // Raw image taken during
                                            // calibration
const std::string CALIB_SNAPSHOT_DEBUG_PATH =
    "./share/snapshot_osd.jpg"; // Snapshot with OSD/debug info

const std::string g_default_input_image_path = CALIB_SNAPSHOT_RAW_PATH;

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
    break;
  case 'k':
    point.y += step;
    break;
  case 'j':
    point.x -= step;
    break;
  case 'l':
    point.x += step;
    break;
  }
  // Boundary checks
  point.x =
      std::max(0.0f, std::min(static_cast<float>(frame_width - 1), point.x));
  point.y =
      std::max(0.0f, std::min(static_cast<float>(frame_height - 1), point.y));
}

int processCalibrationKeyPress(int key, cv::Point2f &tl, cv::Point2f &tr,
                               cv::Point2f &bl, cv::Point2f &br,
                               int frame_width, int frame_height,
                               ActiveCorner &activeCorner) {
  const int step = 5;

  switch (key) {
  case '1':
    activeCorner = ActiveCorner::TOP_LEFT;
    LOG_DEBUG << "Active corner set to TOP_LEFT." << std::endl;
    break;
  case '2':
    activeCorner = ActiveCorner::TOP_RIGHT;
    LOG_DEBUG << "Active corner set to TOP_RIGHT." << std::endl;
    break;
  case '3':
    activeCorner = ActiveCorner::BOTTOM_LEFT;
    LOG_DEBUG << "Active corner set to BOTTOM_LEFT." << std::endl;
    break;
  case '4':
    activeCorner = ActiveCorner::BOTTOM_RIGHT;
    LOG_DEBUG << "Active corner set to BOTTOM_RIGHT." << std::endl;
    break;
  case 'i':
  case 'k':
  case 'j':
  case 'l':
    if (activeCorner == ActiveCorner::TOP_LEFT)
      movePoint(tl, key, step, frame_width, frame_height);
    else if (activeCorner == ActiveCorner::TOP_RIGHT)
      movePoint(tr, key, step, frame_width, frame_height);
    else if (activeCorner == ActiveCorner::BOTTOM_LEFT)
      movePoint(bl, key, step, frame_width, frame_height);
    else if (activeCorner == ActiveCorner::BOTTOM_RIGHT)
      movePoint(br, key, step, frame_width, frame_height);
    LOG_DEBUG << "Moved active corner. Key: " << static_cast<char>(key)
              << std::endl;
    break;
  case 's':
    LOG_INFO << "'s' key pressed, signaling save for calibration." << std::endl;
    return 's';
  case 27:
    LOG_INFO << "ESC key pressed, signaling exit from calibration."
             << std::endl;
    return 27;
  default:
    return key; // Pass unhandled keys
  }
  return 0;
}

// This function is older and less used, processCalibrationKeyPress is primary
// for interactive.
int handleCalibrationInput(int key, cv::Point2f &topLeft, cv::Point2f &topRight,
                           cv::Point2f &bottomLeft, cv::Point2f &bottomRight,
                           int frame_width, int frame_height) {
  LOG_DEBUG << "Legacy handleCalibrationInput called with key: " << key
            << std::endl;
  const int step = 5;
  int return_signal = 0;

  switch (key) {
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
  case 'k':
    bottomLeft.y -= step;
    bottomRight.y -= step;
    break;
  case 'j':
    bottomLeft.y += step;
    bottomRight.y += step;
    break;
  case 'l':
    bottomLeft.x -= step;
    bottomRight.x += step;
    break;
  case 'm':
    bottomLeft.x += step;
    bottomRight.x -= step;
    break;
  case 's':
    return_signal = 's';
    break;
  case 27:
    return_signal = 27;
    break;
  default:
    if (key != -1)
      return_signal = key;
    break;
  }
  // Boundary & anti-crossing checks as before
  topLeft.x =
      std::max(0.0f, std::min(static_cast<float>(frame_width - 1), topLeft.x));
  topLeft.y =
      std::max(0.0f, std::min(static_cast<float>(frame_height - 1), topLeft.y));
  topRight.x =
      std::max(0.0f, std::min(static_cast<float>(frame_width - 1), topRight.x));
  topRight.y = std::max(
      0.0f, std::min(static_cast<float>(frame_height - 1), topRight.y));
  bottomLeft.x = std::max(
      0.0f, std::min(static_cast<float>(frame_width - 1), bottomLeft.x));
  bottomLeft.y = std::max(
      0.0f, std::min(static_cast<float>(frame_height - 1), bottomLeft.y));
  bottomRight.x = std::max(
      0.0f, std::min(static_cast<float>(frame_width - 1), bottomRight.x));
  bottomRight.y = std::max(
      0.0f, std::min(static_cast<float>(frame_height - 1), bottomRight.y));
  if (topRight.x < topLeft.x + 10.0f) {
    topRight.x = topLeft.x + 10.0f;
  }
  if (bottomRight.x < bottomLeft.x + 10.0f) {
    bottomRight.x = bottomLeft.x + 10.0f;
  }
  return return_signal;
}

void drawCalibrationOSD(cv::Mat &display_frame, const cv::Point2f &tl,
                        const cv::Point2f &tr, const cv::Point2f &bl,
                        const cv::Point2f &br, ActiveCorner activeCorner) {
  // Drawing function, no console output.
  int circle_radius = 5;
  int active_circle_radius = 8;
  cv::Scalar inactive_color(150, 150, 150);
  cv::circle(display_frame, tl,
             (activeCorner == ActiveCorner::TOP_LEFT ? active_circle_radius
                                                     : circle_radius),
             (activeCorner == ActiveCorner::TOP_LEFT ? cv::Scalar(0, 0, 255)
                                                     : inactive_color),
             -1);
  cv::circle(display_frame, tr,
             (activeCorner == ActiveCorner::TOP_RIGHT ? active_circle_radius
                                                      : circle_radius),
             (activeCorner == ActiveCorner::TOP_RIGHT ? cv::Scalar(255, 0, 0)
                                                      : inactive_color),
             -1);
  cv::circle(display_frame, bl,
             (activeCorner == ActiveCorner::BOTTOM_LEFT ? active_circle_radius
                                                        : circle_radius),
             (activeCorner == ActiveCorner::BOTTOM_LEFT
                  ? cv::Scalar(255, 0, 255)
                  : inactive_color),
             -1);
  cv::circle(display_frame, br,
             (activeCorner == ActiveCorner::BOTTOM_RIGHT ? active_circle_radius
                                                         : circle_radius),
             (activeCorner == ActiveCorner::BOTTOM_RIGHT
                  ? cv::Scalar(0, 255, 255)
                  : inactive_color),
             -1);
  cv::Scalar line_color(0, 255, 0);
  int line_thickness = 1;
  cv::line(display_frame, tl, tr, line_color, line_thickness);
  cv::line(display_frame, tr, br, line_color, line_thickness);
  cv::line(display_frame, br, bl, line_color, line_thickness);
  cv::line(display_frame, bl, tl, line_color, line_thickness);
  int font_face = cv::FONT_HERSHEY_SIMPLEX;
  cv::Scalar help_text_color(0, 0, 255);
  cv::Scalar coord_text_color(255, 200, 0);
  int text_thickness = 1;
  double help_font_scale = 0.45;
  cv::Point help_text_origin(10, 20);
  cv::Point help_text_origin_line2(10, 35);
  cv::Point help_text_origin_line3(10, 50);
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

static void
drawCorrectedPreviewOSD(cv::Mat &corrected_frame, int preview_width,
                        int preview_height,
                        const std::vector<cv::Point2f> &corrected_dest_points,
                        int adaptive_radius) {
  // Drawing function, no console output.
  cv::Point2f sample_pt_TL_stone = corrected_dest_points[0];
  cv::Point2f sample_pt_TR_stone = corrected_dest_points[1];
  cv::Point2f sample_pt_BR_stone = corrected_dest_points[2];
  cv::Point2f sample_pt_BL_stone = corrected_dest_points[3];
  float board_width = sample_pt_TR_stone.x - sample_pt_TL_stone.x;
  float board_height = sample_pt_BL_stone.y - sample_pt_TL_stone.y;
  cv::Scalar grid_color(100, 100, 100);
  float x_step = board_width / 18.0f;
  float y_step = board_height / 18.0f;
  float x_start = sample_pt_TL_stone.x;
  float y_start = sample_pt_TL_stone.y;
  float x_end = sample_pt_TR_stone.x;
  float y_end = sample_pt_BL_stone.y;

  for (int i = 0; i < 19; ++i) {
    cv::line(corrected_frame, cv::Point2f(x_start + i * x_step, y_start),
             cv::Point2f(x_start + i * x_step, y_end), grid_color, 1);
    cv::line(corrected_frame, cv::Point2f(x_start, y_start + i * y_step),
             cv::Point2f(x_end, y_start + i * y_step), grid_color, 1);
  }
  cv::Scalar white_marker_outline(255, 0, 0);
  cv::Scalar black_marker_outline(0, 0, 255);
  cv::circle(corrected_frame, sample_pt_TL_stone, adaptive_radius,
             black_marker_outline, 1);
  cv::circle(corrected_frame, sample_pt_TR_stone, adaptive_radius,
             white_marker_outline, 1);
  cv::circle(corrected_frame, sample_pt_BR_stone, adaptive_radius,
             white_marker_outline, 1);
  cv::circle(corrected_frame, sample_pt_BL_stone, adaptive_radius,
             black_marker_outline, 1);
  std::string help_text =
      "Align physical stones to CORNER markers. Grid should be straight.";
  cv::putText(corrected_frame, help_text, cv::Point(10, preview_height - 10),
              cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1,
              cv::LINE_AA);
}

bool saveCornerConfig(
    const std::string &filename, const std::string &device_path_for_config,
    int frame_width, int frame_height, const cv::Point2f &tl_raw,
    const cv::Point2f &tr_raw, const cv::Point2f &bl_raw,
    const cv::Point2f &br_raw, const cv::Vec3f &lab_tl_sampled,
    const cv::Vec3f &lab_tr_sampled, const cv::Vec3f &lab_bl_sampled,
    const cv::Vec3f &lab_br_sampled, const cv::Vec3f &avg_lab_board_sampled,
    bool enhanced_data_available,
    const std::vector<cv::Vec3f> *lab_corners_sampled_raw_enhanced,
    float detected_avg_stone_radius_raw) {

  std::ofstream outFile(filename);
  if (!outFile.is_open()) {
    LOG_ERROR << "Could not open config file for writing: " << filename
              << std::endl;
    return false;
  }
  LOG_INFO << "Saving configuration to " << filename << std::endl;
  outFile << "# Go Board Calibration Configuration" << std::endl;
  outFile << "# Generated by GEM" << std::endl;
  outFile << "\n# Device and Resolution at Calibration Time" << std::endl;
  outFile << "DevicePath=" << device_path_for_config << std::endl;
  outFile << "ImageWidth=" << frame_width << std::endl;
  outFile << "ImageHeight=" << frame_height << std::endl;
  outFile << std::fixed << std::setprecision(1);
  outFile << "\n# Corner Pixel Coordinates (Raw Image - TL, TR, BL, BR)"
          << std::endl;
  outFile << "TL_X_PX=" << tl_raw.x << std::endl;
  outFile << "TL_Y_PX=" << tl_raw.y << std::endl;
  outFile << "TR_X_PX=" << tr_raw.x << std::endl;
  outFile << "TR_Y_PX=" << tr_raw.y << std::endl;
  outFile << "BL_X_PX=" << bl_raw.x
          << std::endl; // Note: BL before BR in this specific save order from
                        // original code
  outFile << "BL_Y_PX=" << bl_raw.y << std::endl;
  outFile << "BR_X_PX=" << br_raw.x << std::endl;
  outFile << "BR_Y_PX=" << br_raw.y << std::endl;
  outFile << "\n# Sampled Lab Colors from Corrected Image (L:0-255, A:0-255, "
             "B:0-255)"
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
  outFile << "BOARD_L_AVG=" << avg_lab_board_sampled[0] << std::endl;
  outFile << "BOARD_A_AVG=" << avg_lab_board_sampled[1] << std::endl;
  outFile << "BOARD_B_AVG=" << avg_lab_board_sampled[2] << std::endl;

  if (frame_width > 0 && frame_height > 0) {
    outFile << "\n# Corner Percentage Coordinates (Raw Image %)" << std::endl;
    outFile << "TL_X_PC="
            << (tl_raw.x / static_cast<float>(frame_width) * 100.0f)
            << std::endl;
    outFile << "TL_Y_PC="
            << (tl_raw.y / static_cast<float>(frame_height) * 100.0f)
            << std::endl;
    outFile << "TR_X_PC="
            << (tr_raw.x / static_cast<float>(frame_width) * 100.0f)
            << std::endl;
    outFile << "TR_Y_PC="
            << (tr_raw.y / static_cast<float>(frame_height) * 100.0f)
            << std::endl;
    outFile << "BL_X_PC="
            << (bl_raw.x / static_cast<float>(frame_width) * 100.0f)
            << std::endl;
    outFile << "BL_Y_PC="
            << (bl_raw.y / static_cast<float>(frame_height) * 100.0f)
            << std::endl;
    outFile << "BR_X_PC="
            << (br_raw.x / static_cast<float>(frame_width) * 100.0f)
            << std::endl;
    outFile << "BR_Y_PC="
            << (br_raw.y / static_cast<float>(frame_height) * 100.0f)
            << std::endl;
  }
  if (enhanced_data_available && lab_corners_sampled_raw_enhanced &&
      lab_corners_sampled_raw_enhanced->size() == 4) {
    outFile
        << "\n# Enhanced Stone Detection Data (from Raw Image, if successful)"
        << std::endl;
    outFile << "DETECTED_AVG_STONE_RADIUS_PX=" << detected_avg_stone_radius_raw
            << std::endl;
    // Assuming order in lab_corners_sampled_raw_enhanced is TL, TR, BR, BL (as
    // per detectFourCornersGoBoard output)
    outFile << "DETECTED_TL_L=" << (*lab_corners_sampled_raw_enhanced)[0][0]
            << std::endl;
    outFile << "DETECTED_TL_A=" << (*lab_corners_sampled_raw_enhanced)[0][1]
            << std::endl;
    outFile << "DETECTED_TL_B=" << (*lab_corners_sampled_raw_enhanced)[0][2]
            << std::endl;
    outFile << "DETECTED_TR_L=" << (*lab_corners_sampled_raw_enhanced)[1][0]
            << std::endl;
    outFile << "DETECTED_TR_A=" << (*lab_corners_sampled_raw_enhanced)[1][1]
            << std::endl;
    outFile << "DETECTED_TR_B=" << (*lab_corners_sampled_raw_enhanced)[1][2]
            << std::endl;
    outFile << "DETECTED_BR_L=" << (*lab_corners_sampled_raw_enhanced)[2][0]
            << std::endl;
    outFile << "DETECTED_BR_A=" << (*lab_corners_sampled_raw_enhanced)[2][1]
            << std::endl;
    outFile << "DETECTED_BR_B=" << (*lab_corners_sampled_raw_enhanced)[2][2]
            << std::endl;
    outFile << "DETECTED_BL_L=" << (*lab_corners_sampled_raw_enhanced)[3][0]
            << std::endl;
    outFile << "DETECTED_BL_A=" << (*lab_corners_sampled_raw_enhanced)[3][1]
            << std::endl;
    outFile << "DETECTED_BL_B=" << (*lab_corners_sampled_raw_enhanced)[3][2]
            << std::endl;
  } else if (enhanced_data_available) {
    outFile << "\n# Enhanced Stone Detection was attempted but data was "
               "incomplete or not provided to saveCornerConfig."
            << std::endl;
  }

  outFile.close();
  if (!outFile) {
    LOG_ERROR << "Failed to properly close config file after writing: "
              << filename << std::endl;
    return false;
  }
  CONSOLE_OUT << "Configuration saved to " << filename << std::endl;
  LOG_INFO << "Configuration successfully saved to " << filename << std::endl;
  return true;
}

bool trySetCameraResolution(cv::VideoCapture &cap, int desired_width,
                            int desired_height, bool attempt_fallback_format) {
  if (!cap.isOpened()) {
    LOG_WARN << "trySetCameraResolution - VideoCapture not open." << std::endl;
    return false;
  }
  LOG_DEBUG << "trySetCameraResolution - Attempting resolution "
            << desired_width << "x" << desired_height << std::endl;
  bool success = false;
  // ... (MJPEG attempt as before) ...
  cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
  cap.set(cv::CAP_PROP_FRAME_WIDTH, static_cast<double>(desired_width));
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<double>(desired_height));
  int actual_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int actual_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  LOG_DEBUG << "After 1st attempt (MJPEG): Actual Size " << actual_width << "x"
            << actual_height << std::endl;
  if (actual_width == desired_width && actual_height == desired_height)
    success = true;

  if (!success && attempt_fallback_format) {
    LOG_DEBUG << "Initial attempt failed. Trying fallback YUYV for resolution."
              << std::endl;
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, static_cast<double>(desired_width));
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<double>(desired_height));
    actual_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    actual_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    LOG_DEBUG << "After 2nd attempt (YUYV): Actual Size " << actual_width << "x"
              << actual_height << std::endl;
    if (actual_width == desired_width && actual_height == desired_height)
      success = true;
  }

  if (success) {
    LOG_INFO << "Successfully set camera resolution to " << actual_width << "x"
             << actual_height << std::endl;
  } else {
    LOG_WARN << "Failed to set desired resolution " << desired_width << "x"
             << desired_height << ". Camera using: " << actual_width << "x"
             << actual_height << std::endl;
  }
  return success;
}

void sampleDataForConfig(const cv::Mat &final_corrected_lab,
                         const std::vector<cv::Point2f> &corrected_dest_points,
                         int adaptive_radius,
                         std::vector<cv::Vec3f> &output_lab_values) {
  LOG_DEBUG << "Sampling Lab data for config. Corrected points count: "
            << corrected_dest_points.size()
            << ", Adaptive radius: " << adaptive_radius << std::endl;
  if (corrected_dest_points.size() != 4) {
    std::string error_msg = "sampleDataForConfig expects exactly 4 corrected "
                            "destination points (TL, TR, BR, BL). Got: " +
                            std::to_string(corrected_dest_points.size());
    LOG_ERROR << error_msg << std::endl;
    THROWGEMERROR(error_msg);
  }

  output_lab_values.clear();

  // Order of sampling for output_lab_values: TL, TR, BL, BR, BoardAvg
  cv::Point2f sample_pt_TL_stone = corrected_dest_points[0]; // TL
  cv::Point2f sample_pt_TR_stone = corrected_dest_points[1]; // TR
  // Note: corrected_dest_points from getBoardCornersCorrected is TL, TR, BR, BL
  // So, for our BL sample, we use index 3, and for BR sample, we use index 2.
  cv::Point2f sample_pt_BL_stone = corrected_dest_points[3]; // BL
  cv::Point2f sample_pt_BR_stone = corrected_dest_points[2]; // BR

  output_lab_values.push_back(
      getAverageLab(final_corrected_lab, sample_pt_TL_stone, adaptive_radius));
  output_lab_values.push_back(
      getAverageLab(final_corrected_lab, sample_pt_TR_stone, adaptive_radius));
  output_lab_values.push_back(
      getAverageLab(final_corrected_lab, sample_pt_BL_stone, adaptive_radius));
  output_lab_values.push_back(
      getAverageLab(final_corrected_lab, sample_pt_BR_stone, adaptive_radius));

  std::vector<cv::Point2f> board_empty_sample_pts;
  float board_pixel_width =
      corrected_dest_points[1].x - corrected_dest_points[0].x;
  float board_pixel_height =
      corrected_dest_points[3].y - corrected_dest_points[0].y;

  if (board_pixel_width < 18.0f ||
      board_pixel_height < 18.0f) { // Ensure positive and sensible dimensions
    std::string error_msg = "Corrected board dimensions (" +
                            Num2Str(board_pixel_width).str() + "x" +
                            Num2Str(board_pixel_height).str() +
                            ") too small for full empty space sampling.";
    LOG_ERROR << error_msg << std::endl;
    THROWGEMERROR(error_msg);
  }
  float x_step = board_pixel_width / 18.0f;
  float y_step = board_pixel_height / 18.0f;
  cv::Point2f c_tl = corrected_dest_points[0];

  for (int r = 0; r < 19; ++r) {
    for (int c = 0; c < 19; ++c) {
      bool is_stone_corner = (r == 0 && c == 0) || (r == 0 && c == 18) ||
                             (r == 18 && c == 0) || (r == 18 && c == 18);
      if (!is_stone_corner) {
        board_empty_sample_pts.push_back(
            cv::Point2f(c_tl.x + c * x_step, c_tl.y + r * y_step));
      }
    }
  }

  cv::Vec3f sum_lab_board(0, 0, 0);
  int valid_board_samples = 0;
  if (!board_empty_sample_pts.empty()) {
    for (const auto &pt : board_empty_sample_pts) {
      cv::Vec3f s = getAverageLab(final_corrected_lab, pt, adaptive_radius);
      if (s[0] >= 0) {
        sum_lab_board += s;
        valid_board_samples++;
      }
    }
  }
  cv::Vec3f avg_lab_board_sampled =
      (valid_board_samples > 0)
          ? (sum_lab_board / static_cast<float>(valid_board_samples))
          : cv::Vec3f(-1, -1, -1);
  output_lab_values.push_back(avg_lab_board_sampled);

  LOG_DEBUG << "Sampled Lab data: TL=" << output_lab_values[0]
            << ", TR=" << output_lab_values[1]
            << ", BL=" << output_lab_values[2]
            << ", BR=" << output_lab_values[3]
            << ", BoardAvg=" << output_lab_values[4] << std::endl;
}

static bool
verifyCalibrationAfterSave(const cv::Mat &raw_image_for_verification) {
  LOG_INFO
      << "Verifying calibration settings (strict check) using current config..."
      << std::endl;

  if (raw_image_for_verification.empty()) {
    LOG_ERROR << "Verification Error: Input image for verification is empty."
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
      LOG_ERROR << "Verification Error: processGoBoard did not return a valid "
                   "19x19 board state."
                << std::endl;
      return false;
    }

    for (int r = 0; r < 19; ++r) {
      for (int c = 0; c < 19; ++c) {
        int color = board_state_matrix.at<uchar>(r, c);
        if ((r == 0 && c == 0) ||
            (r == 18 && c == 0)) { // TL, BL should be Black
          if (color != BLACK) {
            LOG_ERROR << "Verification Failed: Expected BLACK at corner (" << r
                      << "," << c << "), found " << color << std::endl;
            if (bDebug) {
              cv::imshow("Calibration Verification Failed (Stone)",
                         board_with_stones_display);
              cv::waitKey(0);
            }
            return false;
          }
        } else if ((r == 0 && c == 18) ||
                   (r == 18 && c == 18)) { // TR, BR should be White
          if (color != WHITE) {
            LOG_ERROR << "Verification Failed: Expected WHITE at corner (" << r
                      << "," << c << "), found " << color << std::endl;
            if (bDebug) {
              cv::imshow("Calibration Verification Failed (Stone)",
                         board_with_stones_display);
              cv::waitKey(0);
            }
            return false;
          }
        } else { // Other points should be empty
          if (color != EMPTY) {
            LOG_ERROR << "Verification Failed: Expected EMPTY at non-corner ("
                      << r << "," << c << "), found " << color << std::endl;
            if (bDebug) {
              cv::imshow("Calibration Verification Failed (Empty)",
                         board_with_stones_display);
              cv::waitKey(0);
            }
            return false;
          }
        }
      }
    }
  } catch (const GEMError &ge) {
    LOG_ERROR << "Verification GEMError during processGoBoard: " << ge.what()
              << std::endl;
    return false;
  } catch (const std::exception &e) {
    LOG_ERROR << "Verification std::exception during processGoBoard: "
              << e.what() << std::endl;
    return false;
  }
  LOG_INFO << "Calibration verification PASSED." << std::endl;
  return true;
}

bool processAndSaveCalibration(
    const cv::Mat &final_raw_bgr_for_snapshot,
    const std::vector<cv::Point2f>
        &current_source_points, // Order TL, TR, BR, BL
    bool enhanced_detection_was_successful,
    const std::vector<cv::Vec3f>
        *enhanced_lab_colors, // Order TL, TR, BR, BL from detectFourCorners
    float enhanced_avg_radius_px) {

  int frame_width = final_raw_bgr_for_snapshot.cols;
  int frame_height = final_raw_bgr_for_snapshot.rows;

  LOG_INFO
      << "processAndSaveCalibration: Starting. Enhanced detection success: "
      << (enhanced_detection_was_successful ? "Yes" : "No") << std::endl;
  if (current_source_points.size() != 4) {
    LOG_ERROR << "processAndSaveCalibration requires 4 source points. Got "
              << current_source_points.size() << std::endl;
    THROWGEMERROR("processAndSaveCalibration requires 4 source points.");
  }

  if (enhanced_detection_was_successful && enhanced_lab_colors &&
      enhanced_lab_colors->size() == 4) {
    LOG_INFO << "    Enhanced detection data IS available and will be used for "
                "some config values."
             << std::endl;
    LOG_INFO << "    Enhanced Avg Radius (raw): " << enhanced_avg_radius_px
             << std::endl;
    LOG_DEBUG << "    Enhanced TL Lab (raw): " << (*enhanced_lab_colors)[0]
              << std::endl;
    LOG_DEBUG << "    Enhanced TR Lab (raw): " << (*enhanced_lab_colors)[1]
              << std::endl;
    LOG_DEBUG << "    Enhanced BR Lab (raw): " << (*enhanced_lab_colors)[2]
              << std::endl;
    LOG_DEBUG << "    Enhanced BL Lab (raw): " << (*enhanced_lab_colors)[3]
              << std::endl;
  } else {
    LOG_INFO << "    Enhanced detection data IS NOT available or not used. "
                "Using standard sampling from corrected view."
             << std::endl;
  }

  std::vector<cv::Point2f> corrected_dest_points =
      getBoardCornersCorrected(frame_width, frame_height); // TL, TR, BR, BL
  cv::Mat transform_matrix =
      cv::getPerspectiveTransform(current_source_points, corrected_dest_points);
  cv::Mat final_corrected_bgr_for_sampling;
  cv::warpPerspective(final_raw_bgr_for_snapshot,
                      final_corrected_bgr_for_sampling, transform_matrix,
                      cv::Size(frame_width, frame_height));

  if (!cv::imwrite(CALIB_SNAPSHOT_RAW_PATH, final_raw_bgr_for_snapshot))
    LOG_ERROR << "Failed to write raw calibration snapshot to "
              << CALIB_SNAPSHOT_RAW_PATH << std::endl;
  else
    LOG_INFO << "Raw calibration snapshot saved to " << CALIB_SNAPSHOT_RAW_PATH
             << std::endl;

  if (!cv::imwrite(CALIB_SNAPSHOT_PATH, final_corrected_bgr_for_sampling)) {
    LOG_ERROR << "Failed to write corrected calibration snapshot to "
              << CALIB_SNAPSHOT_PATH << std::endl;
  } else {
    LOG_INFO << "Corrected calibration snapshot (for standard color sampling) "
                "saved to "
             << CALIB_SNAPSHOT_PATH << std::endl;
  }

  cv::Mat final_corrected_lab;
  cv::cvtColor(final_corrected_bgr_for_sampling, final_corrected_lab,
               cv::COLOR_BGR2Lab);

  int correct_board_width_px = static_cast<int>(
      corrected_dest_points[1].x - corrected_dest_points[0].x); // TR.x - TL.x
  int correct_board_height_px = static_cast<int>(
      corrected_dest_points[3].y - corrected_dest_points[0].y); // BL.y - TL.y
  int adaptive_radius_corrected = calculateAdaptiveSampleRadius(
      correct_board_width_px, correct_board_height_px);

  std::vector<cv::Vec3f>
      standard_sample_data_vector; // Expects 5 Vec3f: TL, TR, BL, BR, BoardAvg
  sampleDataForConfig(final_corrected_lab, corrected_dest_points,
                      adaptive_radius_corrected, standard_sample_data_vector);

  if (standard_sample_data_vector.size() < 5) {
    LOG_ERROR << "Standard sampling (sampleDataForConfig) failed to produce "
                 "enough data points."
              << std::endl;
    THROWGEMERROR("Standard sampling failed to produce sufficient data.");
  }

  // saveCornerConfig expects raw points in TL, TR, BL, BR order
  // current_source_points is TL, TR, BR, BL
  bool save_success = saveCornerConfig(
      CALIB_CONFIG_PATH, g_device_path, frame_width, frame_height,
      current_source_points[0], current_source_points[1], // TL_raw, TR_raw
      current_source_points[3], current_source_points[2], // BL_raw, BR_raw
      standard_sample_data_vector[0],
      standard_sample_data_vector[1], // Sampled TL, TR
      standard_sample_data_vector[2],
      standard_sample_data_vector[3], // Sampled BL, BR
      standard_sample_data_vector[4], // BoardAvg
      enhanced_detection_was_successful, enhanced_lab_colors,
      enhanced_avg_radius_px);

  if (!save_success) {
    // saveCornerConfig already logs error
    return false;
  }
  // Message already logged by saveCornerConfig

  bool verification_passed =
      verifyCalibrationAfterSave(final_raw_bgr_for_snapshot);
  // Messages logged by verifyCalibrationAfterSave
  return verification_passed;
}

void runInteractiveCalibration(int camera_index) {
  cv::VideoCapture cap;
  LOG_INFO << "Starting INTERACTIVE calibration for camera index: "
           << camera_index << " (derived from " << g_device_path << ")"
           << std::endl;

  cap.open(camera_index, cv::CAP_V4L2);
  if (!cap.isOpened()) {
    LOG_WARN << "Opening camera with CAP_V4L2 failed for index " << camera_index
             << ", trying default backend." << std::endl;
    cap.open(camera_index);
    if (!cap.isOpened()) {
      LOG_ERROR << "OpenCV failed to open camera index " << camera_index
                << std::endl;
      THROWGEMERROR("OpenCV failed to open camera index " +
                    Num2Str(camera_index).str());
    }
  }
  LOG_INFO << "Successfully opened Camera Index: " << camera_index << "."
           << std::endl;

  if (!trySetCameraResolution(cap, g_capture_width, g_capture_height, true)) {
    // trySetCameraResolution logs warnings on failure
    LOG_WARN << "Proceeding with current camera resolution after failed "
                "attempt to set "
             << g_capture_width << "x" << g_capture_height << "." << std::endl;
  }
  int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  LOG_INFO << "Calibration - Using frame dimensions: " << frame_width << "x"
           << frame_height << std::endl;

  cv::namedWindow(WINDOW_RAW_FEED, cv::WINDOW_AUTOSIZE);
  cv::namedWindow(WINDOW_CORRECTED_PREVIEW, cv::WINDOW_AUTOSIZE);
  cv::moveWindow(WINDOW_RAW_FEED, 0, 0);
  cv::moveWindow(WINDOW_CORRECTED_PREVIEW, frame_width + 10, 0);

  cv::Mat raw_frame, display_raw, display_corrected_bgr;
  cv::Point2f topLeft_raw(frame_width * 0.15f, frame_height * 0.15f);
  cv::Point2f topRight_raw(frame_width * 0.85f, frame_height * 0.15f);
  cv::Point2f bottomLeft_raw(frame_width * 0.15f, frame_height * 0.85f);
  cv::Point2f bottomRight_raw(frame_width * 0.85f, frame_height * 0.85f);
  ActiveCorner currentActiveCorner = ActiveCorner::TOP_LEFT;

  std::vector<cv::Point2f> corrected_dest_points =
      getBoardCornersCorrected(frame_width, frame_height);

  CONSOLE_OUT << "\n--- Interactive Calibration (Live Corrected Preview) ---"
              << std::endl;
  CONSOLE_OUT << "INSTRUCTIONS:" << std::endl;
  CONSOLE_OUT << "1. Place BLACK stones near physical Top-Left & Bottom-Left "
                 "board corners."
              << std::endl;
  CONSOLE_OUT << "2. Place WHITE stones near physical Top-Right & Bottom-Right "
                 "board corners."
              << std::endl;
  CONSOLE_OUT << "3. Ensure board mid-points & center are EMPTY for board "
                 "color sampling."
              << std::endl;
  CONSOLE_OUT << "4. Adjust markers in '" << WINDOW_RAW_FEED
              << "' (using 1-4, ijkl keys)" << std::endl;
  CONSOLE_OUT << "   until physical stones appear at CORNER MARKERS in '"
              << WINDOW_CORRECTED_PREVIEW << "'," << std::endl;
  CONSOLE_OUT << "   and the grid there looks straight and uniform."
              << std::endl;
  CONSOLE_OUT << "ACTIONS: 's' to save, 'esc' to exit." << std::endl;
  CONSOLE_OUT << "------------------------------------" << std::endl;

  float corrected_board_width_px =
      corrected_dest_points[1].x - corrected_dest_points[0].x;
  float corrected_board_height_px =
      corrected_dest_points[3].y - corrected_dest_points[0].y;
  int adaptive_radius = calculateAdaptiveSampleRadius(
      corrected_board_width_px, corrected_board_height_px);
  LOG_DEBUG
      << "Sampling Lab colors from CORRECTED image using adaptive radius: "
      << adaptive_radius << std::endl;

  while (true) {
    if (!cap.read(raw_frame) || raw_frame.empty()) {
      LOG_ERROR
          << "Could not read frame from camera during interactive calibration."
          << std::endl;
      cv::waitKey(100);
      continue;
    }
    display_raw = raw_frame.clone();
    drawCalibrationOSD(display_raw, topLeft_raw, topRight_raw, bottomLeft_raw,
                       bottomRight_raw, currentActiveCorner);
    cv::imshow(WINDOW_RAW_FEED, display_raw);
    std::vector<cv::Point2f> current_source_points = {
        topLeft_raw, topRight_raw, bottomRight_raw, bottomLeft_raw};
    cv::Mat transform_matrix = cv::getPerspectiveTransform(
        current_source_points, corrected_dest_points);
    cv::warpPerspective(raw_frame, display_corrected_bgr, transform_matrix,
                        cv::Size(frame_width, frame_height));
    drawCorrectedPreviewOSD(display_corrected_bgr, frame_width, frame_height,
                            corrected_dest_points, adaptive_radius);
    cv::imshow(WINDOW_CORRECTED_PREVIEW, display_corrected_bgr);

    int key = cv::waitKey(30);
    int key_action_type = processCalibrationKeyPress(
        key, topLeft_raw, topRight_raw, bottomLeft_raw, bottomRight_raw,
        frame_width, frame_height, currentActiveCorner);

    if (key_action_type == 27) {
      LOG_INFO << "Interactive calibration cancelled by user (ESC pressed)."
               << std::endl;
      break;
    } else if (key_action_type == 's') {
      LOG_INFO << "Processing 'save' command for interactive calibration..."
               << std::endl;
      cv::Mat final_raw_bgr_for_snapshot = raw_frame.clone();

      // Enhanced detection is not triggered in this simple interactive path by
      // default. To enable it, detectFourCornersGoBoard would need to be called
      // here on final_raw_bgr_for_snapshot. For now, passing false and nullptr
      // for enhanced data.
      if (processAndSaveCalibration(final_raw_bgr_for_snapshot,
                                    current_source_points, false, nullptr,
                                    -1.0f)) {
        LOG_INFO << "Interactive calibration saved and verified successfully."
                 << std::endl;
        CONSOLE_OUT << "Calibration saved and verified successfully!"
                    << std::endl; // User feedback
        break;
      } else {
        LOG_WARN << "Interactive calibration saved BUT VERIFICATION FAILED!"
                 << std::endl;
        CONSOLE_OUT << "WARNING: Calibration saved, but verification FAILED! "
                       "Detected corner stones might not match expected."
                    << std::endl;
        CONSOLE_OUT << "ADVICE: Check stone placement (Black TL&BL, White "
                       "TR&BR), lighting, and corner marking accuracy."
                    << std::endl;
        CONSOLE_OUT << "        You can continue adjusting and save again, or "
                       "ESC to exit and retry later."
                    << std::endl;
      }
    }
  }
  cap.release();
  cv::destroyAllWindows();
  LOG_INFO << "Interactive calibration finished." << std::endl;
}

void runCaptureCalibration() {
  LOG_INFO << "Starting AUTOMATED Capture Calibration from snapshot: "
           << CALIB_SNAPSHOT_PATH << std::endl;
  cv::Mat raw_frame = cv::imread(CALIB_SNAPSHOT_PATH);
  if (raw_frame.empty()) {
    LOG_ERROR << "Failed to load snapshot image for calibration: "
              << CALIB_SNAPSHOT_PATH << std::endl;
    THROWGEMERROR("Failed to load snapshot image for calibration: " +
                  CALIB_SNAPSHOT_PATH);
  }
  LOG_INFO << "Loaded image: " << CALIB_SNAPSHOT_PATH << " (" << raw_frame.cols
           << "x" << raw_frame.rows << ")" << std::endl;

  if (bDebug) {
    cv::imshow("Calibration Input Snapshot", raw_frame);
    LOG_DEBUG << "Displaying input snapshot. Press any key." << std::endl;
    cv::waitKey(0);
  }

  LOG_INFO << "Attempting to auto-detect corners and save calibration..."
           << std::endl;
  std::vector<cv::Point2f> detected_corners_raw;
  std::vector<float> detected_radii_raw;

  // detectFourCornersGoBoard is expected to populate detected_corners_raw with
  // points in the raw_frame's coordinate system
  if (detectFourCornersGoBoard(raw_frame, detected_corners_raw,
                               detected_radii_raw)) {
    LOG_INFO << "detectFourCornersGoBoard successful. Detected raw corners."
             << std::endl;
    if (bDebug) { // Show detected corners if debug is on
      cv::Mat raw_frame_detected_dbg = raw_frame.clone();
      for (size_t i = 0; i < detected_corners_raw.size(); ++i) {
        if (i < detected_radii_raw.size() &&
            detected_radii_raw[i] > 0) { // Check for valid radius
          cv::circle(raw_frame_detected_dbg, detected_corners_raw[i],
                     std::max(5.0f, detected_radii_raw[i]),
                     cv::Scalar(0, 255, 0), 2);
        } else {
          cv::circle(raw_frame_detected_dbg, detected_corners_raw[i], 5,
                     cv::Scalar(0, 0, 255), 2); // Fallback if radius is bad
        }
      }
      cv::imshow("Capture Calibration - Auto Detected Corners on Raw",
                 raw_frame_detected_dbg);
      LOG_DEBUG
          << "Displaying auto-detected corners on raw snapshot. Press any key."
          << std::endl;
      cv::waitKey(0);
    }

    // For this automated path, enhanced detection data is typically not
    // pre-calculated unless detectFourCornersGoBoard provides it. Assuming it
    // does not for now, passing false and nullptr.
    if (processAndSaveCalibration(raw_frame, detected_corners_raw, false,
                                  nullptr, -1.0f)) {
      LOG_INFO << "Auto-calibration from snapshot SUCCEEDED and was VERIFIED!"
               << std::endl;
      CONSOLE_OUT
          << "Auto-calibration from snapshot SUCCEEDED and was VERIFIED!"
          << std::endl;
    } else {
      LOG_ERROR << "Auto-calibration from snapshot VERIFICATION FAILED! "
                   "Detected corner stones might not match expected."
                << std::endl;
      CONSOLE_ERR << "ERROR: Auto-calibration from snapshot VERIFICATION "
                     "FAILED! Detected corner stones might not match expected."
                  << std::endl;
      CONSOLE_ERR << "ADVICE: Please check the snapshot image ("
                  << CALIB_SNAPSHOT_PATH << ")," << std::endl;
      CONSOLE_ERR
          << "        ensure it has Black stones at TL & BL, White at TR & BR,"
          << std::endl;
      CONSOLE_ERR << "        and good lighting conditions." << std::endl;
      CONSOLE_ERR << "        You might need to re-run interactive calibration "
                     "(-B) to get a good snapshot."
                  << std::endl;
    }
  } else {
    LOG_ERROR << "detectFourCornersGoBoard FAILED on " << CALIB_SNAPSHOT_PATH
              << "." << std::endl;
    CONSOLE_ERR << "ERROR: detectFourCornersGoBoard FAILED on "
                << CALIB_SNAPSHOT_PATH << "." << std::endl;
    CONSOLE_ERR
        << "Please ensure the image contains a clear view of the board with:"
        << std::endl;
    CONSOLE_ERR << "  - Two BLACK stones at Top-Left & Bottom-Left corners."
                << std::endl;
    CONSOLE_ERR << "  - Two WHITE stones at Top-Right & Bottom-Right corners."
                << std::endl;
    CONSOLE_ERR << "Consider using interactive calibration (-B) to set up "
                   "manually and save a good snapshot."
                << std::endl;
  }

  if (bDebug) {
    // Any debug windows from helpers might still be open; waitKey(0) could be
    // here for review. However, individual functions should manage their debug
    // displays.
  }
  cv::destroyAllWindows();
  LOG_INFO << "Automated Capture Calibration (from snapshot) Finished."
           << std::endl;
}