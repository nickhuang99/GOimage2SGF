#include "common.h" // Includes logger.h
#include "logger.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
// #include <iostream> // Replaced by logger
#include <limits>
#include <map>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <ostream>
#include <regex>
#include <set>
#include <vector>

// Using namespace std; // Avoid global using namespace std for better practice
// Using namespace cv;  // Avoid global using namespace cv for better practice

// --- Definition of Global Constants for Lab Color Tolerances ---
// These are declared as extern in common.h and defined here
const float CALIB_L_TOLERANCE_STONE = 35.0f;
const float CALIB_AB_TOLERANCE_STONE = 15.0f;

// --- NEW CONSTANT DEFINITIONS for detectSpecificColoredRoundShape ---
const int MORPH_OPEN_KERNEL_SIZE_STONE = 5;
const int MORPH_OPEN_ITERATIONS_STONE = 1; // MODIFIED (was 2, less aggressive)
const int MORPH_CLOSE_KERNEL_SIZE_STONE = 3;
const int MORPH_CLOSE_ITERATIONS_STONE =
    2; // MODIFIED (was 1, more aggressive for hole filling)
const double ABS_STONE_AREA_MIN_FACTOR = 0.4;
const double ABS_STONE_AREA_MAX_FACTOR = 2.5;

const double MIN_STONE_CIRCULARITY_WHITE = 0.65;
const double MIN_STONE_CIRCULARITY_BLACK =
    0.50; // More lenient for black stones
const int MIN_CONTOUR_POINTS_STONE = 5;

const float MAX_ROI_FACTOR_FOR_CALC = 1.0f;
struct Line {
  double value; // y for horizontal, x for vertical
  double angle;
};

// Helper struct to store matching information and define comparison
struct LineMatch {
  int matched_count;
  double score;
  double start_value;
  std::vector<std::pair<double, double>>
      matched_values; // Store (clustered_value, distance)

  bool operator<(const LineMatch &other) const {
    if (matched_count != other.matched_count) {
      return matched_count > other.matched_count;
    }
    if (score != other.score) {
      return score < other.score;
    }
    return start_value < other.start_value;
  }
};

// Helper struct for find_best_round_shape_iterative
// Helper struct for find_best_round_shape_iterative
// Helper struct for find_best_round_shape_iterative
// Helper struct for find_best_round_shape_iterative
struct CandidateBlob {
  cv::Point2f center_in_roi_coords; // Center within the ROI it was found
  double area;
  double circularity;
  float l_base_used;
  float l_tolerance_used;
  double score; // Can be used to reflect confidence or just mark as found
  std::vector<cv::Point> contour_points_in_roi; // Relative to ROI
  cv::Vec3f sampled_lab_color_from_contour;     // Lab color sampled from this
                                                // specific blob
  int classified_color_after_shape_found;       // BLACK, WHITE, EMPTY/OTHER
  cv::Rect roi_used_in_search;                  // <<-- ADD THIS LINE

  CandidateBlob()
      : area(0), circularity(0), l_base_used(0), l_tolerance_used(0),
        score(-1.0), sampled_lab_color_from_contour(-1, -1, -1),
        classified_color_after_shape_found(EMPTY) {}

  bool isValid() const {
    return area > 0 && score >= 0;
  } // Valid if area is positive and score indicates found
};

std::ostream &operator<<(std::ostream &os, CornerQuadrant quadrant) {
  os << toString(quadrant);
  return os;
}

bool compareLines(const Line &a, const Line &b) { return a.value < b.value; }

// Forward declarations for static helper functions if defined later
static int classifySingleIntersectionByDistance(
    const cv::Vec3f &intersection_lab_color, const cv::Vec3f &avg_black_calib,
    const cv::Vec3f &avg_white_calib, const cv::Vec3f &board_calib_lab);

static void
classifyIntersectionsByCalibration( // Renamed from performDirectClassification
    const std::vector<cv::Vec3f> &average_lab_values,
    const CalibrationData &calib_data,
    const std::vector<cv::Point2f> &intersection_points, int num_intersections,
    const cv::Mat &corrected_bgr_image, int adaptive_sample_radius_for_drawing,
    cv::Mat &board_state_output, cv::Mat &board_with_stones_output);

static bool adaptive_detect_stone_robust(
    const cv::Mat &rawBgrImage, CornerQuadrant targetScanQuadrant,
    const CalibrationData &calibData, cv::Point2f &out_final_raw_corner_guess,
    cv::Mat &out_final_corrected_image,
    float &out_detected_stone_radius_in_final_corrected, // Radius from final
                                                         // Pass 2 verification
    int &out_pass1_classified_color // Color classified from the robustly found
                                    // shape in Pass 1
);

cv::Rect calculateGridIntersectionROI(int target_col, int target_row,
                                      int corrected_image_width_px,
                                      int corrected_image_height_px,
                                      int grid_lines) {

  if (target_col < 0 || target_col >= grid_lines || target_row < 0 ||
      target_row >= grid_lines) {
    std::string error_msg =
        "Target column/row (" + std::to_string(target_col) + "," +
        std::to_string(target_row) +
        ") out of bounds for grid_lines=" + std::to_string(grid_lines);
    LOG_ERROR << error_msg << std::endl;
    THROWGEMERROR(error_msg);
  }
  if (grid_lines <= 1) {
    std::string error_msg = "Grid lines must be greater than 1 in "
                            "calculateGridIntersectionROI. Got: " +
                            std::to_string(grid_lines);
    LOG_ERROR << error_msg << std::endl;
    THROWGEMERROR(error_msg);
  }

  std::vector<cv::Point2f> ideal_board_corners = getBoardCornersCorrected(
      corrected_image_width_px, corrected_image_height_px);
  if (ideal_board_corners.size() != 4) {
    std::string msg = "getBoardCornersCorrected did not return 4 points in "
                      "calculateGridIntersectionROI.";
    LOG_ERROR << msg << std::endl;
    THROWGEMERROR(msg);
  }

  cv::Point2f board_top_left_px = ideal_board_corners[0];
  float grid_area_width_px =
      ideal_board_corners[1].x - ideal_board_corners[0].x;
  float grid_area_height_px =
      ideal_board_corners[3].y - ideal_board_corners[0].y;

  if (grid_area_width_px <= 0 || grid_area_height_px <= 0) {
    std::string error_msg = "Calculated grid area dimensions are non-positive "
                            "in calculateGridIntersectionROI. Width=" +
                            Num2Str(grid_area_width_px).str() +
                            ", Height=" + Num2Str(grid_area_height_px).str();
    LOG_ERROR << error_msg << std::endl;
    THROWGEMERROR(error_msg);
  }

  float avg_grid_spacing_x =
      grid_area_width_px / static_cast<float>(grid_lines - 1);
  float avg_grid_spacing_y =
      grid_area_height_px / static_cast<float>(grid_lines - 1);

  float center_x_px =
      board_top_left_px.x + static_cast<float>(target_col) * avg_grid_spacing_x;
  float center_y_px =
      board_top_left_px.y + static_cast<float>(target_row) * avg_grid_spacing_y;

  int color_sampling_radius =
      calculateAdaptiveSampleRadius(grid_area_width_px, grid_area_height_px);

  int roi_half_width =
      static_cast<int>(static_cast<float>(color_sampling_radius) * 2.5f);
  int min_practical_roi_half_width = 5;
  roi_half_width = std::max(roi_half_width, min_practical_roi_half_width);

  int max_allowed_half_width =
      static_cast<int>(std::min(avg_grid_spacing_x, avg_grid_spacing_y) *
                       MAX_ROI_FACTOR_FOR_CALC);
  max_allowed_half_width =
      std::max(max_allowed_half_width, min_practical_roi_half_width);
  roi_half_width = std::min(roi_half_width, max_allowed_half_width);

  int roi_x = static_cast<int>(center_x_px - roi_half_width);
  int roi_y = static_cast<int>(center_y_px - roi_half_width);
  int roi_side = 2 * roi_half_width;

  LOG_DEBUG << "calculateGridIntersectionROI for (" << target_row << ","
            << target_col << "): " << "Img WxH: " << corrected_image_width_px
            << "x" << corrected_image_height_px
            << ", Grid Area WxH: " << grid_area_width_px << "x"
            << grid_area_height_px << ", Spacing X,Y: " << avg_grid_spacing_x
            << "," << avg_grid_spacing_y << ", Center Px X,Y: " << center_x_px
            << "," << center_y_px << ", ColorSampRad: " << color_sampling_radius
            << ", ROI Half Width: " << roi_half_width << " (Side: " << roi_side
            << ")" << ", Calc ROI Rect: [" << roi_x << "," << roi_y << " - "
            << roi_side << "x" << roi_side << "]" << std::endl;

  return cv::Rect(roi_x, roi_y, roi_side, roi_side);
}

int detectStoneAtPosition(const cv::Mat &corrected_bgr_image, int target_col,
                          int target_row, const CalibrationData &calib_data) {
  LOG_DEBUG << "Detecting stone at position: Col=" << target_col
            << ", Row=" << target_row << std::endl;
  if (corrected_bgr_image.empty()) {
    std::string msg =
        "Input (corrected_bgr_image) is empty in detectStoneAtPosition.";
    LOG_ERROR << msg << std::endl;
    THROWGEMERROR(msg);
  }
  if (target_col < 0 || target_col > 18 || target_row < 0 || target_row > 18) {
    std::string error_msg = "Target column/row (" + std::to_string(target_col) +
                            "," + std::to_string(target_row) +
                            ") out of 0-18 bounds in detectStoneAtPosition.";
    LOG_ERROR << error_msg << std::endl;
    THROWGEMERROR(error_msg);
  }
  if (!calib_data.colors_loaded) {
    std::string msg =
        "Calibration data (stone colors) not loaded in detectStoneAtPosition.";
    LOG_ERROR << msg << std::endl;
    THROWGEMERROR(msg);
  }

  cv::Rect roi = calculateGridIntersectionROI(target_col, target_row,
                                              corrected_bgr_image.cols,
                                              corrected_bgr_image.rows);

  cv::Rect image_bounds(0, 0, corrected_bgr_image.cols,
                        corrected_bgr_image.rows);
  roi &= image_bounds;

  if (roi.width <= 0 || roi.height <= 0) {
    LOG_WARN << "Calculated ROI for (" << target_row << "," << target_col
             << ") is invalid or outside image bounds after clamping. ROI: x="
             << roi.x << ",y=" << roi.y << ",w=" << roi.width
             << ",h=" << roi.height
             << ". Image size: " << corrected_bgr_image.cols << "x"
             << corrected_bgr_image.rows << ". Assuming EMPTY." << std::endl;
    return EMPTY;
  }
  // <<< MODIFIED: Use specific corner lab values if target is a corner, else
  // use average >>>
  cv::Vec3f ref_black_lab;
  cv::Vec3f ref_white_lab;

  if (target_row == 0 && target_col == 0)
    ref_black_lab = calib_data.lab_tl; // TL
  else if (target_row == 18 && target_col == 0)
    // BL (assuming row-major, (0,18) is bottom-left-most col)
    ref_black_lab = calib_data.lab_bl;
  else
    ref_black_lab = (calib_data.lab_tl + calib_data.lab_bl) * 0.5f;
  if (target_row == 0 && target_col == 18)
    ref_white_lab = calib_data.lab_tr; // TR
  else if (target_row == 18 && target_col == 18)
    ref_white_lab = calib_data.lab_br; // BR
  else
    ref_white_lab = (calib_data.lab_tr + calib_data.lab_br) * 0.5f;

  // Fallback to averages if a specific corner logic isn't perfectly matched or
  // for non-corners
  if (ref_black_lab[0] < 0)
    ref_black_lab = (calib_data.lab_tl + calib_data.lab_bl) * 0.5f;
  if (ref_white_lab[0] < 0)
    ref_white_lab = (calib_data.lab_tr + calib_data.lab_br) * 0.5f;

  cv::Point2f detected_center;
  float detected_radius;

  LOG_DEBUG << "Checking for BLACK stone at (" << target_row << ","
            << target_col << ") within ROI {x:" << roi.x << ",y:" << roi.y
            << ",w:" << roi.width << ",h:" << roi.height << "}"
            << " (Ref Lab: " << ref_black_lab[0] << "," << ref_black_lab[1]
            << "," << ref_black_lab[2] << ")" << std::endl;
  float expected_stone_radius_for_detection = calculateAdaptiveSampleRadius(
      corrected_bgr_image.cols, corrected_bgr_image.rows);

  if (detectSpecificColoredRoundShape(
          corrected_bgr_image, roi, ref_black_lab, CALIB_L_TOLERANCE_STONE,
          CALIB_AB_TOLERANCE_STONE, expected_stone_radius_for_detection,
          detected_center, detected_radius)) {
    LOG_DEBUG << "Found BLACK stone at (" << target_row << "," << target_col
              << ")" << std::endl;
    return BLACK;
  }

  LOG_DEBUG << "Checking for WHITE stone at (" << target_row << ","
            << target_col << ") within ROI {x:" << roi.x << ",y:" << roi.y
            << ",w:" << roi.width << ",h:" << roi.height << "}"
            << " (Ref Lab: " << ref_white_lab[0] << "," << ref_white_lab[1]
            << "," << ref_white_lab[2] << ")" << std::endl;

  if (detectSpecificColoredRoundShape(
          corrected_bgr_image, roi, ref_white_lab, CALIB_L_TOLERANCE_STONE,
          CALIB_AB_TOLERANCE_STONE, expected_stone_radius_for_detection,
          detected_center, detected_radius)) {
    LOG_DEBUG << "Found WHITE stone at (" << target_row << "," << target_col
              << ")" << std::endl;
    return WHITE;
  }

  LOG_DEBUG << "No stone (BLACK or WHITE) found at (" << target_row << ","
            << target_col << "). Assuming EMPTY." << std::endl;
  return EMPTY;
}

cv::Vec3f getAverageLab(const cv::Mat &image_lab, cv::Point2f center,
                        int radius) {
  cv::Vec3d sum(0.0, 0.0, 0.0);
  std::vector<uchar> l_values, a_values, b_values;
  int x_min = std::max(0, static_cast<int>(center.x - radius));
  int x_max = std::min(image_lab.cols - 1, static_cast<int>(center.x + radius));
  int y_min = std::max(0, static_cast<int>(center.y - radius));
  int y_max = std::min(image_lab.rows - 1, static_cast<int>(center.y + radius));

  for (int y = y_min; y <= y_max; ++y) {
    for (int x = x_min; x <= x_max; ++x) {
      if (std::pow(x - center.x, 2) + std::pow(y - center.y, 2) <=
          std::pow(radius, 2)) {
        cv::Vec3b lab = image_lab.at<cv::Vec3b>(y, x);
        l_values.push_back(lab[0]);
        a_values.push_back(lab[1]);
        b_values.push_back(lab[2]);
      }
    }
  }
  size_t count = l_values.size();
  if (count > 0) {
    std::sort(l_values.begin(), l_values.end());
    std::sort(a_values.begin(), a_values.end());
    std::sort(b_values.begin(), b_values.end());
    size_t mid_index = count / 2;
    return cv::Vec3f(static_cast<float>(l_values[mid_index]),
                     static_cast<float>(a_values[mid_index]),
                     static_cast<float>(b_values[mid_index]));
  }
  LOG_WARN << "No valid pixels found for Lab averaging at center (" << center.x
           << "," << center.y << ") with radius " << radius
           << ". ROI bounds: x[" << x_min << "-" << x_max << "], y[" << y_min
           << "-" << y_max << "]. Returning invalid Lab (-1,-1,-1)."
           << std::endl;
  return cv::Vec3f(-1.0f, -1.0f, -1.0f);
}

std::vector<cv::Point2f>
loadCornersFromConfigFile(const std::string &config_path) {
  std::vector<cv::Point2f> corners;
  std::ifstream configFile(config_path);
  if (!configFile.is_open()) {
    LOG_DEBUG << "Config file '" << config_path
              << "' not found during corner load attempt." << std::endl;
    return corners;
  }
  LOG_DEBUG << "Found config file: " << config_path
            << ". Attempting to parse corners." << std::endl;
  std::map<std::string, std::string> config_data;
  std::string line;
  try {
    while (getline(configFile, line)) {
      line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
      line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);
      if (line.empty() || line[0] == '#')
        continue;
      size_t equals_pos = line.find('=');
      if (equals_pos != std::string::npos) {
        std::string key = line.substr(0, equals_pos);
        std::string value = line.substr(equals_pos + 1);
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        config_data[key] = value;
      }
    }
    configFile.close();

    if (config_data.count("TL_X_PX") && config_data.count("TL_Y_PX") &&
        config_data.count("TR_X_PX") && config_data.count("TR_Y_PX") &&
        config_data.count("BL_X_PX") && config_data.count("BL_Y_PX") &&
        config_data.count("BR_X_PX") && config_data.count("BR_Y_PX")) {
      cv::Point2f tl(std::stof(config_data["TL_X_PX"]),
                     std::stof(config_data["TL_Y_PX"]));
      cv::Point2f tr(std::stof(config_data["TR_X_PX"]),
                     std::stof(config_data["TR_Y_PX"]));
      cv::Point2f bl(std::stof(config_data["BL_X_PX"]),
                     std::stof(config_data["BL_Y_PX"]));
      cv::Point2f br(std::stof(config_data["BR_X_PX"]),
                     std::stof(config_data["BR_Y_PX"]));
      corners = {tl, tr, br, bl};
      LOG_DEBUG << "Successfully loaded corners from config file." << std::endl;
    } else {
      LOG_WARN << "Config file " << config_path
               << " missing one or more pixel coordinate keys (_PX)."
               << std::endl;
    }
  } catch (const std::exception &e) {
    LOG_ERROR << "Error parsing config file '" << config_path
              << "': " << e.what() << std::endl;
    if (configFile.is_open())
      configFile.close();
    corners.clear();
  }
  return corners;
}

void drawSimulatedGoBoard(const std::string &full_tournament_sgf_content,
                          int display_up_to_move_idx, cv::Mat &output_image,
                          int highlight_this_move_idx, int canvas_size_px) {

  // --- Drawing Constants ---
  const int base_margin_px = std::max(20, canvas_size_px / 25);
  const int label_space_px =
      std::max(20, canvas_size_px / 38); // Increased label space a bit
  const int total_margin_px = base_margin_px + label_space_px;
  const int board_proper_size_px = canvas_size_px - 2 * total_margin_px;

  if (board_proper_size_px <=
      18 * 5) { // Ensure at least 5px per line_spacing for visibility
    LOG_ERROR << "Error (drawSimulatedGoBoard): Canvas size " << canvas_size_px
              << "px is too small for margins and a legible board. Board "
                 "proper size: "
              << board_proper_size_px << std::endl;
    output_image = cv::Mat::zeros(canvas_size_px, canvas_size_px, CV_8UC3);
    cv::putText(output_image, "Canvas too small",
                cv::Point(10, canvas_size_px / 2), cv::FONT_HERSHEY_SIMPLEX,
                0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    return;
  }

  const float line_spacing_px =
      static_cast<float>(board_proper_size_px) / 18.0f;
  const int stone_radius_px = static_cast<int>(line_spacing_px * 0.47f);
  const int hoshi_radius_px = std::max(
      2, static_cast<int>(line_spacing_px * 0.10f)); // Slightly smaller hoshi
  const double label_font_scale =
      std::max(0.3, line_spacing_px * 0.012 * (760.0 / canvas_size_px));
  const int label_font_thickness = 1;
  const int font_face = cv::FONT_HERSHEY_SIMPLEX;

  const cv::Scalar board_color_bgr(210, 180, 140); // Wood color: BGR (Tan-like)
  const cv::Scalar line_color_bgr(30, 30, 30);
  const cv::Scalar label_color_bgr(10, 10, 10);
  const cv::Scalar stone_color_black_bgr(25, 25, 25);
  const cv::Scalar stone_color_white_bgr(235, 235, 235);
  const cv::Scalar stone_outline_color_bgr(70, 70, 70);
  const cv::Scalar highlight_color_bgr(
      50, 220, 255); // Brighter Yellow/Gold for highlight
  const int highlight_thickness = std::max(2, stone_radius_px / 6);

  output_image =
      cv::Mat(canvas_size_px, canvas_size_px, CV_8UC3, board_color_bgr);

  // --- Draw Grid Lines ---
  for (int i = 0; i < 19; ++i) {
    float current_pos_on_board = i * line_spacing_px;
    // Add 0.5f for better pixel alignment of thin lines
    float x_coord =
        static_cast<float>(total_margin_px) + current_pos_on_board + 0.5f;
    float y_coord =
        static_cast<float>(total_margin_px) + current_pos_on_board + 0.5f;

    cv::line(output_image,
             cv::Point2f(x_coord, static_cast<float>(total_margin_px)),
             cv::Point2f(x_coord, static_cast<float>(total_margin_px +
                                                     board_proper_size_px)),
             line_color_bgr, 1, cv::LINE_AA);
    cv::line(
        output_image, cv::Point2f(static_cast<float>(total_margin_px), y_coord),
        cv::Point2f(static_cast<float>(total_margin_px + board_proper_size_px),
                    y_coord),
        line_color_bgr, 1, cv::LINE_AA);
  }

  // --- Draw Hoshi Points ---
  int hoshi_indices[] = {3, 9, 15};
  for (int r_idx : hoshi_indices) {
    for (int c_idx : hoshi_indices) {
      float hoshi_x_px =
          static_cast<float>(total_margin_px) + c_idx * line_spacing_px;
      float hoshi_y_px =
          static_cast<float>(total_margin_px) + r_idx * line_spacing_px;
      cv::circle(output_image, cv::Point2f(hoshi_x_px, hoshi_y_px),
                 hoshi_radius_px, line_color_bgr, -1, cv::LINE_AA);
    }
  }

  // --- Draw Coordinate Labels ---
  std::string letters = "ABCDEFGHJKLMNOPQRST";
  for (int i = 0; i < 19; ++i) {
    std::string num_label = std::to_string(
        19 - i); // Numbers 19-1 (traditional top to bottom for rows from human
                 // perspective) If SGF row 0 is top, display i+1
    num_label = std::to_string(i + 1); // 1-19 from top to bottom for rows

    std::string char_label = "";
    if (i < static_cast<int>(letters.length()))
      char_label += letters[i];

    float line_center_on_board_px =
        i * line_spacing_px +
        line_spacing_px / 2.0f; // Center of the cell/line band
    float absolute_line_pos_px =
        static_cast<float>(total_margin_px) + i * line_spacing_px;

    cv::Size num_text_size = cv::getTextSize(
        num_label, font_face, label_font_scale, label_font_thickness, nullptr);
    // Left Numeric Label (numbers 1-19, for rows)
    cv::putText(
        output_image, num_label,
        cv::Point(base_margin_px - (num_text_size.width > base_margin_px
                                        ? 0
                                        : num_text_size.width /
                                              2), // Adjust to keep in margin
                  static_cast<int>(absolute_line_pos_px +
                                   num_text_size.height / 2.0f)),
        font_face, label_font_scale, label_color_bgr, label_font_thickness,
        cv::LINE_AA);
    // Right Numeric Label
    cv::putText(output_image, num_label,
                cv::Point(canvas_size_px - base_margin_px -
                              (num_text_size.width > base_margin_px
                                   ? num_text_size.width
                                   : num_text_size.width / 2),
                          static_cast<int>(absolute_line_pos_px +
                                           num_text_size.height / 2.0f)),
                font_face, label_font_scale, label_color_bgr,
                label_font_thickness, cv::LINE_AA);

    if (!char_label.empty()) {
      cv::Size char_text_size =
          cv::getTextSize(char_label, font_face, label_font_scale,
                          label_font_thickness, nullptr);
      // Top Character Label (letters A-T, for columns)
      cv::putText(output_image, char_label,
                  cv::Point(static_cast<int>(absolute_line_pos_px -
                                             char_text_size.width / 2.0f),
                            base_margin_px + char_text_size.height / 2),
                  font_face, label_font_scale, label_color_bgr,
                  label_font_thickness, cv::LINE_AA);
      // Bottom Character Label
      cv::putText(output_image, char_label,
                  cv::Point(static_cast<int>(absolute_line_pos_px -
                                             char_text_size.width / 2.0f),
                            canvas_size_px - base_margin_px -
                                char_text_size.height / 2),
                  font_face, label_font_scale, label_color_bgr,
                  label_font_thickness, cv::LINE_AA);
    }
  }

  // --- Parse SGF & Reconstruct Board State (Logic from your Phase 2 version)
  // ---
  std::set<std::pair<int, int>> initial_setup_black, initial_setup_white;
  std::vector<Move> all_game_moves;
  SGFHeader header;
  try {
    header = parseSGFHeader(full_tournament_sgf_content);
    parseSGFGame(full_tournament_sgf_content, initial_setup_black,
                 initial_setup_white, all_game_moves);
  } catch (const SGFError &e) { /* ... error handling ... */
    return;
  }

  std::vector<std::tuple<int, int, int, int>> stones_on_board_with_numbers;
  cv::Mat current_board_state_internal(19, 19, CV_8U, cv::Scalar(EMPTY));

  for (const auto &sc : initial_setup_black) {
    if (sc.first >= 0 && sc.first < 19 && sc.second >= 0 && sc.second < 19) {
      current_board_state_internal.at<uchar>(sc.first, sc.second) = BLACK;
      stones_on_board_with_numbers.emplace_back(sc.first, sc.second, BLACK, 0);
    }
  }
  for (const auto &sc : initial_setup_white) {
    if (sc.first >= 0 && sc.first < 19 && sc.second >= 0 && sc.second < 19) {
      current_board_state_internal.at<uchar>(sc.first, sc.second) = WHITE;
      stones_on_board_with_numbers.emplace_back(sc.first, sc.second, WHITE, 0);
    }
  }

  int actual_bw_move_count = 0;
  for (const auto &move : all_game_moves) {
    if (move.player == BLACK || move.player == WHITE) {
      actual_bw_move_count++;
      if (actual_bw_move_count > display_up_to_move_idx &&
          display_up_to_move_idx >=
              0) { // display_up_to_move_idx < 0 means show all
        break;
      }
      if (move.row >= 0 && move.row < 19 && move.col >= 0 && move.col < 19) {
        current_board_state_internal.at<uchar>(move.row, move.col) =
            move.player;
        stones_on_board_with_numbers.erase(
            std::remove_if(stones_on_board_with_numbers.begin(),
                           stones_on_board_with_numbers.end(),
                           [&](const auto &s) {
                             return std::get<0>(s) == move.row &&
                                    std::get<1>(s) == move.col;
                           }),
            stones_on_board_with_numbers.end());
        stones_on_board_with_numbers.emplace_back(
            move.row, move.col, move.player, actual_bw_move_count);
      }
      for (const auto &cap_coord : move.capturedStones) {
        if (cap_coord.first >= 0 && cap_coord.first < 19 &&
            cap_coord.second >= 0 && cap_coord.second < 19) {
          current_board_state_internal.at<uchar>(cap_coord.first,
                                                 cap_coord.second) = EMPTY;
          stones_on_board_with_numbers.erase(
              std::remove_if(stones_on_board_with_numbers.begin(),
                             stones_on_board_with_numbers.end(),
                             [&](const auto &s) {
                               return std::get<0>(s) == cap_coord.first &&
                                      std::get<1>(s) == cap_coord.second;
                             }),
              stones_on_board_with_numbers.end());
        }
      }
    } else if (move.player ==
               EMPTY) { /* ... handle standalone AE nodes if necessary ... */
    }
  }

  // --- Draw Stones with Numbers from the reconstructed state ---
  auto drawStoneWithNumberAndHighlight = [&](int r, int c, int stone_color,
                                             int move_num_label,
                                             bool highlight) {
    // ... (stone drawing and numbering logic from your Phase 2 version) ...
    // This includes font scaling, text centering, and highlight drawing.
    if (r >= 0 && r < 19 && c >= 0 && c < 19 && stone_color != EMPTY) {
      float stone_x_px =
          static_cast<float>(total_margin_px) + c * line_spacing_px;
      float stone_y_px =
          static_cast<float>(total_margin_px) + r * line_spacing_px;
      cv::Scalar color_bgr = (stone_color == BLACK) ? stone_color_black_bgr
                                                    : stone_color_white_bgr;
      cv::circle(output_image, cv::Point2f(stone_x_px, stone_y_px),
                 stone_radius_px, color_bgr, -1, cv::LINE_AA);
      cv::circle(output_image, cv::Point2f(stone_x_px, stone_y_px),
                 stone_radius_px, stone_outline_color_bgr, 1, cv::LINE_AA);
      if (highlight) {
        cv::circle(output_image, cv::Point2f(stone_x_px, stone_y_px),
                   stone_radius_px + highlight_thickness, highlight_color_bgr,
                   highlight_thickness, cv::LINE_AA);
      }
      if (move_num_label >= 0) {
        std::string num_str = std::to_string(move_num_label);
        cv::Scalar text_color = (stone_color == BLACK)
                                    ? cv::Scalar(230, 230, 230)
                                    : cv::Scalar(25, 25, 25);
        double base_font_scale_for_number =
            line_spacing_px * 0.018 * (760.0 / canvas_size_px);
        double current_font_scale = base_font_scale_for_number;
        if (num_str.length() == 2)
          current_font_scale *= 0.80;
        else if (num_str.length() >= 3)
          current_font_scale *= 0.65;
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(num_str, font_face,
                                             current_font_scale, 1, &baseline);
        if (text_size.width > stone_radius_px * 1.6 ||
            text_size.height > stone_radius_px * 1.6) {
          current_font_scale *= ((stone_radius_px * 1.6) /
                                 std::max(text_size.width, text_size.height));
          text_size = cv::getTextSize(num_str, font_face, current_font_scale, 1,
                                      &baseline);
        }
        cv::Point text_org(
            static_cast<int>(stone_x_px - text_size.width / 2.0f),
            static_cast<int>(
                stone_y_px + text_size.height / 2.0f -
                baseline *
                    0.6f)); // Adjusted baseline slightly for better centering
        cv::putText(output_image, num_str, text_org, font_face,
                    current_font_scale, text_color, label_font_thickness,
                    cv::LINE_AA);
      }
    }
  };

  for (const auto &stone_data : stones_on_board_with_numbers) {
    int r = std::get<0>(stone_data);
    int c = std::get<1>(stone_data);
    int color = std::get<2>(stone_data);
    int move_label = std::get<3>(stone_data);

    if (current_board_state_internal.at<uchar>(r, c) == color) {
      bool highlight = false;
      if (highlight_this_move_idx == 0 &&
          move_label == 0) { // Highlight all setup stones if highlight_idx is 0
        highlight = true;
      } else if (move_label == highlight_this_move_idx &&
                 highlight_this_move_idx > 0) { // Highlight specific B/W move
        highlight = true;
      }
      drawStoneWithNumberAndHighlight(r, c, color, move_label, highlight);
    }
  }

  if (bDebug) {
    LOG_DEBUG << "Debug (drawSimulatedGoBoard): Displaying board state after "
              << display_up_to_move_idx << " B/W moves." << std::endl;
    if (highlight_this_move_idx != -1) {
      LOG_DEBUG << "Debug (drawSimulatedGoBoard): Highlighting move index "
                << highlight_this_move_idx << std::endl;
    }
  }
}

static int classifySingleIntersectionByDistance(
    const cv::Vec3f &intersection_lab_color, const cv::Vec3f &avg_black_calib,
    const cv::Vec3f &avg_white_calib, const cv::Vec3f &board_calib_lab) {
  const float WEIGHT_L = 0.4f;
  const float WEIGHT_A = 0.8f;
  const float WEIGHT_B = 2.2f;
  const float MAX_DIST_STONE_WEIGHTED = 30.0f;
  const float MAX_DIST_BOARD_WEIGHTED = 35.0f;
  const float L_HEURISTIC_BLACK_OFFSET = 30.0f;
  const float L_HEURISTIC_WHITE_OFFSET = 20.0f;

  if (intersection_lab_color[0] < 0) {
    std::string msg =
        "Invalid Lab sample in classifySingleIntersectionByDistance: L=" +
        Num2Str(intersection_lab_color[0]).str();
    LOG_ERROR << msg << std::endl;
    THROWGEMERROR(msg);
  }

  if (intersection_lab_color[0] <
      (avg_black_calib[0] - L_HEURISTIC_BLACK_OFFSET)) {
    LOG_DEBUG << "      L-Heuristic: Classified as BLACK (L="
              << intersection_lab_color[0]
              << " < BlackRefL=" << avg_black_calib[0] << " - "
              << L_HEURISTIC_BLACK_OFFSET << ")" << std::endl;
    return BLACK;
  }
  if (intersection_lab_color[0] >
      (avg_white_calib[0] + L_HEURISTIC_WHITE_OFFSET)) {
    LOG_DEBUG << "      L-Heuristic: Classified as WHITE (L="
              << intersection_lab_color[0]
              << " > WhiteRefL=" << avg_white_calib[0] << " + "
              << L_HEURISTIC_WHITE_OFFSET << ")" << std::endl;
    return WHITE;
  }

  auto calculate_weighted_distance = [&](const cv::Vec3f &c1,
                                         const cv::Vec3f &c2) {
    float dL = c1[0] - c2[0];
    float dA = c1[1] - c2[1];
    float dB = c1[2] - c2[2];
    return std::sqrt(WEIGHT_L * dL * dL + WEIGHT_A * dA * dA +
                     WEIGHT_B * dB * dB);
  };

  float dist_b_w =
      calculate_weighted_distance(intersection_lab_color, avg_black_calib);
  float dist_w_w =
      calculate_weighted_distance(intersection_lab_color, avg_white_calib);
  float dist_empty_w =
      calculate_weighted_distance(intersection_lab_color, board_calib_lab);

  float min_weighted_dist = std::min({dist_b_w, dist_w_w, dist_empty_w});
  int classification = EMPTY;

  if (min_weighted_dist == dist_b_w && dist_b_w < MAX_DIST_STONE_WEIGHTED)
    classification = BLACK;
  else if (min_weighted_dist == dist_w_w && dist_w_w < MAX_DIST_STONE_WEIGHTED)
    classification = WHITE;
  else if (min_weighted_dist == dist_empty_w &&
           dist_empty_w < MAX_DIST_BOARD_WEIGHTED)
    classification = EMPTY;
  else {
    LOG_DEBUG << "      WeightedDist: All checks failed or distances too high. "
                 "MinWDist="
              << min_weighted_dist
              << ". Sample Lab:" << intersection_lab_color[0] << ","
              << intersection_lab_color[1] << "," << intersection_lab_color[2]
              << ". Defaulting to Empty." << std::endl;
    classification = EMPTY;
  }
  return classification;
}

static void classifyIntersectionsByCalibration(
    const std::vector<cv::Vec3f> &average_lab_values,
    const CalibrationData &calib_data,
    const std::vector<cv::Point2f> &intersection_points, int num_intersections,
    const cv::Mat &corrected_bgr_image, int adaptive_sample_radius_for_drawing,
    cv::Mat &board_state_output, cv::Mat &board_with_stones_output) {
  LOG_DEBUG << "Classifying " << num_intersections
            << " intersections by calibration..." << std::endl;

  cv::Vec3f avg_black_calib = (calib_data.lab_tl + calib_data.lab_bl) * 0.5f;
  cv::Vec3f avg_white_calib = (calib_data.lab_tr + calib_data.lab_br) * 0.5f;

  board_state_output = cv::Mat(19, 19, CV_8U, cv::Scalar(EMPTY));
  board_with_stones_output = corrected_bgr_image.clone();

  LOG_DEBUG << std::fixed << std::setprecision(1)
            << "    Using Ref Black Lab: " << avg_black_calib[0] << ","
            << avg_black_calib[1] << "," << avg_black_calib[2]
            << ", Ref White Lab: " << avg_white_calib[0] << ","
            << avg_white_calib[1] << "," << avg_white_calib[2]
            << ", Ref Board Lab: " << calib_data.lab_board_avg[0] << ","
            << calib_data.lab_board_avg[1] << "," << calib_data.lab_board_avg[2]
            << " (Using L-heuristic and weighted distances with internal "
               "thresholds/weights)"
            << std::endl;

  for (int i = 0; i < num_intersections; ++i) {
    int row = i / 19;
    int col = i % 19;
    if (row >= 19 || col >= 19)
      continue;

    cv::Vec3f current_intersection_lab = average_lab_values[i];
    int classification = classifySingleIntersectionByDistance(
        current_intersection_lab, avg_black_calib, avg_white_calib,
        calib_data.lab_board_avg);

    board_state_output.at<uchar>(row, col) = classification;

    if (current_intersection_lab[0] >= 0 &&
        Logger::getGlobalLogLevel() >= LogLevel::DEBUG) {
      float dist_b_unweighted =
          cv::norm(current_intersection_lab, avg_black_calib, cv::NORM_L2);
      float dist_w_unweighted =
          cv::norm(current_intersection_lab, avg_white_calib, cv::NORM_L2);
      float dist_e_unweighted = cv::norm(current_intersection_lab,
                                         calib_data.lab_board_avg, cv::NORM_L2);
      std::string stone_type_str =
          (classification == BLACK)
              ? "Black"
              : (classification == WHITE ? "White" : "Empty");
      LOG_DEBUG << "    Int [" << std::setw(2) << row << "," << std::setw(2)
                << col << "] Lab: [" << std::setw(5)
                << current_intersection_lab[0] << "," << std::setw(5)
                << current_intersection_lab[1] << "," << std::setw(5)
                << current_intersection_lab[2] << "]"
                << " D(B):" << std::setw(5) << dist_b_unweighted
                << " D(W):" << std::setw(5) << dist_w_unweighted
                << " D(E):" << std::setw(5) << dist_e_unweighted
                << " -> Class: " << stone_type_str << " (" << classification
                << ")" << std::endl;
    }

    if (static_cast<size_t>(i) < intersection_points.size()) {
      if (classification == BLACK) {
        cv::circle(board_with_stones_output, intersection_points[i],
                   adaptive_sample_radius_for_drawing, cv::Scalar(0, 0, 0), -1);
      } else if (classification == WHITE) {
        cv::circle(board_with_stones_output, intersection_points[i],
                   adaptive_sample_radius_for_drawing,
                   cv::Scalar(255, 255, 255), -1);
      } else if (current_intersection_lab[0] >= 0) {
        cv::circle(board_with_stones_output, intersection_points[i],
                   adaptive_sample_radius_for_drawing / 2,
                   cv::Scalar(0, 255, 0), 1);
      } else {
        cv::circle(board_with_stones_output, intersection_points[i],
                   adaptive_sample_radius_for_drawing / 2,
                   cv::Scalar(0, 165, 255), 1);
      }
    }
  }
  if (Logger::getGlobalLogLevel() >= LogLevel::DEBUG) {
    cv::imshow("Direct Classification Result (Helper)",
               board_with_stones_output);
    cv::waitKey(0);
    cv::destroyWindow("Direct Classification Result (Helper)");
  }
  LOG_DEBUG << "Finished classifying intersections." << std::endl;
}

CalibrationData loadCalibrationData(const std::string &config_path) {
  CalibrationData data;
  std::ifstream configFile(config_path);
  if (!configFile.is_open()) {
    LOG_WARN << "Calibration config file not found: " << config_path
             << ". Returning empty/default data." << std::endl;
    return data;
  }

  LOG_INFO << "Parsing calibration config file: " << config_path << std::endl;
  std::map<std::string, std::string> config_map;
  std::string line;
  int line_num = 0;
  while (getline(configFile, line)) {
    line_num++;
    line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
    line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);
    if (line.empty() || line[0] == '#')
      continue;
    size_t equals_pos = line.find('=');
    if (equals_pos != std::string::npos) {
      std::string key = line.substr(0, equals_pos);
      std::string value = line.substr(equals_pos + 1);
      key.erase(0, key.find_first_not_of(" \t"));
      key.erase(key.find_last_not_of(" \t") + 1);
      value.erase(0, value.find_first_not_of(" \t"));
      value.erase(value.find_last_not_of(" \t") + 1);
      config_map[key] = value;
    } else {
      LOG_WARN << "Invalid line format (no '=') at line " << line_num << " in "
               << config_path << ": " << line << std::endl;
    }
  }
  configFile.close();

  try {
    if (config_map.count("DevicePath")) {
      data.device_path = config_map["DevicePath"];
      data.device_path_loaded = true;
      LOG_DEBUG << "  Loaded DevicePath: " << data.device_path << std::endl;
    } else {
      LOG_WARN << "  DevicePath missing from config " << config_path << "."
               << std::endl;
    }

    if (config_map.count("ImageWidth") && config_map.count("ImageHeight")) {
      data.image_width = std::stoi(config_map["ImageWidth"]);
      data.image_height = std::stoi(config_map["ImageHeight"]);
      data.dimensions_loaded = true;
      LOG_DEBUG << "  Loaded Dimensions at Calib: " << data.image_width << "x"
                << data.image_height << std::endl;
    } else {
      LOG_WARN << "  ImageWidth or ImageHeight missing from config "
               << config_path << std::endl;
    }

    if (config_map.count("TL_X_PX") && config_map.count("TL_Y_PX") &&
        config_map.count("TR_X_PX") && config_map.count("TR_Y_PX") &&
        config_map.count("BL_X_PX") && config_map.count("BL_Y_PX") &&
        config_map.count("BR_X_PX") && config_map.count("BR_Y_PX")) {
      cv::Point2f tl(std::stof(config_map["TL_X_PX"]),
                     std::stof(config_map["TL_Y_PX"]));
      cv::Point2f tr(std::stof(config_map["TR_X_PX"]),
                     std::stof(config_map["TR_Y_PX"]));
      cv::Point2f bl(std::stof(config_map["BL_X_PX"]),
                     std::stof(config_map["BL_Y_PX"]));
      cv::Point2f br(std::stof(config_map["BR_X_PX"]),
                     std::stof(config_map["BR_Y_PX"]));
      data.corners = {tl, tr, br, bl};
      data.corners_loaded = true;
      LOG_DEBUG << "  Loaded Corners (TL,TR,BR,BL): (" << tl.x << "," << tl.y
                << "), (" << tr.x << "," << tr.y << "), (" << br.x << ","
                << br.y << "), (" << bl.x << "," << bl.y << ")" << std::endl;
    } else {
      LOG_WARN << "  One or more corner pixel keys (_PX) missing from config "
               << config_path << std::endl;
    }

    if (config_map.count("TL_L") && config_map.count("TL_A") &&
        config_map.count("TL_B") && config_map.count("TR_L") &&
        config_map.count("TR_A") && config_map.count("TR_B") &&
        config_map.count("BL_L") && config_map.count("BL_A") &&
        config_map.count("BL_B") && config_map.count("BR_L") &&
        config_map.count("BR_A") && config_map.count("BR_B")) {
      data.lab_tl = cv::Vec3f(std::stof(config_map["TL_L"]),
                              std::stof(config_map["TL_A"]),
                              std::stof(config_map["TL_B"]));
      data.lab_tr = cv::Vec3f(std::stof(config_map["TR_L"]),
                              std::stof(config_map["TR_A"]),
                              std::stof(config_map["TR_B"]));
      data.lab_bl = cv::Vec3f(std::stof(config_map["BL_L"]),
                              std::stof(config_map["BL_A"]),
                              std::stof(config_map["BL_B"]));
      data.lab_br = cv::Vec3f(std::stof(config_map["BR_L"]),
                              std::stof(config_map["BR_A"]),
                              std::stof(config_map["BR_B"]));
      data.colors_loaded = true;
      LOG_DEBUG << "  Loaded Corner Stone Colors (TL,TR,BL,BR)." << std::endl;
    } else {
      LOG_WARN
          << "  One or more corner Lab color keys (L/A/B) missing from config "
          << config_path << std::endl;
    }

    if (config_map.count("BOARD_L_AVG") && config_map.count("BOARD_A_AVG") &&
        config_map.count("BOARD_B_AVG")) {
      data.lab_board_avg = cv::Vec3f(std::stof(config_map["BOARD_L_AVG"]),
                                     std::stof(config_map["BOARD_A_AVG"]),
                                     std::stof(config_map["BOARD_B_AVG"]));
      data.board_color_loaded = true;
      LOG_DEBUG << std::fixed << std::setprecision(1)
                << "  Loaded Average Board Lab: [" << data.lab_board_avg[0]
                << "," << data.lab_board_avg[1] << "," << data.lab_board_avg[2]
                << "]" << std::endl;
    } else {
      LOG_WARN << "  Average Board Lab color keys (BOARD_L/A/B_AVG) missing "
                  "from config "
               << config_path << std::endl;
    }

  } catch (const std::invalid_argument &ia) {
    LOG_ERROR << "Invalid number format in config file '" << config_path
              << "'. " << ia.what() << std::endl;
    data = CalibrationData();
  } catch (const std::out_of_range &oor) {
    LOG_ERROR << "Number out of range in config file '" << config_path << "'. "
              << oor.what() << std::endl;
    data = CalibrationData();
  } catch (const std::exception &e) {
    LOG_ERROR << "Generic error parsing config file '" << config_path
              << "': " << e.what() << std::endl;
    data = CalibrationData();
  }
  return data;
}

std::vector<cv::Point2f> getBoardCorners(const cv::Mat &inputImage) {
  std::vector<cv::Point2f> board_corners_result;
  LOG_DEBUG << "getBoardCorners: Determining board corners for image "
            << inputImage.cols << "x" << inputImage.rows << std::endl;

  int current_width = inputImage.cols;
  int current_height = inputImage.rows;
  CalibrationData calib_data = loadCalibrationData(CALIB_CONFIG_PATH);

  if (calib_data.corners_loaded && calib_data.dimensions_loaded &&
      calib_data.image_width == current_width &&
      calib_data.image_height == current_height) {
    LOG_INFO << "Using corners from config file (dimensions match)."
             << std::endl;
    board_corners_result = calib_data.corners;
  } else {
    if (calib_data.corners_loaded && calib_data.dimensions_loaded) {
      LOG_WARN << "Config dimensions (" << calib_data.image_width << "x"
               << calib_data.image_height << ") mismatch current image ("
               << current_width << "x" << current_height
               << "). Ignoring config corners and falling back to defaults."
               << std::endl;
    } else if (!calib_data.corners_loaded) {
      LOG_WARN
          << "Corner data not loaded from config. Falling back to defaults."
          << std::endl;
    } else if (!calib_data.dimensions_loaded) {
      LOG_WARN << "Image dimensions not loaded from config. Falling back to "
                  "defaults."
               << std::endl;
    }

    LOG_INFO << "Falling back to hardcoded percentage values for board corners."
             << std::endl;
    board_corners_result = {
        cv::Point2f(current_width * 0.15f, current_height * 0.15f), // TL
        cv::Point2f(current_width * 0.85f, current_height * 0.15f), // TR
        cv::Point2f(current_width * 0.85f, current_height * 0.85f), // BR
        cv::Point2f(current_width * 0.15f, current_height * 0.85f)  // BL
    };
  }
  return board_corners_result;
}

std::vector<cv::Point2f> getBoardCornersCorrected(int width, int height) {
  float margin_percent = 0.10f;
  return {
      cv::Point2f(width * margin_percent, height * margin_percent),
      cv::Point2f(width * (1.0f - margin_percent), height * margin_percent),
      cv::Point2f(width * (1.0f - margin_percent),
                  height * (1.0f - margin_percent)),
      cv::Point2f(width * margin_percent, height * (1.0f - margin_percent))};
}

cv::Mat correctPerspective(const cv::Mat &image) {
  LOG_INFO << "Correcting perspective for image of size " << image.cols << "x"
           << image.rows << std::endl;
  if (image.empty()) {
    LOG_ERROR << "Input image to correctPerspective is empty." << std::endl;
    return cv::Mat();
  }
  int width = image.cols;
  int height = image.rows;
  std::vector<cv::Point2f> input_corners = getBoardCorners(image);
  if (input_corners.size() != 4) {
    std::string error_msg =
        "Failed to get 4 input corners for perspective correction. Received " +
        std::to_string(input_corners.size()) + " corners.";
    LOG_ERROR << error_msg << std::endl;
    THROWGEMERROR(error_msg);
  }
  std::vector<cv::Point2f> output_corners =
      getBoardCornersCorrected(width, height);
  cv::Mat perspective_matrix =
      cv::getPerspectiveTransform(input_corners, output_corners);
  cv::Mat corrected_image;
  cv::warpPerspective(image, corrected_image, perspective_matrix,
                      cv::Size(width, height));

  if (Logger::getGlobalLogLevel() >= LogLevel::DEBUG) {
    cv::Mat display_original = image.clone();
    cv::Mat display_corrected = corrected_image.clone();
    for (size_t i = 0; i < input_corners.size(); ++i)
      cv::circle(display_original, input_corners[i], 5, cv::Scalar(0, 0, 255),
                 -1);
    for (size_t i = 0; i < output_corners.size(); ++i)
      cv::circle(display_corrected, output_corners[i], 5, cv::Scalar(0, 255, 0),
                 -1);

    LOG_DEBUG << "Displaying original image with input corners for perspective "
                 "correction."
              << std::endl;
    cv::imshow("Original Image with Input Corners (CorrectPerspective)",
               display_original);
    LOG_DEBUG << "Displaying corrected image with output corners." << std::endl;
    cv::imshow("Corrected Image with Output Corners (CorrectPerspective)",
               display_corrected);
    cv::waitKey(0);
    cv::destroyWindow("Original Image with Input Corners (CorrectPerspective)");
    cv::destroyWindow(
        "Corrected Image with Output Corners (CorrectPerspective)");
  }
  LOG_INFO << "Perspective correction finished." << std::endl;
  return corrected_image;
}

cv::Mat preprocessImage(const cv::Mat &image,
                        bool /*bDebug_param_is_now_unused_by_logging*/) {
  LOG_DEBUG << "Preprocessing image " << image.cols << "x" << image.rows
            << "..." << std::endl;
  cv::Mat gray, blurred, edges, result;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

  if (false && Logger::getGlobalLogLevel() >=
                   LogLevel::DEBUG) { // Original was if(bDebug && false)
    cv::imshow("Blurred (preprocessImage)", blurred);
    cv::waitKey(0);
    cv::destroyWindow("Blurred (preprocessImage)");
  }

  cv::adaptiveThreshold(blurred, edges, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv::THRESH_BINARY, 11, 2);
  if (false && Logger::getGlobalLogLevel() >= LogLevel::DEBUG) {
    cv::imshow("Edges (Before Morph) (preprocessImage)", edges);
    cv::waitKey(0);
    cv::destroyWindow("Edges (Before Morph) (preprocessImage)");
  }

  cv::Canny(edges, result, 50, 150, 3);
  if (false && Logger::getGlobalLogLevel() >= LogLevel::DEBUG) {
    cv::imshow("Canny (preprocessImage)", result);
    cv::waitKey(0);
    cv::destroyWindow("Canny (preprocessImage)");
  }
  LOG_DEBUG << "Preprocessing finished." << std::endl;
  return result;
}

std::vector<cv::Vec4i>
detectLineSegments(const cv::Mat &edges,
                   bool /*bDebug_param_is_now_unused_by_logging*/) {
  LOG_DEBUG << "Detecting line segments from edges image " << edges.cols << "x"
            << edges.rows << std::endl;
  int width = edges.cols;
  int height = edges.rows;
  std::vector<cv::Point2f> board_corners =
      getBoardCornersCorrected(width, height);
  if (board_corners.size() != 4) {
    std::string msg =
        "Failed to get corrected board corners in detectLineSegments.";
    LOG_ERROR << msg << std::endl;
    THROWGEMERROR(msg);
  }
  float board_height_px = board_corners[2].y - board_corners[0].y;
  float board_width_px = board_corners[1].x - board_corners[0].x;
  int margin = 10;
  cv::Rect board_rect(static_cast<int>(board_corners[0].x) - margin,
                      static_cast<int>(board_corners[0].y) - margin,
                      static_cast<int>(board_width_px) + 2 * margin,
                      static_cast<int>(board_height_px) + 2 * margin);
  board_rect &= cv::Rect(0, 0, width, height); // Clamp to image boundaries
  cv::Mat board_mask = cv::Mat::zeros(height, width, CV_8U);
  if (board_rect.width > 0 &&
      board_rect.height > 0) { // Ensure rect is valid before using it
    board_mask(board_rect) = 255;
  } else {
    LOG_WARN << "Board rectangle for masking in detectLineSegments is invalid: "
             << board_rect << std::endl;
  }

  cv::Mat masked_edges;
  cv::bitwise_and(edges, board_mask, masked_edges);
  std::vector<cv::Vec4i> all_segments;
  int hough_threshold = 30;
  int min_line_length = 40;
  int max_line_gap = 15;
  cv::HoughLinesP(masked_edges, all_segments, 1, CV_PI / 180, hough_threshold,
                  min_line_length, max_line_gap);

  LOG_DEBUG << "Total detected line segments: " << all_segments.size()
            << std::endl;
  if (false && Logger::getGlobalLogLevel() >=
                   LogLevel::DEBUG) { // Original was if(bDebug && false)
    cv::Mat mask_and_lines = edges.clone();
    cv::rectangle(mask_and_lines, board_rect, cv::Scalar(128), 2);
    LOG_DEBUG << "----all line segments (detectLineSegments)----" << std::endl;
    for (const auto &line_seg : all_segments) {
      cv::line(mask_and_lines, cv::Point(line_seg[0], line_seg[1]),
               cv::Point(line_seg[2], line_seg[3]), cv::Scalar(255), 1);
      LOG_DEBUG << "[" << line_seg[0] << "," << line_seg[1] << "]:" << "["
                << line_seg[2] << "," << line_seg[3] << "]" << std::endl;
    }
    cv::imshow("Board Mask and All Line Segments", mask_and_lines);
    cv::waitKey(0);
    cv::destroyWindow("Board Mask and All Line Segments");
  }
  return all_segments;
}

std::pair<std::vector<Line>, std::vector<Line>>
convertSegmentsToLines(const std::vector<cv::Vec4i> &all_segments,
                       bool /*bDebug_param_is_now_unused_by_logging*/) {
  LOG_DEBUG << "Converting " << all_segments.size() << " segments to lines..."
            << std::endl;
  std::vector<Line> horizontal_lines_raw, vertical_lines_raw;
  double angle_tolerance = CV_PI / 180.0 * 10;
  for (const auto &segment : all_segments) {
    cv::Point pt1(segment[0], segment[1]);
    cv::Point pt2(segment[2], segment[3]);
    double angle = std::atan2(static_cast<double>(pt2.y - pt1.y),
                              static_cast<double>(pt2.x - pt1.x));
    if (angle < 0)
      angle += CV_PI;
    bool is_horizontal = (std::abs(angle) < angle_tolerance ||
                          std::abs(angle - CV_PI) < angle_tolerance);
    bool is_vertical = (std::abs(angle - CV_PI / 2.0) < angle_tolerance);
    if (is_horizontal)
      horizontal_lines_raw.push_back({(pt1.y + pt2.y) / 2.0, angle});
    else if (is_vertical)
      vertical_lines_raw.push_back({(pt1.x + pt2.x) / 2.0, angle});
  }
  std::sort(horizontal_lines_raw.begin(), horizontal_lines_raw.end(),
            compareLines);
  std::sort(vertical_lines_raw.begin(), vertical_lines_raw.end(), compareLines);
  LOG_DEBUG << "Raw horizontal lines count (after angle classification): "
            << horizontal_lines_raw.size() << std::endl;
  LOG_DEBUG << "Raw vertical lines count (after angle classification): "
            << vertical_lines_raw.size() << std::endl;
  return std::make_pair(horizontal_lines_raw, vertical_lines_raw);
}

std::vector<double>
clusterAndAverageLines(const std::vector<Line> &raw_lines, double threshold,
                       bool /*bDebug_param_is_now_unused_by_logging*/) {
  LOG_DEBUG << "Clustering " << raw_lines.size() << " raw lines with threshold "
            << threshold << std::endl;
  std::vector<double> clustered_values;
  if (raw_lines.empty())
    return clustered_values;

  std::vector<bool> processed(raw_lines.size(), false);
  for (size_t i = 0; i < raw_lines.size(); ++i) {
    if (processed[i])
      continue;
    std::vector<double> current_cluster;
    current_cluster.push_back(raw_lines[i].value);
    processed[i] = true;
    for (size_t j = i + 1; j < raw_lines.size(); ++j) {
      if (!processed[j] &&
          std::abs(raw_lines[j].value - raw_lines[i].value) < threshold) {
        if (false && Logger::getGlobalLogLevel() >= LogLevel::DEBUG) {
          LOG_DEBUG << "Clustering: " << raw_lines[j].value << " and "
                    << raw_lines[i].value << " (diff: "
                    << std::abs(raw_lines[j].value - raw_lines[i].value) << ")"
                    << std::endl;
        }
        current_cluster.push_back(raw_lines[j].value);
        processed[j] = true;
      }
    }
    if (!current_cluster.empty()) {
      clustered_values.push_back(
          std::accumulate(current_cluster.begin(), current_cluster.end(), 0.0) /
          current_cluster.size());
    }
  }
  std::sort(clustered_values.begin(), clustered_values.end());
  LOG_DEBUG << "Found " << clustered_values.size() << " clustered lines."
            << std::endl;
  return clustered_values;
}

std::vector<double>
findUniformGridLinesImproved(const std::vector<double> &values,
                             double dominant_distance, int target_count,
                             double tolerance,
                             bool /*bDebug_param_is_now_unused_by_logging*/) {
  LOG_DEBUG << "Finding uniform grid lines (improved) from " << values.size()
            << " values. Target: " << target_count
            << ", DomDist: " << dominant_distance << ", Tol: " << tolerance
            << std::endl;
  std::vector<double> uniform_lines;

  if (values.size() < 2) {
    LOG_DEBUG << "findUniformGridLinesImproved: Less than 2 values, cannot "
                 "find uniform grid."
              << std::endl;
    return uniform_lines;
  }

  double lower_limit = values.front();
  double upper_limit = values.back();
  std::set<LineMatch> match_data;

  for (double start_value : values) {
    int matched_lines_count = 0;
    double current_fit_score = 0.0;
    std::vector<std::pair<double, double>> current_matched_values;
    std::vector<double> expected_lines;
    expected_lines.push_back(start_value);

    double current_line = start_value;
    while (current_line > lower_limit + tolerance) {
      current_line -= dominant_distance;
      if (current_line >= lower_limit - tolerance) {
        expected_lines.push_back(current_line);
      } else
        break;
    }
    current_line = start_value;
    while (current_line < upper_limit - tolerance) {
      current_line += dominant_distance;
      if (current_line <= upper_limit + tolerance) {
        expected_lines.push_back(current_line);
      } else
        break;
    }
    std::sort(expected_lines.begin(), expected_lines.end());
    expected_lines.erase(
        std::unique(expected_lines.begin(), expected_lines.end()),
        expected_lines.end());

    for (double expected_line : expected_lines) {
      double min_diff = std::numeric_limits<double>::max();
      double closest_value = std::numeric_limits<double>::quiet_NaN();
      for (double clustered_value : values) {
        double diff = std::abs(clustered_value - expected_line);
        if (diff < min_diff) {
          min_diff = diff;
          closest_value = clustered_value;
        }
      }
      current_matched_values.push_back({closest_value, min_diff});
      if (min_diff < tolerance) {
        current_fit_score += min_diff;
        matched_lines_count++;
      }
    }
    match_data.insert({matched_lines_count, current_fit_score, start_value,
                       current_matched_values});
  }

  if (match_data.empty()) {
    LOG_DEBUG << "findUniformGridLinesImproved: No matching data found after "
                 "iterating start_values."
              << std::endl;
    return uniform_lines;
  }

  const LineMatch &best_match = *match_data.begin();
  LOG_DEBUG << "Best match: Count=" << best_match.matched_count
            << ", Score=" << best_match.score
            << ", StartValue=" << best_match.start_value
            << ", MatchedValuesSize=" << best_match.matched_values.size()
            << std::endl;

  std::vector<std::pair<double, double>> sorted_candidates =
      best_match.matched_values;
  std::sort(sorted_candidates.begin(), sorted_candidates.end(),
            [](const auto &a, const auto &b) { return a.second < b.second; });

  for (size_t i = 0; i < sorted_candidates.size() &&
                     static_cast<int>(uniform_lines.size()) < target_count;
       ++i) {
    if (sorted_candidates[i].second < tolerance) {
      uniform_lines.push_back(sorted_candidates[i].first);
    }
  }
  std::sort(uniform_lines.begin(), uniform_lines.end());
  uniform_lines.erase(std::unique(uniform_lines.begin(), uniform_lines.end()),
                      uniform_lines.end());

  if (uniform_lines.size() < static_cast<size_t>(target_count)) {
    LOG_WARN << "Could not find enough uniform grid lines (improved). Found: "
             << uniform_lines.size() << ", Expected: " << target_count
             << std::endl;
  } else {
    LOG_DEBUG << "Found " << uniform_lines.size()
              << " uniform grid lines (improved)." << std::endl;
  }
  return uniform_lines;
}

std::vector<double> findUniformGridLines(
    int target_count,
    bool bDebug_param_for_imshow, // Renamed for clarity for imshow
    int corrected_image_width, int corrected_image_height,
    bool is_generating_horizontal_lines) {
  LOG_INFO << "Finding uniform "
           << (is_generating_horizontal_lines ? "horizontal" : "vertical")
           << " grid lines for image " << corrected_image_width << "x"
           << corrected_image_height << ", target count: " << target_count
           << std::endl;

  if (target_count <= 1) {
    std::string msg = "findUniformGridLines: target_count must be > 1.";
    LOG_ERROR << msg << std::endl;
    THROWGEMERROR(msg);
  }
  if (corrected_image_width <= 0 || corrected_image_height <= 0) {
    std::string msg =
        "findUniformGridLines: corrected_image dimensions must be positive.";
    LOG_ERROR << msg << std::endl;
    THROWGEMERROR(msg);
  }

  std::vector<cv::Point2f> corrected_board_corners =
      getBoardCornersCorrected(corrected_image_width, corrected_image_height);
  if (corrected_board_corners.size() != 4) {
    std::string msg = "findUniformGridLines: getBoardCornersCorrected did not "
                      "return 4 points.";
    LOG_ERROR << msg << std::endl;
    THROWGEMERROR(msg);
  }

  cv::Point2f tl = corrected_board_corners[0];
  cv::Point2f tr = corrected_board_corners[1];
  cv::Point2f bl = corrected_board_corners[3];

  LOG_DEBUG
      << "    Corrected board corners used for generation (TL, TR, BR, BL):"
      << " TL: (" << tl.x << "," << tl.y << "), TR: (" << tr.x << "," << tr.y
      << "), BR: (" << corrected_board_corners[2].x << ","
      << corrected_board_corners[2].y << "), BL: (" << bl.x << "," << bl.y
      << ")" << std::endl;

  std::vector<double> lines;
  lines.reserve(target_count);

  if (is_generating_horizontal_lines) {
    float first_line_y = tl.y;
    float last_line_y = bl.y;
    if (std::abs(last_line_y - first_line_y) < 1.0f) {
      std::string msg =
          "Corrected board height is too small for horizontal line generation.";
      LOG_ERROR << msg << std::endl;
      THROWGEMERROR(msg);
    }
    double spacing =
        static_cast<double>(last_line_y - first_line_y) / (target_count - 1.0);
    LOG_DEBUG << "    Horizontal lines: start_y=" << first_line_y
              << ", end_y=" << last_line_y << ", spacing=" << spacing
              << std::endl;
    for (int i = 0; i < target_count; ++i) {
      lines.push_back(static_cast<double>(first_line_y) + i * spacing);
    }
  } else {
    float first_line_x = tl.x;
    float last_line_x = tr.x;
    if (std::abs(last_line_x - first_line_x) < 1.0f) {
      std::string msg =
          "Corrected board width is too small for vertical line generation.";
      LOG_ERROR << msg << std::endl;
      THROWGEMERROR(msg);
    }
    double spacing =
        static_cast<double>(last_line_x - first_line_x) / (target_count - 1.0);
    LOG_DEBUG << "    Vertical lines: start_x=" << first_line_x
              << ", end_x=" << last_line_x << ", spacing=" << spacing
              << std::endl;
    for (int i = 0; i < target_count; ++i) {
      lines.push_back(static_cast<double>(first_line_x) + i * spacing);
    }
  }

  if (lines.size() != static_cast<size_t>(target_count)) {
    std::string msg =
        "Failed to generate the target number of lines. Expected " +
        std::to_string(target_count) + ", got " + std::to_string(lines.size());
    LOG_ERROR << msg << std::endl;
    THROWGEMERROR(msg);
  }
  LOG_DEBUG << "Generated " << lines.size() << " "
            << (is_generating_horizontal_lines ? "horizontal" : "vertical")
            << " lines." << std::endl;
  return lines;
}

std::pair<std::vector<double>, double> findOptimalClusteringForOrientation(
    const std::vector<Line> &raw_lines, int target_count,
    const std::string &orientation, bool bDebug_param_for_imshow) {
  LOG_DEBUG << "Finding optimal clustering for " << orientation
            << " lines. Raw count: " << raw_lines.size()
            << ", Target: " << target_count << std::endl;
  std::vector<double> clustered_lines;
  double optimal_threshold = 1.0;

  if (raw_lines.empty()) {
    LOG_WARN << "No raw lines provided for optimal clustering (" << orientation
             << ")." << std::endl;
    return std::make_pair(clustered_lines, optimal_threshold);
  }

  std::vector<double> prev_clustered_lines =
      clusterAndAverageLines(raw_lines, 0.1, bDebug_param_for_imshow);
  LOG_DEBUG << "Initial Clustering (" << orientation
            << ") with threshold 0.1: " << prev_clustered_lines.size()
            << " lines." << std::endl;
  clustered_lines = prev_clustered_lines;

  double current_threshold = 1.0;
  double threshold_step = 0.5;
  int max_iterations = 30;

  for (int i = 0; i < max_iterations; ++i) {
    prev_clustered_lines = clustered_lines;
    clustered_lines = clusterAndAverageLines(raw_lines, current_threshold,
                                             bDebug_param_for_imshow);
    LOG_DEBUG << "Clustering Attempt (" << orientation << ") " << i + 1
              << " with threshold " << current_threshold << ": "
              << clustered_lines.size() << " lines." << std::endl;

    if (clustered_lines.size() < static_cast<size_t>(target_count)) {
      LOG_DEBUG << "Clustered line count (" << orientation
                << ") dropped below target (" << target_count
                << "). Returning previous threshold's results ("
                << prev_clustered_lines.size() << " lines)." << std::endl;
      return std::make_pair(prev_clustered_lines, optimal_threshold);
    }
    if (clustered_lines.size() == static_cast<size_t>(target_count)) {
      LOG_DEBUG << "Found target number of clustered lines (" << target_count
                << ") for " << orientation << "." << std::endl;
      return std::make_pair(clustered_lines, current_threshold);
    }
    optimal_threshold = current_threshold;
    current_threshold += threshold_step;
  }
  LOG_WARN << "Max iterations reached for " << orientation
           << " without finding target count or dropping below. Returning last "
              "iteration's results ("
           << clustered_lines.size() << " lines)." << std::endl;
  return std::make_pair(clustered_lines, optimal_threshold);
}

std::pair<std::vector<double>, std::vector<double>>
findOptimalClustering(const std::vector<Line> &horizontal_lines_raw,
                      const std::vector<Line> &vertical_lines_raw,
                      int target_count, bool bDebug_param_for_imshow) {
  LOG_DEBUG << "Finding optimal clustering for both orientations." << std::endl;
  std::pair<std::vector<double>, double> horizontal_result =
      findOptimalClusteringForOrientation(horizontal_lines_raw, target_count,
                                          "horizontal",
                                          bDebug_param_for_imshow);
  std::pair<std::vector<double>, double> vertical_result =
      findOptimalClusteringForOrientation(vertical_lines_raw, target_count,
                                          "vertical", bDebug_param_for_imshow);
  return std::make_pair(horizontal_result.first, vertical_result.first);
}

std::pair<std::vector<double>, std::vector<double>>
detectUniformGrid(const cv::Mat &image) {
  LOG_INFO << "Detecting uniform grid in image of size " << image.cols << "x"
           << image.rows << std::endl;
  if (image.empty()) {
    LOG_ERROR << "Input image is empty in detectUniformGrid." << std::endl;
    THROWGEMERROR("detectUniformGrid: Input image is empty.");
  }

  int corrected_width = image.cols;
  int corrected_height = image.rows;
  const int target_line_count = 19;

  // Using the bDebug global for findUniformGridLines as it was in the old code
  // for its imshow. This could be changed to Logger::getGlobalLogLevel() if
  // desired.
  std::vector<double> final_horizontal_y = findUniformGridLines(
      target_line_count, bDebug, corrected_width, corrected_height, true);
  std::vector<double> final_vertical_x = findUniformGridLines(
      target_line_count, bDebug, corrected_width, corrected_height, false);

  if (final_horizontal_y.size() != static_cast<size_t>(target_line_count) ||
      final_vertical_x.size() != static_cast<size_t>(target_line_count)) {
    std::string error_msg = "detectUniformGrid: findUniformGridLines failed to "
                            "return 19x19 lines. Got H:" +
                            Num2Str(final_horizontal_y.size()).str() +
                            ", V:" + Num2Str(final_vertical_x.size()).str();
    LOG_ERROR << error_msg << std::endl;
    THROWGEMERROR(error_msg);
  }

  if (Logger::getGlobalLogLevel() >= LogLevel::DEBUG) {
    LOG_DEBUG
        << "  detectUniformGrid: Generated uniform horizontal lines (y): ";
    for (double y : final_horizontal_y)
      LOG_DEBUG << y << " ";
    LOG_DEBUG << std::endl;
    LOG_DEBUG << "  detectUniformGrid: Generated uniform vertical lines (x): ";
    for (double x : final_vertical_x)
      LOG_DEBUG << x << " ";
    LOG_DEBUG << std::endl;

    cv::Mat grid_visualization_image = image.clone();
    std::vector<cv::Point2f> dbg_corrected_corners =
        getBoardCornersCorrected(corrected_width, corrected_height);

    for (double y_coord : final_horizontal_y) {
      cv::line(
          grid_visualization_image, cv::Point(0, static_cast<int>(y_coord)),
          cv::Point(grid_visualization_image.cols, static_cast<int>(y_coord)),
          cv::Scalar(0, 255, 0), 1);
    }
    for (double x_coord : final_vertical_x) {
      cv::line(
          grid_visualization_image, cv::Point(static_cast<int>(x_coord), 0),
          cv::Point(static_cast<int>(x_coord), grid_visualization_image.rows),
          cv::Scalar(0, 255, 0), 1);
    }
    if (dbg_corrected_corners.size() == 4) {
      for (size_t i = 0; i < 4; ++i)
        cv::circle(grid_visualization_image, dbg_corrected_corners[i], 5,
                   cv::Scalar(0, 0, 255), -1);
    }
    cv::imshow("Generated Grid on Corrected Image (detectUniformGrid)",
               grid_visualization_image);
    cv::waitKey(0);
    cv::destroyWindow("Generated Grid on Corrected Image (detectUniformGrid)");
  }
  return std::make_pair(final_horizontal_y, final_vertical_x);
}

std::vector<cv::Point2f>
findIntersections(const std::vector<double> &horizontal_lines,
                  const std::vector<double> &vertical_lines) {
  std::vector<cv::Point2f> intersections;
  for (double y : horizontal_lines) {
    for (double x : vertical_lines) {
      intersections.push_back(
          cv::Point2f(static_cast<float>(x), static_cast<float>(y)));
    }
  }
  LOG_DEBUG << "Found " << intersections.size() << " intersections from "
            << horizontal_lines.size() << "H and " << vertical_lines.size()
            << "V lines." << std::endl;
  return intersections;
}

int calculateAdaptiveSampleRadius(float board_pixel_width,
                                  float board_pixel_height) {
  const float factor = 0.35f;
  if (board_pixel_width <= 0 || board_pixel_height <= 0) {
    LOG_WARN << "Invalid board dimensions (" << board_pixel_width << "x"
             << board_pixel_height
             << ") in calculateAdaptiveSampleRadius. Defaulting radius to 3."
             << std::endl;
    return 3;
  }
  float avg_grid_spacing_x = board_pixel_width / 18.0f;
  float avg_grid_spacing_y = board_pixel_height / 18.0f;
  float avg_grid_spacing = (avg_grid_spacing_x + avg_grid_spacing_y) * 0.5f;

  int radius = static_cast<int>(avg_grid_spacing * factor);
  radius = std::max(2, radius);

  LOG_DEBUG << "Calculated adaptive sample radius: " << radius
            << " for board W=" << board_pixel_width
            << ", H=" << board_pixel_height << std::endl;
  return radius;
}

void processGoBoard(const cv::Mat &image_bgr_in, cv::Mat &board_state_out,
                    cv::Mat &board_with_stones_out,
                    std::vector<cv::Point2f> &intersection_points_out) {
  LOG_INFO << "Starting board processing for image " << image_bgr_in.cols << "x"
           << image_bgr_in.rows << std::endl;
  if (image_bgr_in.empty()) {
    LOG_ERROR << "Input BGR image is empty in processGoBoard." << std::endl;
    THROWGEMERROR("Input BGR image is empty in processGoBoard.");
  }

  CalibrationData calib_data = loadCalibrationData(CALIB_CONFIG_PATH);
  if (!calib_data.corners_loaded || !calib_data.colors_loaded ||
      !calib_data.board_color_loaded || !calib_data.dimensions_loaded) {
    std::string err_msg =
        "ProcessGoBoard Error: Incomplete calibration data loaded from " +
        CALIB_CONFIG_PATH + ".";
    LOG_ERROR << err_msg << " Flags - Corners:" << calib_data.corners_loaded
              << " Colors:" << calib_data.colors_loaded
              << " BoardColor:" << calib_data.board_color_loaded
              << " Dims:" << calib_data.dimensions_loaded << std::endl;
    THROWGEMERROR(err_msg);
  }
  LOG_DEBUG << "Full calibration data loaded for processGoBoard." << std::endl;

  cv::Mat image_bgr_corrected = correctPerspective(image_bgr_in);
  if (image_bgr_corrected.empty()) {
    LOG_ERROR << "Corrected perspective image is empty in processGoBoard."
              << std::endl;
    THROWGEMERROR("Corrected perspective image is empty in processGoBoard.");
  }
  if (Logger::getGlobalLogLevel() >= LogLevel::DEBUG) {
    cv::imshow("Corrected Perspective (processGoBoard)", image_bgr_corrected);
    cv::waitKey(1);
  }

  cv::Mat image_lab;
  cv::cvtColor(image_bgr_corrected, image_lab, cv::COLOR_BGR2Lab);
  if (image_lab.empty()) {
    LOG_ERROR << "Lab converted image is empty in processGoBoard." << std::endl;
    THROWGEMERROR("Lab converted image is empty in processGoBoard.");
  }
  if (Logger::getGlobalLogLevel() >= LogLevel::DEBUG) {
    cv::imshow("Lab Image (processGoBoard)", image_lab);
    cv::waitKey(1);
  }

  std::pair<std::vector<double>, std::vector<double>> grid_lines =
      detectUniformGrid(image_bgr_corrected);
  std::vector<double> horizontal_lines = grid_lines.first;
  std::vector<double> vertical_lines = grid_lines.second;

  intersection_points_out = findIntersections(horizontal_lines, vertical_lines);
  int num_intersections = intersection_points_out.size();
  LOG_DEBUG << "Found " << num_intersections << " intersection points."
            << std::endl;
  if (num_intersections != 361 && image_bgr_corrected.cols > 0 &&
      image_bgr_corrected.rows > 0) {
    LOG_WARN << "Expected 361 intersections, found " << num_intersections
             << ". This might indicate grid detection issues." << std::endl;
    if (num_intersections == 0) {
      LOG_ERROR << "No intersection points found in processGoBoard."
                << std::endl;
      THROWGEMERROR("No intersection points found in processGoBoard.");
    }
  } else if (num_intersections == 0) {
    LOG_ERROR << "No intersection points found (image might be invalid) in "
                 "processGoBoard."
              << std::endl;
    THROWGEMERROR("No intersection points found (image might be invalid) in "
                  "processGoBoard.");
  }

  float board_pixel_width_corrected =
      (!vertical_lines.empty())
          ? std::abs(vertical_lines.back() - vertical_lines.front())
          : 0.0f;
  float board_pixel_height_corrected =
      (!horizontal_lines.empty())
          ? std::abs(horizontal_lines.back() - horizontal_lines.front())
          : 0.0f;
  int adaptive_sample_radius = calculateAdaptiveSampleRadius(
      board_pixel_width_corrected, board_pixel_height_corrected);
  LOG_DEBUG << "processGoBoard using adaptive_sample_radius: "
            << adaptive_sample_radius << std::endl;

  std::vector<cv::Vec3f> average_lab_values(num_intersections);
  for (int i = 0; i < num_intersections; ++i) {
    average_lab_values[i] = getAverageLab(image_lab, intersection_points_out[i],
                                          adaptive_sample_radius);
  }
  LOG_DEBUG << "Sampled Lab colors for all intersections." << std::endl;

  classifyIntersectionsByCalibration(
      average_lab_values, calib_data, intersection_points_out,
      num_intersections, image_bgr_corrected, adaptive_sample_radius,
      board_state_out, board_with_stones_out);

  if (Logger::getGlobalLogLevel() >= LogLevel::DEBUG) {
    cv::imshow("processGoBoard - Final Stones Output", board_with_stones_out);
    cv::waitKey(0);
    // It's good practice to destroy specific windows if they are no longer
    // needed or if subsequent debug views might reuse names.
    cv::destroyWindow("Corrected Perspective (processGoBoard)");
    cv::destroyWindow("Lab Image (processGoBoard)");
    // "Direct Classification Result (Helper)" is handled by
    // classifyIntersectionsByCalibration
    cv::destroyWindow("processGoBoard - Final Stones Output");
  }
  LOG_INFO << "Board processing finished." << std::endl;
}

bool detectSpecificColoredRoundShape(const cv::Mat &inputBgrImage,
                                     const cv::Rect &regionOfInterest,
                                     const cv::Vec3f &expectedAvgLabColor,
                                     float l_tolerance, float ab_tolerance,
                                     float expectedPixelRadius,
                                     cv::Point2f &detectedCenter,
                                     float &detectedRadius) {

  LOG_DEBUG << "Detecting specific colored round shape in ROI: {x:"
            << regionOfInterest.x << ",y:" << regionOfInterest.y
            << ",w:" << regionOfInterest.width
            << ",h:" << regionOfInterest.height
            << "} Target Lab: " << expectedAvgLabColor[0] << ","
            << expectedAvgLabColor[1] << "," << expectedAvgLabColor[2]
            << " L_tol: " << l_tolerance << ", AB_tol: " << ab_tolerance
            << " expectedPixelRadius: " << expectedPixelRadius << std::endl;
  if (expectedPixelRadius <= 0) {
    LOG_ERROR << "  detectSpecificColoredRoundShape: expectedPixelRadius must "
                 "be positive. Got: "
              << expectedPixelRadius << std::endl;
    return false; // Or handle with a default if appropriate, but for now,
                  // require it.
  }
  double expectedStoneArea = CV_PI * expectedPixelRadius * expectedPixelRadius;
  double minAbsStoneArea = expectedStoneArea * ABS_STONE_AREA_MIN_FACTOR;
  double maxAbsStoneArea = expectedStoneArea * ABS_STONE_AREA_MAX_FACTOR;
  double minCircularity = (expectedAvgLabColor[0] < 100.0f)
                              ? MIN_STONE_CIRCULARITY_BLACK
                              : MIN_STONE_CIRCULARITY_WHITE;
  LOG_DEBUG << "  For ROI at (" << regionOfInterest.x << ","
            << regionOfInterest.y << "): ExpectedRadius=" << expectedPixelRadius
            << ", ExpectedArea=" << expectedStoneArea
            << ", MinAbsArea=" << minAbsStoneArea
            << ", MaxAbsArea=" << maxAbsStoneArea
            << ", MinCircularity=" << minCircularity << std::endl;
  if (inputBgrImage.empty()) {
    LOG_ERROR << "Input BGR image is empty in detectSpecificColoredRoundShape."
              << std::endl;
    return false;
  }
  cv::Rect roi =
      regionOfInterest & cv::Rect(0, 0, inputBgrImage.cols, inputBgrImage.rows);
  if (roi.width <= 0 || roi.height <= 0) {
    LOG_WARN
        << "Region of interest is invalid or outside image bounds. ROI: {x:"
        << regionOfInterest.x << ",y:" << regionOfInterest.y
        << ",w:" << regionOfInterest.width << ",h:" << regionOfInterest.height
        << "}, Image: " << inputBgrImage.cols << "x" << inputBgrImage.rows
        << std::endl;
    return false;
  }

  cv::Mat roiImageBgr = inputBgrImage(roi);
  cv::Mat roiImageLab;
  cv::cvtColor(roiImageBgr, roiImageLab, cv::COLOR_BGR2Lab);

  cv::Mat colorMask;
  cv::Scalar labLower(std::max(0.f, expectedAvgLabColor[0] - l_tolerance),
                      std::max(0.f, expectedAvgLabColor[1] - ab_tolerance),
                      std::max(0.f, expectedAvgLabColor[2] - ab_tolerance));
  cv::Scalar labUpper(std::min(255.f, expectedAvgLabColor[0] + l_tolerance),
                      std::min(255.f, expectedAvgLabColor[1] + ab_tolerance),
                      std::min(255.f, expectedAvgLabColor[2] + ab_tolerance));

  LOG_DEBUG << "  For ROI at (" << roi.x << "," << roi.y << "): Target Lab: ["
            << expectedAvgLabColor[0] << "," << expectedAvgLabColor[1] << ","
            << expectedAvgLabColor[2] << "]" << ", L_tol: " << l_tolerance
            << ", AB_tol: " << ab_tolerance << "    Calculated Lab Lower: ["
            << labLower[0] << "," << labLower[1] << "," << labLower[2] << "]"
            << ", Lab Upper: [" << labUpper[0] << "," << labUpper[1] << ","
            << labUpper[2] << "]" << std::endl;

  cv::inRange(roiImageLab, labLower, labUpper, colorMask);

  // <<< MODIFIED: Use new constants for morphology >>>
  cv::Mat open_kernel = cv::getStructuringElement(
      cv::MORPH_ELLIPSE,
      cv::Size(MORPH_OPEN_KERNEL_SIZE_STONE, MORPH_OPEN_KERNEL_SIZE_STONE));
  cv::Mat close_kernel = cv::getStructuringElement(
      cv::MORPH_ELLIPSE,
      cv::Size(MORPH_CLOSE_KERNEL_SIZE_STONE, MORPH_CLOSE_KERNEL_SIZE_STONE));

  cv::morphologyEx(colorMask, colorMask, cv::MORPH_OPEN, open_kernel,
                   cv::Point(-1, -1), MORPH_OPEN_ITERATIONS_STONE);
  cv::morphologyEx(colorMask, colorMask, cv::MORPH_CLOSE, close_kernel,
                   cv::Point(-1, -1), MORPH_CLOSE_ITERATIONS_STONE);

  if (bDebug) {
    std::string roi_window_name =
        "Color Mask ROI (L:" +
        std::to_string(static_cast<int>(expectedAvgLabColor[0])) + " X" +
        std::to_string(roi.x) + " Y" + std::to_string(roi.y) + ")";
    cv::imshow(roi_window_name, colorMask);
    cv::waitKey(0);
    cv::destroyWindow(roi_window_name);
  }

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(colorMask, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  LOG_DEBUG << "  For ROI at (" << roi.x << "," << roi.y << "): Found "
            << contours.size() << " raw contours in ROI (" << roi.x << ","
            << roi.y << " " << roi.width << "x" << roi.height << ")."
            << std::endl;

  if (contours.empty()) {
    LOG_DEBUG << "  For ROI at (" << roi.x << "," << roi.y
              << "): No contours found after color masking." << std::endl;
    return false;
  }

  LOG_DEBUG << " MinCircularity=" << minCircularity << std::endl;

  std::vector<cv::Point> bestContour;
  double bestContourScore = 0.0f;
  int contour_idx = 0;
  cv::Mat roi_contour_vis_canvas;
  if (bDebug) {
    roi_contour_vis_canvas = roiImageBgr.clone();
  }

  for (const auto &contour : contours) {
    contour_idx++;
    if (contour.size() < MIN_CONTOUR_POINTS_STONE) {
      LOG_DEBUG << "    Contour " << contour_idx << " (ROI " << roi.x << ","
                << roi.y << ") rejected: too few points (" << contour.size()
                << ")." << std::endl;
      continue;
    }
    double area = cv::contourArea(contour);
    double perimeter = cv::arcLength(contour, true);
    if (perimeter < 1.0) {
      LOG_DEBUG << "    Contour " << contour_idx << " (ROI " << roi.x << ","
                << roi.y << ") rejected: perimeter too small (" << perimeter
                << ")." << std::endl;
      continue;
    }
    double circularity = (4 * CV_PI * area) / (perimeter * perimeter);

    LOG_DEBUG << "    Contour " << contour_idx << " (ROI " << roi.x << ","
              << roi.y << ") evaluating: Area=" << area
              << ", Circ=" << circularity << std::endl;

    if (area < minAbsStoneArea || area > maxAbsStoneArea) {
      LOG_DEBUG << "      -> Contour " << contour_idx << " (ROI " << roi.x
                << "," << roi.y << ") REJECTED by area " << area
                << (area < minAbsStoneArea
                        ? " (too small, min=" + Num2Str(minAbsStoneArea).str() +
                              ")"
                        : " (too large, max=" + Num2Str(maxAbsStoneArea).str() +
                              ")")
                << ", Expected Area: " << Num2Str(expectedStoneArea).str()
                << std::endl;
      if (bDebug)
        cv::drawContours(roi_contour_vis_canvas,
                         std::vector<std::vector<cv::Point>>{contour}, -1,
                         cv::Scalar(0, 165, 255), 1);
      continue;
    }
    if (circularity < minCircularity) {
      LOG_DEBUG << "      -> Contour " << contour_idx << " (ROI " << roi.x
                << "," << roi.y << ") REJECTED by circularity " << circularity
                << " (min_circ=" << minCircularity << ")" << std::endl;
      if (bDebug) {
        cv::drawContours(roi_contour_vis_canvas,
                         std::vector<std::vector<cv::Point>>{contour}, -1,
                         cv::Scalar(255, 0, 0), 1);
        cv::putText(roi_contour_vis_canvas,
                    "C:" + std::to_string(circularity).substr(0, 4),
                    contour[0] - cv::Point(0, 5), cv::FONT_HERSHEY_SIMPLEX, 0.3,
                    cv::Scalar(255, 100, 100));
      }
      continue;
    }
    LOG_DEBUG << "      => Contour " << contour_idx << " (ROI " << roi.x << ","
              << roi.y
              << ") PASSED filters. Current best area: " << bestContourScore
              << std::endl;
    if (bDebug)
      cv::drawContours(roi_contour_vis_canvas,
                       std::vector<std::vector<cv::Point>>{contour}, -1,
                       cv::Scalar(0, 255, 255), 1);

    if (area > bestContourScore) {
      bestContourScore = area;
      bestContour = contour;
      LOG_DEBUG << "        ==> New best contour " << contour_idx << " (ROI "
                << roi.x << "," << roi.y << ") with area " << area << std::endl;
    }
  }

  if (bDebug && !contours.empty()) {
    std::string roi_contours_win_name = "Evaluated Contours (ROI X" +
                                        std::to_string(roi.x) + " Y" +
                                        std::to_string(roi.y) + ")";
    if (!bestContour.empty()) {
      cv::drawContours(roi_contour_vis_canvas,
                       std::vector<std::vector<cv::Point>>{bestContour}, -1,
                       cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow(roi_contours_win_name, roi_contour_vis_canvas);
    cv::waitKey(0);
    cv::destroyWindow(roi_contours_win_name);
  }

  if (!bestContour.empty()) {
    cv::Point2f centerInRoi;
    float radiusInRoi;
    cv::minEnclosingCircle(bestContour, centerInRoi, radiusInRoi);
    detectedCenter.x = centerInRoi.x + roi.x;
    detectedCenter.y = centerInRoi.y + roi.y;
    detectedRadius = radiusInRoi;
    LOG_DEBUG << "  For ROI at (" << roi.x << "," << roi.y
              << "): Found specific stone. Center (orig img): "
              << detectedCenter << ", Radius: " << detectedRadius << std::endl;
    return true;
  }
  LOG_DEBUG
      << "  For ROI at (" << roi.x << "," << roi.y
      << "): No suitable specific stone contour found in ROI after filtering."
      << std::endl;
  return false;
}

bool find_largest_color_blob_in_roi(
    const cv::Mat &image_to_search_bgr, const cv::Rect &roi_in_image,
    const cv::Vec3f &target_lab_color, float l_tol, float ab_tol,
    // Outputs:
    cv::Point2f
        &out_blob_center_in_image_coords, // Relative to image_to_search_bgr
    double &out_blob_area) {

  if (image_to_search_bgr.empty())
    return false;
  cv::Rect valid_roi = roi_in_image & cv::Rect(0, 0, image_to_search_bgr.cols,
                                               image_to_search_bgr.rows);
  if (valid_roi.width <= 0 || valid_roi.height <= 0)
    return false;

  cv::Mat roi_bgr = image_to_search_bgr(valid_roi);
  cv::Mat roi_lab;
  cv::cvtColor(roi_bgr, roi_lab, cv::COLOR_BGR2Lab);

  cv::Mat color_mask;
  cv::Scalar lab_lower(std::max(0.f, target_lab_color[0] - l_tol),
                       std::max(0.f, target_lab_color[1] - ab_tol),
                       std::max(0.f, target_lab_color[2] - ab_tol));
  cv::Scalar lab_upper(std::min(255.f, target_lab_color[0] + l_tol),
                       std::min(255.f, target_lab_color[1] + ab_tol),
                       std::min(255.f, target_lab_color[2] + ab_tol));
  cv::inRange(roi_lab, lab_lower, lab_upper, color_mask);

  cv::Mat open_kernel = cv::getStructuringElement(
      cv::MORPH_ELLIPSE,
      cv::Size(MORPH_OPEN_KERNEL_SIZE_STONE, MORPH_OPEN_KERNEL_SIZE_STONE));
  cv::Mat close_kernel = cv::getStructuringElement(
      cv::MORPH_ELLIPSE,
      cv::Size(MORPH_CLOSE_KERNEL_SIZE_STONE, MORPH_CLOSE_KERNEL_SIZE_STONE));
  cv::morphologyEx(color_mask, color_mask, cv::MORPH_OPEN, open_kernel,
                   cv::Point(-1, -1), MORPH_OPEN_ITERATIONS_STONE);
  cv::morphologyEx(color_mask, color_mask, cv::MORPH_CLOSE, close_kernel,
                   cv::Point(-1, -1), MORPH_CLOSE_ITERATIONS_STONE);

  // Debug: show this mask if needed
  if (Logger::getGlobalLogLevel() >= LogLevel::DEBUG &&
      bDebug) { // Only if bDebug is also true for this specific helper
    cv::imshow("Helper Blob Mask (ROI " + std::to_string(valid_roi.x) + "," +
                   std::to_string(valid_roi.y) + ")",
               color_mask);
    cv::waitKey(1); // Allow some processing
  }

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(color_mask, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  if (contours.empty())
    return false;

  double max_area = 0.0;
  const std::vector<cv::Point> *largest_contour = nullptr;

  for (const auto &contour : contours) {
    double area = cv::contourArea(contour);
    if (area > max_area) {
      max_area = area;
      largest_contour = &contour;
    }
  }

  if (largest_contour && max_area > 0) {
    cv::Moments M = cv::moments(*largest_contour);
    if (M.m00 > 0) { // Avoid division by zero
      out_blob_center_in_image_coords.x =
          static_cast<float>(M.m10 / M.m00) +
          valid_roi.x; // Adjust to full image coords
      out_blob_center_in_image_coords.y =
          static_cast<float>(M.m01 / M.m00) +
          valid_roi.y; // Adjust to full image coords
      out_blob_area = max_area;
      return true;
    }
  }
  return false;
}
// --- Replaces experimental_scan_for_quadrant_stone ---
// --- Phase 1 Utility: findSingleCornerStone_Refined ---
bool findSingleCornerStone_Refined(
    const cv::Mat &rawBgrImage, CornerQuadrant targetQuadrant,
    const CalibrationData &calibData, // For Lab color references
    const std::vector<cv::Point2f>
        &known_raw_corners, // Input: current known raw corners (TL,TR,BR,BL
                            // order)
    // Outputs:
    cv::Point2f &out_found_raw_corner, // The refined raw board corner position
                                       // for targetQuadrant
    cv::Mat &out_debug_final_corrected_image, // The Pass 2 corrected image
    cv::Point2f &out_debug_stone_center,      // Stone center in
                                              // out_debug_final_corrected_image
    float &out_debug_stone_radius) {          // Stone radius in
                                              // out_debug_final_corrected_image

  LOG_INFO << "--- findSingleCornerStone_Refined for "
           << (targetQuadrant == CornerQuadrant::TOP_LEFT
                   ? "TOP_LEFT"
                   : (targetQuadrant == CornerQuadrant::TOP_RIGHT
                          ? "TOP_RIGHT"
                          : (targetQuadrant == CornerQuadrant::BOTTOM_LEFT
                                 ? "BOTTOM_LEFT"
                                 : "BOTTOM_RIGHT")))
           << " ---";

  std::string quadrant_name_str;
  cv::Vec3f target_lab_color;
  size_t target_ideal_dest_corner_idx = 0; // For ideal_corrected_dest_points
  size_t target_raw_corner_idx =
      0; // For known_raw_corners and p1_source_points_raw

  out_debug_final_corrected_image = cv::Mat();
  out_found_raw_corner = cv::Point2f(-1, -1);

  bool use_fallback_color = !calibData.colors_loaded;

  switch (targetQuadrant) {
  case CornerQuadrant::TOP_LEFT:
    quadrant_name_str = "TOP_LEFT";
    target_ideal_dest_corner_idx = 0;
    target_raw_corner_idx = 0;
    if (calibData.colors_loaded && calibData.lab_tl[0] >= 0)
      target_lab_color = calibData.lab_tl;
    else {
      use_fallback_color = true;
      target_lab_color = cv::Vec3f(50, 128, 128);
      LOG_WARN << "RefinedFind: No/Invalid TL color, using fallback Black.";
    }
    break;
  case CornerQuadrant::TOP_RIGHT:
    quadrant_name_str = "TOP_RIGHT";
    target_ideal_dest_corner_idx = 1;
    target_raw_corner_idx = 1;
    if (calibData.colors_loaded && calibData.lab_tr[0] >= 0)
      target_lab_color = calibData.lab_tr;
    else {
      use_fallback_color = true;
      target_lab_color = cv::Vec3f(220, 128, 128);
      LOG_WARN << "RefinedFind: No/Invalid TR color, using fallback White.";
    }
    break;
  case CornerQuadrant::BOTTOM_RIGHT: // Corresponds to
                                     // ideal_corrected_dest_points[2]
    quadrant_name_str = "BOTTOM_RIGHT";
    target_ideal_dest_corner_idx = 2;
    target_raw_corner_idx = 2;
    if (calibData.colors_loaded && calibData.lab_br[0] >= 0)
      target_lab_color = calibData.lab_br;
    else {
      use_fallback_color = true;
      target_lab_color = cv::Vec3f(220, 128, 128);
      LOG_WARN << "RefinedFind: No/Invalid BR color, using fallback White.";
    }
    break;
  case CornerQuadrant::BOTTOM_LEFT: // Corresponds to
                                    // ideal_corrected_dest_points[3]
    quadrant_name_str = "BOTTOM_LEFT";
    target_ideal_dest_corner_idx = 3;
    target_raw_corner_idx = 3;
    if (calibData.colors_loaded && calibData.lab_bl[0] >= 0)
      target_lab_color = calibData.lab_bl;
    else {
      use_fallback_color = true;
      target_lab_color = cv::Vec3f(50, 128, 128);
      LOG_WARN << "RefinedFind: No/Invalid BL color, using fallback Black.";
    }
    break;
  default:
    LOG_ERROR << "RefinedFind: Invalid target quadrant.";
    return false;
  }
  LOG_DEBUG << "RefinedFind: Using Lab ref " << target_lab_color << " for "
            << quadrant_name_str;
  if (rawBgrImage.empty()) {
    LOG_ERROR << "RefinedFind: Raw BGR image empty.";
    return false;
  }

  std::vector<cv::Point2f> ideal_corrected_dest_points =
      getBoardCornersCorrected(rawBgrImage.cols, rawBgrImage.rows);

  // === PASS 1: Initial Rough Correction and Blob Finding ===
  LOG_INFO << "RefinedFind Pass 1 for " << quadrant_name_str;
  cv::Point2f p1_raw_corner_initial_guess;
  // Initial guess for the raw corner based on quadrant center
  if (targetQuadrant == CornerQuadrant::TOP_LEFT)
    p1_raw_corner_initial_guess =
        cv::Point2f(rawBgrImage.cols * 0.25f, rawBgrImage.rows * 0.25f);
  else if (targetQuadrant == CornerQuadrant::TOP_RIGHT)
    p1_raw_corner_initial_guess =
        cv::Point2f(rawBgrImage.cols * 0.75f, rawBgrImage.rows * 0.25f);
  else if (targetQuadrant == CornerQuadrant::BOTTOM_RIGHT)
    p1_raw_corner_initial_guess =
        cv::Point2f(rawBgrImage.cols * 0.75f, rawBgrImage.rows * 0.75f);
  else if (targetQuadrant == CornerQuadrant::BOTTOM_LEFT)
    p1_raw_corner_initial_guess =
        cv::Point2f(rawBgrImage.cols * 0.25f, rawBgrImage.rows * 0.75f);
  LOG_DEBUG << "RefinedFind P1: Initial raw guess for " << quadrant_name_str
            << " corner: " << p1_raw_corner_initial_guess;

  std::vector<cv::Point2f> p1_source_points_raw(4);
  float p1_est_board_span_x = static_cast<float>(rawBgrImage.cols) * 0.75f;
  float p1_est_board_span_y = static_cast<float>(rawBgrImage.rows) * 0.75f;

  // Construct p1_source_points_raw using p1_raw_corner_initial_guess as the
  // anchor for targetQuadrant Use known_raw_corners if available for *other*
  // points
  p1_source_points_raw[target_raw_corner_idx] = p1_raw_corner_initial_guess;
  for (size_t i = 0; i < 4; ++i) {
    if (i == target_raw_corner_idx)
      continue;
    if (known_raw_corners.size() == 4 && known_raw_corners[i].x >= 0 &&
        known_raw_corners[i].y >= 0) {
      p1_source_points_raw[i] = known_raw_corners[i];
      LOG_DEBUG << "RefinedFind P1: Using known raw corner for point " << i
                << ": " << known_raw_corners[i];
    } else { // Fallback to heuristic if not known
      // Simple heuristic relative to p1_raw_corner_initial_guess (needs to be
      // smarter if target is not TL) This heuristic is very basic if target is
      // not TL. Example: if target is TR, p1_source_points_raw[0] (TL) would be
      // (guess.x - span, guess.y)
      if (i == 0)
        p1_source_points_raw[i] = cv::Point2f(
            p1_raw_corner_initial_guess.x -
                (target_raw_corner_idx == 1 || target_raw_corner_idx == 2
                     ? p1_est_board_span_x
                     : 0),
            p1_raw_corner_initial_guess.y -
                (target_raw_corner_idx == 3 || target_raw_corner_idx == 2
                     ? p1_est_board_span_y
                     : 0));
      if (i == 1)
        p1_source_points_raw[i] = cv::Point2f(
            p1_raw_corner_initial_guess.x +
                (target_raw_corner_idx == 0 || target_raw_corner_idx == 3
                     ? p1_est_board_span_x
                     : 0),
            p1_raw_corner_initial_guess.y -
                (target_raw_corner_idx == 2 || target_raw_corner_idx == 3
                     ? p1_est_board_span_y
                     : 0));
      // This heuristic for other corners needs robust definition based on
      // target_raw_corner_idx for general case For now, let's simplify assuming
      // TL is found first, then TR etc. For TL target, other corners are
      // relative:
      if (targetQuadrant == CornerQuadrant::TOP_LEFT) {
        if (i == 1)
          p1_source_points_raw[1] =
              cv::Point2f(p1_raw_corner_initial_guess.x + p1_est_board_span_x,
                          p1_raw_corner_initial_guess.y); // TR
        if (i == 3)
          p1_source_points_raw[3] = cv::Point2f(p1_raw_corner_initial_guess.x,
                                                p1_raw_corner_initial_guess.y +
                                                    p1_est_board_span_y); // BL
        if (i == 2)
          p1_source_points_raw[2] = cv::Point2f(
              p1_raw_corner_initial_guess.x + p1_est_board_span_x,
              p1_raw_corner_initial_guess.y + p1_est_board_span_y); // BR
      } else { // Basic fallback if not TL - this needs improvement for general
               // case
        if (i != target_raw_corner_idx)
          p1_source_points_raw[i] =
              ideal_corrected_dest_points[i]; // Map ideal corrected back to raw
                                              // using an identity as
                                              // placeholder
      }
    }
  }

  for (cv::Point2f &pt : p1_source_points_raw) { /* Clamp */
  }
  if (cv::contourArea(p1_source_points_raw) <
      (p1_est_board_span_x * p1_est_board_span_y * 0.01)) {
    LOG_ERROR << "RefinedFind P1: Degenerate quad for P1.";
    return false;
  }

  cv::Mat M1 = cv::getPerspectiveTransform(p1_source_points_raw,
                                           ideal_corrected_dest_points);
  if (M1.empty() || cv::determinant(M1) < 1e-6) {
    LOG_ERROR << "RefinedFind P1: Degenerate M1.";
    return false;
  }

  cv::Mat image_pass1_corrected;
  cv::warpPerspective(rawBgrImage, image_pass1_corrected, M1,
                      rawBgrImage.size());
  if (image_pass1_corrected.empty()) {
    LOG_ERROR << "RefinedFind P1: Warped P1 empty.";
    return false;
  }

  out_debug_final_corrected_image = image_pass1_corrected.clone();
  out_found_raw_corner = p1_raw_corner_initial_guess; // Placeholder

  // Pass 1 Blob Search ROI: Use calculateGridIntersectionROI for the target
  // corner
  int ideal_grid_col = (targetQuadrant == CornerQuadrant::TOP_LEFT ||
                        targetQuadrant == CornerQuadrant::BOTTOM_LEFT)
                           ? 0
                           : 18;
  int ideal_grid_row = (targetQuadrant == CornerQuadrant::TOP_LEFT ||
                        targetQuadrant == CornerQuadrant::TOP_RIGHT)
                           ? 0
                           : 18;
  cv::Rect roi_pass1_targeted = calculateGridIntersectionROI(
      ideal_grid_col, ideal_grid_row, image_pass1_corrected.cols,
      image_pass1_corrected.rows);
  LOG_DEBUG << "RefinedFind P1: Targeted ROI in P1_corrected: "
            << roi_pass1_targeted;

  cv::Point2f p1_blob_center_in_pass1_corrected;
  double p1_blob_area = 0.0;
  bool blob_found_p1 = find_largest_color_blob_in_roi(
      image_pass1_corrected, roi_pass1_targeted, target_lab_color,
      CALIB_L_TOLERANCE_STONE, CALIB_AB_TOLERANCE_STONE,
      p1_blob_center_in_pass1_corrected, p1_blob_area);

  if (bDebug && !image_pass1_corrected.empty()) { /* Visualization as before */
  }

  if (!blob_found_p1) {
    LOG_WARN << "RefinedFind P1: No blob in targeted ROI for "
             << quadrant_name_str;
    return false;
  }
  LOG_INFO << "RefinedFind P1: Largest blob. Area: " << p1_blob_area
           << ", Center in P1_corr: " << p1_blob_center_in_pass1_corrected;

  // === PASS 2: Refined Perspective Transformation and Focused Verification ===
  LOG_INFO << "RefinedFind Pass 2 for " << quadrant_name_str;

  cv::Mat M1_inv;
  if (!cv::invert(M1, M1_inv, cv::DECOMP_SVD) || M1_inv.empty()) {
    LOG_ERROR << "RefinedFind P2: Invert M1 failed.";
    return false;
  }

  std::vector<cv::Point2f> p1_blob_center_vec_tf = {
      p1_blob_center_in_pass1_corrected};
  std::vector<cv::Point2f> p1_blob_center_raw_vec_tf;
  cv::perspectiveTransform(p1_blob_center_vec_tf, p1_blob_center_raw_vec_tf,
                           M1_inv);
  if (p1_blob_center_raw_vec_tf.empty()) {
    LOG_ERROR << "RefinedFind P2: Map blob to raw failed.";
    return false;
  }
  cv::Point2f p1_blob_center_in_raw_image = p1_blob_center_raw_vec_tf[0];
  LOG_DEBUG << "RefinedFind P2: P1 blob mapped to raw: "
            << p1_blob_center_in_raw_image;

  if (bDebug) { /* Visualization of p1_blob_center_in_raw_image on rawBgrImage
                 */
  }

  float est_radius_p1_corr =
      calculateAdaptiveSampleRadius(image_pass1_corrected.cols,
                                    image_pass1_corrected.rows) *
      (0.47f / 0.35f); // Estimate actual stone radius
  float est_raw_radius_val_offset =
      est_radius_p1_corr; // Simple initial heuristic for offset
  if (est_radius_p1_corr >
      1.0f) { /* ... est_raw_radius_val_offset calculation as before ... */
  }

  cv::Point2f p2_raw_corner_guess;
  // Offset blob center to get estimated corner location
  if (targetQuadrant == CornerQuadrant::TOP_LEFT)
    p2_raw_corner_guess =
        p1_blob_center_in_raw_image -
        cv::Point2f(est_raw_radius_val_offset, est_raw_radius_val_offset);
  else if (targetQuadrant == CornerQuadrant::TOP_RIGHT)
    p2_raw_corner_guess =
        p1_blob_center_in_raw_image +
        cv::Point2f(est_raw_radius_val_offset, -est_raw_radius_val_offset);
  else if (targetQuadrant == CornerQuadrant::BOTTOM_LEFT)
    p2_raw_corner_guess =
        p1_blob_center_in_raw_image +
        cv::Point2f(-est_raw_radius_val_offset, est_raw_radius_val_offset);
  else if (targetQuadrant == CornerQuadrant::BOTTOM_RIGHT)
    p2_raw_corner_guess =
        p1_blob_center_in_raw_image +
        cv::Point2f(est_raw_radius_val_offset, est_raw_radius_val_offset);
  else {
    return false;
  }

  // Clamp p2_raw_corner_guess
  p2_raw_corner_guess.x =
      std::max(0.0f, std::min(static_cast<float>(rawBgrImage.cols - 1),
                              p2_raw_corner_guess.x));
  p2_raw_corner_guess.y =
      std::max(0.0f, std::min(static_cast<float>(rawBgrImage.rows - 1),
                              p2_raw_corner_guess.y));

  LOG_DEBUG << "RefinedFind P2: Refined raw corner guess for M2: "
            << p2_raw_corner_guess;

  std::vector<cv::Point2f> p2_source_points_raw(4);
  p2_source_points_raw[target_raw_corner_idx] = p2_raw_corner_guess;
  // Fill other 3 points for p2_source_points_raw using known_raw_corners if
  // valid, else heuristic from p2_raw_corner_guess
  for (size_t i = 0; i < 4; ++i) {
    if (i == target_raw_corner_idx)
      continue;
    if (known_raw_corners.size() == 4 && known_raw_corners[i].x >= 0 &&
        known_raw_corners[i].y >= 0) {
      p2_source_points_raw[i] = known_raw_corners[i];
    } else { // Fallback heuristic for unknown corners in P2
      if (targetQuadrant ==
          CornerQuadrant::TOP_LEFT) { // Example if TL is target
        if (i == 1)
          p2_source_points_raw[1] =
              cv::Point2f(p2_raw_corner_guess.x + p1_est_board_span_x,
                          p2_raw_corner_guess.y);
        if (i == 3)
          p2_source_points_raw[3] =
              cv::Point2f(p2_raw_corner_guess.x,
                          p2_raw_corner_guess.y + p1_est_board_span_y);
        if (i == 2)
          p2_source_points_raw[2] =
              cv::Point2f(p2_raw_corner_guess.x + p1_est_board_span_x,
                          p2_raw_corner_guess.y + p1_est_board_span_y);
      } // This heuristic needs to be carefully generalized
    }
  }
  for (cv::Point2f &pt : p2_source_points_raw) { /* Clamp */
  }
  if (cv::contourArea(p2_source_points_raw) <
      (p1_est_board_span_x * p1_est_board_span_y * 0.01)) {
    LOG_ERROR << "RefinedFind P2: Degenerate quad for M2.";
    return false;
  }

  cv::Mat M2 = cv::getPerspectiveTransform(p2_source_points_raw,
                                           ideal_corrected_dest_points);
  if (M2.empty() || cv::determinant(M2) < 1e-6) {
    LOG_ERROR << "RefinedFind P2: Degenerate M2.";
    return false;
  }

  cv::Mat image_pass2_corrected;
  cv::warpPerspective(rawBgrImage, image_pass2_corrected, M2,
                      rawBgrImage.size());
  if (image_pass2_corrected.empty()) {
    LOG_ERROR << "RefinedFind P2: Warped P2 empty.";
    return false;
  }

  float expected_radius_p2_corr =
      calculateAdaptiveSampleRadius(image_pass2_corrected.cols * 0.8f,
                                    image_pass2_corrected.rows * 0.8f) *
      (0.47f / 0.35f);
  if (expected_radius_p2_corr < 1.0f) { /* ... */
    return false;
  }

  // Focused ROI for Pass 2 validation: Centered on the ideal grid point for
  // targetQuadrant in image_pass2_corrected
  cv::Rect focused_roi_pass2 = calculateGridIntersectionROI(
      ideal_grid_col, ideal_grid_row, image_pass2_corrected.cols,
      image_pass2_corrected.rows);

  LOG_DEBUG << "RefinedFind P2: Focused ROI in P2_corrected: "
            << focused_roi_pass2 << ", ExpR: " << expected_radius_p2_corr;

  if (bDebug) { /* Visualization of image_pass2_corrected and focused_roi_pass2
                 */
  }

  bool final_stone_found = detectSpecificColoredRoundShape(
      image_pass2_corrected, focused_roi_pass2, target_lab_color,
      CALIB_L_TOLERANCE_STONE, CALIB_AB_TOLERANCE_STONE,
      expected_radius_p2_corr, out_debug_stone_center, out_debug_stone_radius);

  if (bDebug) { /* Final visualization of detection result on
                   image_pass2_corrected */
  }

  if (final_stone_found) {
    LOG_INFO << "RefinedFind P2 SUCCESS: Stone validated for "
             << quadrant_name_str;
    return true;
  } else {
    LOG_WARN << "RefinedFind P2 FAILED: Stone NOT validated for "
             << quadrant_name_str;
    return false;
  }
}
bool detectFourCornersGoBoard(
    const cv::Mat &rawBgrImage,
    std::vector<cv::Point2f> &out_detected_raw_board_corners_tl_tr_br_bl) {

  LOG_INFO << "--- detectFourCornersGoBoard (using adapted "
              "experimental_scan_for_quadrant_stone logic) ---";
  if (rawBgrImage.empty()) {
    LOG_ERROR << "detectFourCornersGoBoard: Input rawBgrImage is empty.";
    return false;
  }

  out_detected_raw_board_corners_tl_tr_br_bl.assign(4,
                                                    cv::Point2f(-1.0f, -1.0f));

  CalibrationData calibData = loadCalibrationData(CALIB_CONFIG_PATH);
  // adaptive_detect_stone now handles fallbacks if calibData.colors_loaded is
  // false or specific colors are invalid. A general warning if colors_loaded is
  // false can still be useful here.
  if (!calibData.colors_loaded) {
    LOG_WARN << "detectFourCornersGoBoard: Calibration color data not fully "
                "loaded from "
             << CALIB_CONFIG_PATH
             << ". adaptive_detect_stone will use internal fallbacks for stone "
                "colors.";
  }

  cv::Mat temp_corrected_image; // Not used by caller, but adaptive_detect_stone
                                // populates it.
  float temp_detected_radius;   // Used by caller.

  CornerQuadrant quadrants_to_scan[] = {
      CornerQuadrant::TOP_LEFT, CornerQuadrant::TOP_RIGHT,
      CornerQuadrant::BOTTOM_RIGHT, // Order TL, TR, BR, BL for output vector
      CornerQuadrant::BOTTOM_LEFT};
  std::string quadrant_names[] = {"TOP_LEFT", "TOP_RIGHT", "BOTTOM_RIGHT",
                                  "BOTTOM_LEFT"};

  for (int i = 0; i < 4; ++i) {
    CornerQuadrant current_quad = quadrants_to_scan[i];
    LOG_INFO << "detectFourCornersGoBoard: Attempting to find "
             << quadrant_names[i] << " corner.";
    bool success = false;
    if (g_use_robust_corner_detection) {
      int classified_color_p1; // To store the color classified by the robust
                               // Pass 1
      success = adaptive_detect_stone_robust(
          rawBgrImage, current_quad, calibData,
          out_detected_raw_board_corners_tl_tr_br_bl[i], temp_corrected_image,
          temp_detected_radius,
          classified_color_p1); // New output param
      LOG_INFO << "Robust detection for " << quadrant_names[i]
               << " resulted in Pass 1 classified color: "
               << (classified_color_p1 == BLACK
                       ? "BLACK"
                       : (classified_color_p1 == WHITE ? "WHITE"
                                                       : "OTHER/EMPTY"));
    } else {
      success =
          adaptive_detect_stone(rawBgrImage, current_quad, calibData,
                                out_detected_raw_board_corners_tl_tr_br_bl[i],
                                temp_corrected_image, temp_detected_radius);
    }
    if (!success) {
      LOG_ERROR << "detectFourCornersGoBoard: Failed to find "
                << quadrant_names[i] << " stone.";
      // If any corner fails, the whole process fails.
      // Optionally, could try to collect any successful ones and return
      // partial, but current design is all or nothing.
      return false;
    }
    LOG_INFO << "detectFourCornersGoBoard: " << quadrant_names[i]
             << " stone found at raw coordinates: "
             << out_detected_raw_board_corners_tl_tr_br_bl[i]
             << " (Detected radius in its corrected view: "
             << temp_detected_radius << ")";
  }

  // Final check if all points are valid (not -1,-1), though
  // adaptive_detect_stone should return false if it fails.
  for (size_t i = 0; i < out_detected_raw_board_corners_tl_tr_br_bl.size();
       ++i) {
    if (out_detected_raw_board_corners_tl_tr_br_bl[i].x < 0 ||
        out_detected_raw_board_corners_tl_tr_br_bl[i].y < 0) {
      LOG_ERROR << "detectFourCornersGoBoard: Corner " << quadrant_names[i]
                << " was not successfully detected (still default value).";
      return false; // Should have been caught by adaptive_detect_stone
                    // returning false
    }
  }

  LOG_INFO << "detectFourCornersGoBoard: All four corners detected "
              "successfully using adaptive_detect_stone.";
  if (bDebug) {
    cv::Mat debug_display_raw = rawBgrImage.clone();
    if (out_detected_raw_board_corners_tl_tr_br_bl.size() == 4) {
      // Draw lines between TL-TR, TR-BR, BR-BL, BL-TL
      cv::line(debug_display_raw, out_detected_raw_board_corners_tl_tr_br_bl[0],
               out_detected_raw_board_corners_tl_tr_br_bl[1],
               cv::Scalar(0, 255, 0), 2); // TL-TR
      cv::line(debug_display_raw, out_detected_raw_board_corners_tl_tr_br_bl[1],
               out_detected_raw_board_corners_tl_tr_br_bl[2],
               cv::Scalar(0, 255, 0), 2); // TR-BR
      cv::line(debug_display_raw, out_detected_raw_board_corners_tl_tr_br_bl[2],
               out_detected_raw_board_corners_tl_tr_br_bl[3],
               cv::Scalar(0, 255, 0), 2); // BR-BL
      cv::line(debug_display_raw, out_detected_raw_board_corners_tl_tr_br_bl[3],
               out_detected_raw_board_corners_tl_tr_br_bl[0],
               cv::Scalar(0, 255, 0), 2); // BL-TL

      for (size_t i = 0; i < 4; ++i) {
        cv::circle(debug_display_raw,
                   out_detected_raw_board_corners_tl_tr_br_bl[i], 7,
                   cv::Scalar(0, 0, 255), -1);
        cv::putText(debug_display_raw, quadrant_names[i],
                    out_detected_raw_board_corners_tl_tr_br_bl[i] +
                        cv::Point2f(10, 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
      }
    }
    cv::imshow("detectFourCornersGoBoard - Final Raw Detections",
               debug_display_raw);
    cv::waitKey(0);
  }
  return true;
}

// --- Experimental Function V7 (Corrected Two-Pass Refinement with ENHANCED
// DEBUG VISUALIZATIONS) ---
bool experimental_scan_for_quadrant_stone(
    const cv::Mat &rawBgrImage,
    CornerQuadrant targetScanQuadrant, // e.g., CornerQuadrant::TOP_LEFT
    const CalibrationData &calibData,
    cv::Point2f &out_final_raw_corner_guess, // The raw corner guess that led to
                                             // the successful PASS 2 warp
    cv::Mat &out_final_corrected_image,      // The image from the PASS 2 warp
    cv::Point2f &out_detected_stone_center_in_final_corrected,
    float &out_detected_stone_radius_in_final_corrected,
    cv::Rect &out_focused_roi_in_final_corrected) {

  LOG_INFO << "--- Starting Experimental V7 (Enhanced Debug Vis) for "
           << (targetScanQuadrant == CornerQuadrant::TOP_LEFT ? "TOP_LEFT"
                                                              : "OTHER_QUAD")
           << " ---";

  std::string quadrant_name_str;
  cv::Vec3f target_lab_color;
  size_t target_ideal_dest_corner_idx = 0;

  out_final_corrected_image = cv::Mat();
  out_final_raw_corner_guess = cv::Point2f(-1, -1);

  // Determine target_lab_color
  if (!calibData.colors_loaded) {
    LOG_ERROR << "ScanV7: Calibration color data not loaded.";
    if (targetScanQuadrant == CornerQuadrant::TOP_LEFT) {
      LOG_WARN << "ScanV7: Using fallback L=50,A=128,B=128 for TL black stone.";
      target_lab_color = cv::Vec3f(50, 128, 128);
    } else {
      return false;
    }
  } else {
    switch (targetScanQuadrant) {
    case CornerQuadrant::TOP_LEFT:
      quadrant_name_str = "TOP_LEFT";
      target_lab_color = calibData.lab_tl;
      target_ideal_dest_corner_idx = 0;
      break;
    default:
      LOG_ERROR << "ScanV7: This experiment currently focuses on TOP_LEFT.";
      return false;
    }
  }
  if (target_lab_color[0] < 0 &&
      targetScanQuadrant == CornerQuadrant::TOP_LEFT) {
    LOG_WARN << "ScanV7: Invalid Lab for TL (L=" << target_lab_color[0]
             << "). Using fallback.";
    target_lab_color = cv::Vec3f(50, 128, 128);
  } else if (target_lab_color[0] < 0) {
    LOG_ERROR << "ScanV7: Invalid Lab color for " << quadrant_name_str
              << " (L=" << target_lab_color[0] << ").";
    return false;
  }
  LOG_DEBUG << "ScanV7: Using Lab ref " << target_lab_color << " for "
            << quadrant_name_str;
  if (rawBgrImage.empty()) {
    LOG_ERROR << "ScanV7: Raw BGR image empty.";
    return false;
  }

  std::vector<cv::Point2f> ideal_corrected_dest_points =
      getBoardCornersCorrected(rawBgrImage.cols, rawBgrImage.rows);

  // === PASS 1: Initial Rough Correction and Blob Finding ===
  LOG_INFO << "ScanV7 Pass 1: Initial rough perspective and blob finding.";
  cv::Point2f p1_raw_corner_initial_guess;
  if (targetScanQuadrant == CornerQuadrant::TOP_LEFT) {
    p1_raw_corner_initial_guess =
        cv::Point2f(static_cast<float>(rawBgrImage.cols) * 0.25f,
                    static_cast<float>(rawBgrImage.rows) * 0.25f);
  } else {
    return false;
  }
  LOG_DEBUG << "ScanV7 Pass 1: Initial raw " << quadrant_name_str
            << " corner guess: " << p1_raw_corner_initial_guess;

  std::vector<cv::Point2f> p1_source_points_raw(4);
  float p1_est_board_span_x = static_cast<float>(rawBgrImage.cols) * 0.75f;
  float p1_est_board_span_y = static_cast<float>(rawBgrImage.rows) * 0.75f;

  if (targetScanQuadrant == CornerQuadrant::TOP_LEFT) {
    p1_source_points_raw[0] = p1_raw_corner_initial_guess;
    p1_source_points_raw[1] =
        cv::Point2f(p1_raw_corner_initial_guess.x + p1_est_board_span_x,
                    p1_raw_corner_initial_guess.y);
    p1_source_points_raw[3] =
        cv::Point2f(p1_raw_corner_initial_guess.x,
                    p1_raw_corner_initial_guess.y + p1_est_board_span_y);
    p1_source_points_raw[2] =
        cv::Point2f(p1_raw_corner_initial_guess.x + p1_est_board_span_x,
                    p1_raw_corner_initial_guess.y + p1_est_board_span_y);
  }
  for (cv::Point2f &pt : p1_source_points_raw) {
    pt.x = std::max(0.0f,
                    std::min(static_cast<float>(rawBgrImage.cols - 1), pt.x));
    pt.y = std::max(0.0f,
                    std::min(static_cast<float>(rawBgrImage.rows - 1), pt.y));
  }
  if (cv::contourArea(p1_source_points_raw) <
      (p1_est_board_span_x * p1_est_board_span_y * 0.01)) {
    LOG_ERROR << "ScanV7 Pass 1: Degenerate raw quad p1_source_points_raw.";
    out_final_raw_corner_guess = p1_raw_corner_initial_guess;
    return false;
  }

  cv::Mat M1 = cv::getPerspectiveTransform(p1_source_points_raw,
                                           ideal_corrected_dest_points);
  if (M1.empty() || cv::determinant(M1) < 1e-6) {
    LOG_ERROR << "ScanV7 Pass 1: Degenerate transform M1.";
    out_final_raw_corner_guess = p1_raw_corner_initial_guess;
    return false;
  }

  cv::Mat image_pass1_corrected;
  cv::warpPerspective(rawBgrImage, image_pass1_corrected, M1,
                      rawBgrImage.size());
  if (image_pass1_corrected.empty()) {
    LOG_ERROR << "ScanV7 Pass 1: Warped image_pass1_corrected empty.";
    out_final_raw_corner_guess = p1_raw_corner_initial_guess;
    return false;
  }

  // Tentatively set outputs based on P1 - will be overwritten by P2 if P2 is
  // successful
  out_final_corrected_image = image_pass1_corrected.clone();
  out_final_raw_corner_guess = p1_raw_corner_initial_guess;

  cv::Rect roi_quadrant_pass1(0, 0, image_pass1_corrected.cols / 2,
                              image_pass1_corrected.rows / 2);

  cv::Point2f p1_blob_center_in_pass1_corrected;
  double p1_blob_area = 0.0;
  bool blob_found_p1 = find_largest_color_blob_in_roi(
      image_pass1_corrected, roi_quadrant_pass1, target_lab_color,
      CALIB_L_TOLERANCE_STONE, CALIB_AB_TOLERANCE_STONE,
      p1_blob_center_in_pass1_corrected, p1_blob_area);

  if (bDebug) { // DEBUG VISUALIZATION 1A: Pass 1 Corrected View + Blob
    cv::Mat debug_p1_corrected_disp = image_pass1_corrected.clone();
    cv::rectangle(debug_p1_corrected_disp, roi_quadrant_pass1,
                  cv::Scalar(255, 0, 0), 2);
    cv::putText(debug_p1_corrected_disp, "P1 ROI",
                roi_quadrant_pass1.tl() + cv::Point(5, 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
    if (blob_found_p1) {
      cv::circle(debug_p1_corrected_disp, p1_blob_center_in_pass1_corrected, 7,
                 cv::Scalar(0, 255, 0), -1);
      cv::putText(debug_p1_corrected_disp,
                  "Blob Area: " + std::to_string((int)p1_blob_area),
                  p1_blob_center_in_pass1_corrected + cv::Point2f(10, 0),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    } else {
      cv::putText(debug_p1_corrected_disp, "No Blob Found in P1 ROI",
                  cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  cv::Scalar(0, 0, 255), 1);
    }
    cv::imshow("Debug V7 - P1 Corrected + Blob", debug_p1_corrected_disp);
    cv::waitKey(0);
  }

  if (!blob_found_p1) {
    LOG_WARN << "ScanV7 Pass 1: No color blob found in " << quadrant_name_str
             << " of image_pass1_corrected.";
    return false;
  }
  LOG_INFO << "ScanV7 Pass 1: Largest blob. Area: " << p1_blob_area
           << ", Center in P1_corrected: " << p1_blob_center_in_pass1_corrected;

  // === PASS 2: Refined Perspective Transformation and Focused Verification ===
  LOG_INFO << "ScanV7 Pass 2: Refining perspective based on Pass 1 blob's raw "
              "position.";

  cv::Mat M1_inv;
  if (!cv::invert(M1, M1_inv, cv::DECOMP_SVD) || M1_inv.empty()) {
    LOG_ERROR << "ScanV7 Pass 2: Failed to invert M1 transform.";
    return false;
  }

  std::vector<cv::Point2f> p1_blob_center_vec_corrected_tf = {
      p1_blob_center_in_pass1_corrected};
  std::vector<cv::Point2f> p1_blob_center_in_raw_image_vec;
  cv::perspectiveTransform(p1_blob_center_vec_corrected_tf,
                           p1_blob_center_in_raw_image_vec, M1_inv);
  if (p1_blob_center_in_raw_image_vec.empty()) {
    LOG_ERROR << "ScanV7 Pass 2: Transform p1_blob_center to raw failed.";
    return false;
  }
  cv::Point2f p1_blob_center_in_raw_image = p1_blob_center_in_raw_image_vec[0];
  LOG_DEBUG << "ScanV7 Pass 2: Pass 1 blob center mapped to raw image coords: "
            << p1_blob_center_in_raw_image;

  if (bDebug) { // DEBUG VISUALIZATION 1B: Raw Image + Initial P1 Guess & Mapped
                // Blob Center
    cv::Mat debug_raw_disp_p1_map = rawBgrImage.clone();
    cv::circle(debug_raw_disp_p1_map, p1_raw_corner_initial_guess, 8,
               cv::Scalar(255, 0, 0), 2);
    cv::putText(debug_raw_disp_p1_map, "P1 Initial Guess",
                p1_raw_corner_initial_guess + cv::Point2f(10, -10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
    cv::circle(debug_raw_disp_p1_map, p1_blob_center_in_raw_image, 8,
               cv::Scalar(0, 255, 0), 2);
    cv::putText(debug_raw_disp_p1_map, "P1 Mapped Blob Center",
                p1_blob_center_in_raw_image + cv::Point2f(10, 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    cv::imshow("Debug V7 - Raw + P1 Guess & Mapped Blob",
               debug_raw_disp_p1_map);
    cv::waitKey(0);
  }

  std::vector<cv::Point2f> ideal_corners_p1_corr = getBoardCornersCorrected(
      image_pass1_corrected.cols, image_pass1_corrected.rows);
  float board_w_p1_corr =
      ideal_corners_p1_corr[1].x - ideal_corners_p1_corr[0].x;
  float board_h_p1_corr =
      ideal_corners_p1_corr[3].y - ideal_corners_p1_corr[0].y;
  float avg_spacing_p1_corr =
      ((board_w_p1_corr / 18.0f) + (board_h_p1_corr / 18.0f)) * 0.5f;
  float est_radius_in_p1_corrected = avg_spacing_p1_corr * 0.47f;

  float est_raw_radius_for_offset =
      std::min(rawBgrImage.cols, rawBgrImage.rows) * 0.015f;
  if (est_radius_in_p1_corrected > 1.0f) {
    std::vector<cv::Point2f> radius_pts_p1_corr_tf = {
        p1_blob_center_in_pass1_corrected,
        cv::Point2f(p1_blob_center_in_pass1_corrected.x +
                        est_radius_in_p1_corrected,
                    p1_blob_center_in_pass1_corrected.y)};
    std::vector<cv::Point2f> radius_pts_raw_tf;
    cv::perspectiveTransform(radius_pts_p1_corr_tf, radius_pts_raw_tf, M1_inv);
    if (radius_pts_raw_tf.size() == 2) {
      float temp_raw_rad =
          cv::norm(radius_pts_raw_tf[0] - radius_pts_raw_tf[1]);
      if (temp_raw_rad > 1.0f &&
          temp_raw_rad < std::min(rawBgrImage.cols, rawBgrImage.rows) * 0.1) {
        est_raw_radius_for_offset = temp_raw_rad;
      }
    }
  }
  LOG_DEBUG << "ScanV7 Pass 2: Estimated raw radius for offset from stone "
               "center to corner: "
            << est_raw_radius_for_offset;

  cv::Point2f raw_corner_guess_pass2;
  if (targetScanQuadrant == CornerQuadrant::TOP_LEFT)
    raw_corner_guess_pass2 =
        p1_blob_center_in_raw_image -
        cv::Point2f(est_raw_radius_for_offset, est_raw_radius_for_offset);
  else {
    return false;
  }

  raw_corner_guess_pass2.x =
      std::max(0.0f, std::min(static_cast<float>(rawBgrImage.cols - 1),
                              raw_corner_guess_pass2.x));
  raw_corner_guess_pass2.y =
      std::max(0.0f, std::min(static_cast<float>(rawBgrImage.rows - 1),
                              raw_corner_guess_pass2.y));

  out_final_raw_corner_guess = raw_corner_guess_pass2;
  LOG_DEBUG << "ScanV7 Pass 2: Refined raw " << quadrant_name_str
            << " corner guess for M2: " << raw_corner_guess_pass2;

  std::vector<cv::Point2f> source_points_pass2_raw(4);
  // Use same est_board_span for M2 as for M1 for this heuristic
  if (targetScanQuadrant == CornerQuadrant::TOP_LEFT) {
    source_points_pass2_raw[0] = raw_corner_guess_pass2;
    source_points_pass2_raw[1] =
        cv::Point2f(raw_corner_guess_pass2.x + p1_est_board_span_x,
                    raw_corner_guess_pass2.y);
    source_points_pass2_raw[3] =
        cv::Point2f(raw_corner_guess_pass2.x,
                    raw_corner_guess_pass2.y + p1_est_board_span_y);
    source_points_pass2_raw[2] =
        cv::Point2f(raw_corner_guess_pass2.x + p1_est_board_span_x,
                    raw_corner_guess_pass2.y + p1_est_board_span_y);
  }
  for (cv::Point2f &pt : source_points_pass2_raw) { /* Clamp */
    pt.x = std::max(0.0f,
                    std::min(static_cast<float>(rawBgrImage.cols - 1), pt.x));
    pt.y = std::max(0.0f,
                    std::min(static_cast<float>(rawBgrImage.rows - 1), pt.y));
  }

  if (bDebug) { // DEBUG VISUALIZATION 2A: Raw Image with M2 Setup
    cv::Mat debug_raw_p2_setup = rawBgrImage.clone();
    cv::circle(debug_raw_p2_setup, p1_blob_center_in_raw_image, 5,
               cv::Scalar(0, 255, 0), -1);
    cv::putText(debug_raw_p2_setup, "P1 Blob (raw)",
                p1_blob_center_in_raw_image + cv::Point2f(5, -5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0));
    cv::circle(debug_raw_p2_setup, raw_corner_guess_pass2, 5,
               cv::Scalar(0, 0, 255), -1);
    cv::putText(debug_raw_p2_setup, "P2 Raw Corner Guess",
                raw_corner_guess_pass2 + cv::Point2f(5, 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255));
    for (size_t i = 0; i < source_points_pass2_raw.size(); ++i) {
      cv::line(debug_raw_p2_setup, source_points_pass2_raw[i],
               source_points_pass2_raw[(i + 1) % 4], cv::Scalar(255, 0, 255),
               1);
    }
    cv::imshow("Debug V7 - P2 Raw Setup for M2", debug_raw_p2_setup);
    cv::waitKey(0);
  }

  if (cv::contourArea(source_points_pass2_raw) <
      (p1_est_board_span_x * p1_est_board_span_y * 0.01)) {
    LOG_ERROR << "ScanV7 Pass 2: Degenerate raw quad for M2.";
    return false;
  }

  cv::Mat M2 = cv::getPerspectiveTransform(source_points_pass2_raw,
                                           ideal_corrected_dest_points);
  if (M2.empty() || cv::determinant(M2) < 1e-6) {
    LOG_ERROR << "ScanV7 Pass 2: Degenerate transform M2.";
    return false;
  }

  cv::Mat image_pass2_corrected;
  cv::warpPerspective(rawBgrImage, image_pass2_corrected, M2,
                      rawBgrImage.size());
  if (image_pass2_corrected.empty()) {
    LOG_ERROR << "ScanV7 Pass 2: Warped image_pass2_corrected empty.";
    return false;
  }
  out_final_corrected_image = image_pass2_corrected.clone();

  std::vector<cv::Point2f> ideal_corners_p2_corr = getBoardCornersCorrected(
      image_pass2_corrected.cols, image_pass2_corrected.rows);
  float board_w_p2_corr =
      ideal_corners_p2_corr[1].x - ideal_corners_p2_corr[0].x;
  float board_h_p2_corr =
      ideal_corners_p2_corr[3].y - ideal_corners_p2_corr[0].y;
  if (board_w_p2_corr < 19.0f || board_h_p2_corr < 19.0f) {
    LOG_WARN << "ScanV7 P2: Corrected board small in P2 image.";
    return false;
  }

  float expected_radius_for_pass2_corrected =
      calculateAdaptiveSampleRadius(board_w_p2_corr, board_h_p2_corr);
  LOG_DEBUG << "ScanV7 P2: Exp radius for P2 "
            << expected_radius_for_pass2_corrected << std::endl;
  if (expected_radius_for_pass2_corrected < 1.0f) {
    LOG_WARN << "ScanV7 P2: Exp radius for P2 verification small.";
    return false;
  }

  // Per user instruction: Center focused ROI on the ideal corrected board
  // corner for this quadrant.
  cv::Point2f focused_roi_center_ideal_target =
      ideal_corrected_dest_points[target_ideal_dest_corner_idx];
  LOG_DEBUG
      << "ScanV7 Pass 2: Centering focused ROI on ideal corrected corner: "
      << focused_roi_center_ideal_target;

  // Also, let's find where the p1_blob_center_in_raw_image projects to with M2,
  // for interest
  std::vector<cv::Point2f> raw_blob_center_for_M2_tf = {
      p1_blob_center_in_raw_image};
  std::vector<cv::Point2f> p2_projected_blob_center_vec;
  cv::perspectiveTransform(raw_blob_center_for_M2_tf,
                           p2_projected_blob_center_vec, M2);
  cv::Point2f p2_projected_blob_center = p2_projected_blob_center_vec[0];
  LOG_DEBUG << "ScanV7 Pass 2: P1's raw blob center projects to "
            << p2_projected_blob_center << " in image_pass2_corrected";

  out_focused_roi_in_final_corrected = calculateGridIntersectionROI(
      0, 0, image_pass2_corrected.cols, image_pass2_corrected.rows);

  if (bDebug) { // DEBUG VISUALIZATION 2B: Pass 2 Corrected View + Focused ROI
                // (Pre-Detect)
    cv::Mat debug_p2_corrected_disp = image_pass2_corrected.clone();
    cv::circle(debug_p2_corrected_disp, focused_roi_center_ideal_target, 5,
               cv::Scalar(255, 0, 255),
               -1); // Magenta: Ideal corner (ROI center)
    cv::putText(debug_p2_corrected_disp, "Ideal Corner (ROI Center)",
                focused_roi_center_ideal_target + cv::Point2f(5, -5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 255));
    cv::circle(debug_p2_corrected_disp, p2_projected_blob_center, 5,
               cv::Scalar(0, 255, 255), -1); // Yellow: Projected P1 Blob Center
    cv::putText(debug_p2_corrected_disp, "Projected P1 Blob",
                p2_projected_blob_center + cv::Point2f(5, 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 255));
    cv::rectangle(debug_p2_corrected_disp, out_focused_roi_in_final_corrected,
                  cv::Scalar(255, 255, 0), 2); // Cyan: focused ROI
    cv::imshow("Debug V7 - P2 Corrected + Focused ROI (Pre-Detect)",
               debug_p2_corrected_disp);
    cv::waitKey(0);
  }
  LOG_DEBUG << "ScanV7 Pass 2: Focused ROI in final image: "
            << out_focused_roi_in_final_corrected
            << ", ExpR for detection: " << expected_radius_for_pass2_corrected;

  bool final_stone_found = detectSpecificColoredRoundShape(
      image_pass2_corrected, out_focused_roi_in_final_corrected,
      target_lab_color, CALIB_L_TOLERANCE_STONE, CALIB_AB_TOLERANCE_STONE,
      expected_radius_for_pass2_corrected,
      out_detected_stone_center_in_final_corrected,
      out_detected_stone_radius_in_final_corrected);

  if (bDebug) { // DEBUG VISUALIZATION 2C: Pass 2 Final Verification Result
    cv::Mat final_debug_disp =
        image_pass2_corrected
            .clone(); // out_final_corrected_image could also be used
    cv::rectangle(final_debug_disp, out_focused_roi_in_final_corrected,
                  cv::Scalar(255, 255, 0), 1); // Cyan ROI
    cv::circle(final_debug_disp, focused_roi_center_ideal_target, 3,
               cv::Scalar(255, 0, 255),
               -1); // Magenta: Ideal corner (ROI center)
    cv::circle(final_debug_disp, p2_projected_blob_center, 3,
               cv::Scalar(0, 128, 255), -1); // Orange: Projected P1 Blob Center

    if (final_stone_found) {
      cv::circle(final_debug_disp, out_detected_stone_center_in_final_corrected,
                 static_cast<int>(out_detected_stone_radius_in_final_corrected),
                 cv::Scalar(0, 255, 0), 2); // Green: Detected Stone
      cv::putText(final_debug_disp, "DETECTED",
                  out_detected_stone_center_in_final_corrected,
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    } else {
      cv::putText(final_debug_disp, "NOT FOUND in focused ROI",
                  cv::Point(out_focused_roi_in_final_corrected.x,
                            out_focused_roi_in_final_corrected.y - 10),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    }
    cv::imshow("Debug V7 - P2 Final Verification Result", final_debug_disp);
    cv::waitKey(0);
  }

  if (final_stone_found) {
    LOG_INFO << "ScanV7 Pass 2 SUCCESS: Stone validated for "
             << quadrant_name_str << ". Final Raw Corner Guess used for M2: "
             << raw_corner_guess_pass2
             << ". Detected R=" << out_detected_stone_radius_in_final_corrected;
    return true;
  } else {
    LOG_WARN << "ScanV7 Pass 2 FAILED: Stone NOT validated in focused ROI for "
             << quadrant_name_str;
    return false;
  }
}
// END EDITING

bool adaptive_detect_stone(
    const cv::Mat &rawBgrImage,
    CornerQuadrant targetScanQuadrant, // e.g., CornerQuadrant::TOP_LEFT
    const CalibrationData &calibData,
    cv::Point2f &out_final_raw_corner_guess, // The raw corner guess that led to
                                             // the successful PASS 2 warp
    cv::Mat &out_final_corrected_image,      // The image from the PASS 2 warp

    float &out_detected_stone_radius_in_final_corrected) {
  cv::Point2f out_detected_stone_center_in_final_corrected;
  cv::Rect out_focused_roi_in_final_corrected;
  LOG_INFO << "--- adaptive_detect_stone for " << targetScanQuadrant << " ---";

  std::string quadrant_name_str = toString(targetScanQuadrant);
  cv::Vec3f target_lab_color;
  size_t target_ideal_dest_corner_idx = static_cast<size_t>(targetScanQuadrant);

  out_final_corrected_image = cv::Mat();
  out_final_raw_corner_guess = cv::Point2f(-1, -1);

  const cv::Vec3f default_black_stone_color(50, 128, 128);
  const cv::Vec3f default_white_stone_color(223, 120, 120);

  // Determine target_lab_color
  if (!calibData.colors_loaded) {
    LOG_WARN << "adaptive_detect_stone: Calibration color data not loaded."
             << " Using fallback  for" << quadrant_name_str;
  }
  float tl_x = static_cast<float>(rawBgrImage.cols) * 0.25f;
  float tl_y = static_cast<float>(rawBgrImage.rows) * 0.25f;
  float br_x = static_cast<float>(rawBgrImage.cols) * 0.75f;
  float br_y = static_cast<float>(rawBgrImage.rows) * 0.75f;

  std::vector<cv::Point2f> p1_source_points_raw = {
      cv::Point2f(tl_x, tl_y), cv::Point2f(br_x, tl_y), cv::Point2f(br_x, br_y),
      cv::Point2f(tl_x, br_y)};

  switch (targetScanQuadrant) {
  case CornerQuadrant::TOP_LEFT:
    target_lab_color =
        calibData.colors_loaded ? calibData.lab_tl : default_black_stone_color;
    break;
  case CornerQuadrant::TOP_RIGHT:
    target_lab_color =
        calibData.colors_loaded ? calibData.lab_tr : default_white_stone_color;
    break;
  case CornerQuadrant::BOTTOM_LEFT:
    target_lab_color =
        calibData.colors_loaded ? calibData.lab_bl : default_black_stone_color;
    break;
  case CornerQuadrant::BOTTOM_RIGHT:
    target_lab_color =
        calibData.colors_loaded ? calibData.lab_br : default_white_stone_color;
    break;
  }

  LOG_DEBUG << "adaptive_detect_stone: Using Lab ref " << target_lab_color
            << " for " << quadrant_name_str;
  if (rawBgrImage.empty()) {
    LOG_ERROR << "adaptive_detect_stone: Raw BGR image empty.";
    return false;
  }

  std::vector<cv::Point2f> ideal_corrected_dest_points =
      getBoardCornersCorrected(rawBgrImage.cols, rawBgrImage.rows);

  // === PASS 1: Initial Rough Correction and Blob Finding ===
  LOG_INFO << "adaptive_detect_stone Pass 1: Initial rough perspective and "
              "blob finding.";

  LOG_DEBUG << "adaptive_detect_stone Pass 1: Initial raw " << quadrant_name_str
            << " corner guess: "
            << p1_source_points_raw[target_ideal_dest_corner_idx];

  for (cv::Point2f &pt : p1_source_points_raw) {
    pt.x = std::max(0.0f,
                    std::min(static_cast<float>(rawBgrImage.cols - 1), pt.x));
    pt.y = std::max(0.0f,
                    std::min(static_cast<float>(rawBgrImage.rows - 1), pt.y));
  }

  cv::Mat M1 = cv::getPerspectiveTransform(p1_source_points_raw,
                                           ideal_corrected_dest_points);
  if (M1.empty() || cv::determinant(M1) < 1e-6) {
    LOG_ERROR << "adaptive_detect_stone Pass 1: Degenerate transform M1.";
    return false;
  }

  cv::Mat image_pass1_corrected;
  cv::warpPerspective(rawBgrImage, image_pass1_corrected, M1,
                      rawBgrImage.size());
  if (image_pass1_corrected.empty()) {
    LOG_ERROR
        << "adaptive_detect_stone Pass 1: Warped image_pass1_corrected empty.";
    return false;
  }

  // Tentatively set outputs based on P1 - will be overwritten by P2 if P2 is
  // successful
  out_final_corrected_image = image_pass1_corrected.clone();
  out_final_raw_corner_guess =
      p1_source_points_raw[target_ideal_dest_corner_idx];

  cv::Rect roi_quadrant_pass1;
  switch (targetScanQuadrant) {
  case CornerQuadrant::TOP_LEFT:
    roi_quadrant_pass1 = cv::Rect(0, 0, image_pass1_corrected.cols / 2,
                                  image_pass1_corrected.rows / 2);
    break;
  case CornerQuadrant::TOP_RIGHT:
    roi_quadrant_pass1 = cv::Rect(image_pass1_corrected.cols / 2, 0,
                                  image_pass1_corrected.cols / 2,
                                  image_pass1_corrected.rows / 2);
    break;
  case CornerQuadrant::BOTTOM_LEFT:
    roi_quadrant_pass1 = cv::Rect(0, image_pass1_corrected.rows / 2,
                                  image_pass1_corrected.cols / 2,
                                  image_pass1_corrected.rows / 2);
    break;
  case CornerQuadrant::BOTTOM_RIGHT:
    roi_quadrant_pass1 = cv::Rect(
        image_pass1_corrected.cols / 2, image_pass1_corrected.rows / 2,
        image_pass1_corrected.cols / 2, image_pass1_corrected.rows / 2);
    break;
  }

  cv::Point2f p1_blob_center_in_pass1_corrected;
  double p1_blob_area = 0.0;
  bool blob_found_p1 = find_largest_color_blob_in_roi(
      image_pass1_corrected, roi_quadrant_pass1, target_lab_color,
      CALIB_L_TOLERANCE_STONE, CALIB_AB_TOLERANCE_STONE,
      p1_blob_center_in_pass1_corrected, p1_blob_area);

  if (bDebug) { // DEBUG VISUALIZATION 1A: Pass 1 Corrected View + Blob
    cv::Mat debug_p1_corrected_disp = image_pass1_corrected.clone();
    cv::rectangle(debug_p1_corrected_disp, roi_quadrant_pass1,
                  cv::Scalar(255, 0, 0), 2);
    cv::putText(debug_p1_corrected_disp, "P1 ROI",
                roi_quadrant_pass1.tl() + cv::Point(5, 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
    if (blob_found_p1) {
      cv::circle(debug_p1_corrected_disp, p1_blob_center_in_pass1_corrected, 7,
                 cv::Scalar(0, 255, 0), -1);
      cv::putText(debug_p1_corrected_disp,
                  "Blob Area: " + std::to_string((int)p1_blob_area),
                  p1_blob_center_in_pass1_corrected + cv::Point2f(10, 0),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    } else {
      cv::putText(debug_p1_corrected_disp, "No Blob Found in P1 ROI",
                  cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  cv::Scalar(0, 0, 255), 1);
    }
    cv::imshow("Debug V7 - P1 Corrected + Blob", debug_p1_corrected_disp);
    cv::waitKey(0);
  }

  if (!blob_found_p1) {
    LOG_WARN << "adaptive_detect_stone Pass 1: No color blob found in "
             << quadrant_name_str << " of image_pass1_corrected.";
    return false;
  }
  LOG_INFO << "adaptive_detect_stone Pass 1: Largest blob. Area: "
           << p1_blob_area
           << ", Center in P1_corrected: " << p1_blob_center_in_pass1_corrected;

  // === PASS 2: Refined Perspective Transformation and Focused Verification ===
  LOG_INFO << "adaptive_detect_stone Pass 2: Refining perspective based on "
              "Pass 1 blob's raw "
              "position.";

  cv::Mat M1_inv;
  if (!cv::invert(M1, M1_inv, cv::DECOMP_SVD) || M1_inv.empty()) {
    LOG_ERROR << "adaptive_detect_stone Pass 2: Failed to invert M1 transform.";
    return false;
  }

  std::vector<cv::Point2f> p1_blob_center_vec_corrected_tf = {
      p1_blob_center_in_pass1_corrected};
  std::vector<cv::Point2f> p1_blob_center_in_raw_image_vec;
  cv::perspectiveTransform(p1_blob_center_vec_corrected_tf,
                           p1_blob_center_in_raw_image_vec, M1_inv);
  if (p1_blob_center_in_raw_image_vec.empty()) {
    LOG_ERROR << "adaptive_detect_stone Pass 2: Transform p1_blob_center to "
                 "raw failed.";
    return false;
  }
  cv::Point2f p1_blob_center_in_raw_image = p1_blob_center_in_raw_image_vec[0];
  LOG_DEBUG << "adaptive_detect_stone Pass 2: Pass 1 blob center mapped to raw "
               "image coords: "
            << p1_blob_center_in_raw_image;

  std::vector<cv::Point2f> source_points_pass2_raw = p1_source_points_raw;
  source_points_pass2_raw[target_ideal_dest_corner_idx] =
      p1_blob_center_in_raw_image;

  cv::Mat M2 = cv::getPerspectiveTransform(source_points_pass2_raw,
                                           ideal_corrected_dest_points);
  if (M2.empty() || cv::determinant(M2) < 1e-6) {
    LOG_ERROR << "adaptive_detect_stone Pass 2: Degenerate transform M2.";
    return false;
  }

  cv::Mat image_pass2_corrected;
  cv::warpPerspective(rawBgrImage, image_pass2_corrected, M2,
                      rawBgrImage.size());
  if (image_pass2_corrected.empty()) {
    LOG_ERROR
        << "adaptive_detect_stone Pass 2: Warped image_pass2_corrected empty.";
    return false;
  }
  out_final_corrected_image = image_pass2_corrected.clone();
  float board_w_p2_corr =
      ideal_corrected_dest_points[1].x - ideal_corrected_dest_points[0].x;
  float board_h_p2_corr =
      ideal_corrected_dest_points[3].y - ideal_corrected_dest_points[0].y;

  float expected_radius_for_pass2_corrected =
      calculateAdaptiveSampleRadius(board_w_p2_corr, board_h_p2_corr);
  LOG_DEBUG << "adaptive_detect_stone P2: Exp radius for P2 "
            << expected_radius_for_pass2_corrected << std::endl;
  if (expected_radius_for_pass2_corrected < 1.0f) {
    LOG_WARN
        << "adaptive_detect_stone P2: Exp radius for P2 verification small.";
    return false;
  }

  // Per user instruction: Center focused ROI on the ideal corrected board
  // corner for this quadrant.
  cv::Point2f focused_roi_center_ideal_target =
      ideal_corrected_dest_points[target_ideal_dest_corner_idx];
  LOG_DEBUG << "adaptive_detect_stone Pass 2: Centering focused ROI on ideal "
               "corrected corner: "
            << focused_roi_center_ideal_target;

  out_focused_roi_in_final_corrected = calculateGridIntersectionROI(
      0, 0, image_pass2_corrected.cols, image_pass2_corrected.rows);

  LOG_DEBUG << "adaptive_detect_stone Pass 2: Focused ROI in final image: "
            << out_focused_roi_in_final_corrected
            << ", ExpR for detection: " << expected_radius_for_pass2_corrected;

  bool final_stone_found = detectSpecificColoredRoundShape(
      image_pass2_corrected, out_focused_roi_in_final_corrected,
      target_lab_color, CALIB_L_TOLERANCE_STONE, CALIB_AB_TOLERANCE_STONE,
      expected_radius_for_pass2_corrected,
      out_detected_stone_center_in_final_corrected,
      out_detected_stone_radius_in_final_corrected);

  if (final_stone_found) {
    LOG_INFO << "adaptive_detect_stone Pass 2 SUCCESS: Stone validated for "
             << quadrant_name_str
             << ". Detected R=" << out_detected_stone_radius_in_final_corrected;
    return true;
  } else {
    LOG_WARN << "adaptive_detect_stone Pass 2 FAILED: Stone NOT validated in "
                "focused ROI for "
             << quadrant_name_str;
    return false;
  }
}
// =================================================================================
// START OF REFACTORING: find_best_round_shape_iterative
// =================================================================================

// A small struct to hold the geometry constraints for a valid stone.
struct StoneGeometryConstraints {
  double min_acceptable_area;
  double max_acceptable_area;
  double min_acceptable_circularity;
};

// --- Refactored Utility Function 1: Prepare Iteration Parameters & Constraints ---
// Validates inputs and prepares common variables and geometry constraints for the search.
static bool prepare_iteration_parameters(
    const cv::Mat &image_to_search_bgr, const cv::Rect &roi_in_image,
    float expected_stone_radius_in_image,
    // Outputs
    cv::Rect &out_valid_roi, cv::Mat &out_roi_lab_content,
    StoneGeometryConstraints &out_constraints) {

  if (image_to_search_bgr.empty()) {
    LOG_ERROR << "FBS: Input BGR image empty.";
    return false;
  }
  out_valid_roi = roi_in_image & cv::Rect(0, 0, image_to_search_bgr.cols, image_to_search_bgr.rows);
  if (out_valid_roi.width <= 0 || out_valid_roi.height <= 0) {
    LOG_ERROR << "FBS: Invalid ROI after clamping: " << out_valid_roi;
    return false;
  }

  cv::cvtColor(image_to_search_bgr(out_valid_roi), out_roi_lab_content, cv::COLOR_BGR2Lab);

  double expected_area = CV_PI * expected_stone_radius_in_image * expected_stone_radius_in_image;
  out_constraints.min_acceptable_area = expected_area * ABS_STONE_AREA_MIN_FACTOR;
  out_constraints.max_acceptable_area = expected_area * ABS_STONE_AREA_MAX_FACTOR;
  // Use a general, strict circularity for the initial shape search.
  out_constraints.min_acceptable_circularity = std::max(MIN_STONE_CIRCULARITY_WHITE, MIN_STONE_CIRCULARITY_BLACK);

  LOG_DEBUG << "  FBS Prep: Expected Area: " << expected_area 
            << " (Range: " << out_constraints.min_acceptable_area << "-" << out_constraints.max_acceptable_area << ")"
            << ", Min Circ: " << out_constraints.min_acceptable_circularity;
            
  return true;
}

// --- Refactored Utility Function 2: Generate L-Value Candidates ---
// Creates the sorted list of L-channel base values to iterate through.
static std::vector<float> generate_l_value_candidates(
    float initial_target_L_value_hint, float l_base_min, float l_base_max, float l_base_step) {

  std::vector<float> l_base_values_to_try;
  // Prioritize the hint value by adding it first.
  if (initial_target_L_value_hint >= l_base_min && initial_target_L_value_hint <= l_base_max) {
    l_base_values_to_try.push_back(initial_target_L_value_hint);
  }

  // Add all other values in the range.
  for (float l_val = l_base_min; l_val <= l_base_max; l_val += l_base_step) {
    // Avoid re-adding the hint value if it falls on a step.
    if (std::abs(l_val - initial_target_L_value_hint) > 1e-3) {
      l_base_values_to_try.push_back(l_val);
    }
  }
  
  // Sort and unique the values to ensure a clean list, though the generation
  // logic mostly prevents duplicates already.
  std::sort(l_base_values_to_try.begin(), l_base_values_to_try.end());
  l_base_values_to_try.erase(
      std::unique(l_base_values_to_try.begin(), l_base_values_to_try.end()),
      l_base_values_to_try.end());
      
  return l_base_values_to_try;
}

// --- Refactored Utility Function 3: Find Largest Contour for Color Range ---
// The workhorse function that creates a color mask and finds the largest raw contour.
static bool find_largest_contour_for_color_range(
    const cv::Mat &roi_lab_content, float current_base_L, float current_l_tol,
    float fixed_ab_target_A, float fixed_ab_target_B, float fixed_ab_tolerance,
    // Output
    std::vector<cv::Point> &out_largest_contour) {

  cv::Mat color_mask;
  cv::Scalar lab_lower(
      std::max(0.f, current_base_L - current_l_tol),
      std::max(0.f, fixed_ab_target_A - fixed_ab_tolerance),
      std::max(0.f, fixed_ab_target_B - fixed_ab_tolerance));
  cv::Scalar lab_upper(
      std::min(255.f, current_base_L + current_l_tol),
      std::min(255.f, fixed_ab_target_A + fixed_ab_tolerance),
      std::min(255.f, fixed_ab_target_B + fixed_ab_tolerance));
      
  cv::inRange(roi_lab_content, lab_lower, lab_upper, color_mask);

  cv::Mat open_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(MORPH_OPEN_KERNEL_SIZE_STONE, MORPH_OPEN_KERNEL_SIZE_STONE));
  cv::Mat close_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(MORPH_CLOSE_KERNEL_SIZE_STONE, MORPH_CLOSE_KERNEL_SIZE_STONE));
  cv::morphologyEx(color_mask, color_mask, cv::MORPH_OPEN, open_kernel, cv::Point(-1,-1), MORPH_OPEN_ITERATIONS_STONE);
  cv::morphologyEx(color_mask, color_mask, cv::MORPH_CLOSE, close_kernel, cv::Point(-1,-1), MORPH_CLOSE_ITERATIONS_STONE);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(color_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  if (contours.empty()) return false;

  double max_area_this_iter = 0;
  const std::vector<cv::Point> *largest_contour_ptr = nullptr;
  for (const auto &contour : contours) {
    double area = cv::contourArea(contour);
    if (area > max_area_this_iter) {
      max_area_this_iter = area;
      largest_contour_ptr = &contour;
    }
  }

  if (largest_contour_ptr) {
    out_largest_contour = *largest_contour_ptr;
    return true;
  }
  return false;
}

// --- Refactored Utility Function 4: Validate Contour Geometry ---
// Checks if a given contour meets the strict geometric requirements of a Go stone.
static bool validate_contour_geometry(
    const std::vector<cv::Point> &contour,
    const StoneGeometryConstraints &constraints) {

  if (contour.size() < MIN_CONTOUR_POINTS_STONE) {
    return false; // Too few points to be a reliable shape.
  }

  double area = cv::contourArea(contour);
  double perimeter = cv::arcLength(contour, true);

  if (perimeter < 1.0) {
    return false; // Degenerate contour.
  }
  double circularity = (4 * CV_PI * area) / (perimeter * perimeter);
  
  LOG_DEBUG << "    Validating contour: Area=" << area << " Circ=" << circularity;

  if (area < constraints.min_acceptable_area || area > constraints.max_acceptable_area ||
      circularity < constraints.min_acceptable_circularity) {
    LOG_DEBUG << "    -> REJECTED on geometry.";
    return false;
  }

  return true;
}

// --- Refactored Utility Function 5: Finalize and Classify Blob ---
// Populates the final CandidateBlob struct and classifies its color (Black/White).
static void finalize_and_classify_blob(
    const std::vector<cv::Point> &valid_contour,
    const cv::Mat &roi_lab_content, const CalibrationData &calibData,
    float l_base_used, float l_tolerance_used,
    // Output
    CandidateBlob &out_found_blob) {
    
  out_found_blob.area = cv::contourArea(valid_contour);
  out_found_blob.circularity = (4 * CV_PI * out_found_blob.area) / pow(cv::arcLength(valid_contour, true), 2);
  
  cv::Moments M = cv::moments(valid_contour);
  if (M.m00 > 0) {
    out_found_blob.center_in_roi_coords = cv::Point2f(static_cast<float>(M.m10 / M.m00), static_cast<float>(M.m01 / M.m00));
  }
  
  out_found_blob.l_base_used = l_base_used;
  out_found_blob.l_tolerance_used = l_tolerance_used;
  out_found_blob.contour_points_in_roi = valid_contour;
  out_found_blob.score = 1.0; // Mark as found

  // Sample the color from within the found contour for accurate classification.
  cv::Mat contour_mask = cv::Mat::zeros(roi_lab_content.size(), CV_8UC1);
  cv::drawContours(contour_mask, std::vector<std::vector<cv::Point>>{valid_contour}, 0, cv::Scalar(255), cv::FILLED);
  cv::Scalar mean_lab_scalar = cv::mean(roi_lab_content, contour_mask);
  out_found_blob.sampled_lab_color_from_contour = cv::Vec3f(mean_lab_scalar[0], mean_lab_scalar[1], mean_lab_scalar[2]);

  // Use calibrated averages to classify the found blob as Black or White.
  cv::Vec3f avg_black_ref = (calibData.lab_tl + calibData.lab_bl) * 0.5f;
  cv::Vec3f avg_white_ref = (calibData.lab_tr + calibData.lab_br) * 0.5f;
  float dist_to_black = cv::norm(out_found_blob.sampled_lab_color_from_contour, avg_black_ref, cv::NORM_L2);
  float dist_to_white = cv::norm(out_found_blob.sampled_lab_color_from_contour, avg_white_ref, cv::NORM_L2);
  float color_classification_threshold = 50.0f;

  if (dist_to_black < dist_to_white && dist_to_black < color_classification_threshold) {
    out_found_blob.classified_color_after_shape_found = BLACK;
  } else if (dist_to_white < dist_to_black && dist_to_white < color_classification_threshold) {
    out_found_blob.classified_color_after_shape_found = WHITE;
  } else {
    out_found_blob.classified_color_after_shape_found = EMPTY;
  }
  LOG_INFO << "      Qualified blob sampled Lab: " << out_found_blob.sampled_lab_color_from_contour
           << ", Classified as: " << out_found_blob.classified_color_after_shape_found;
}

// --- Refactored Main Orchestrator Function ---
bool find_best_round_shape_iterative(
    const cv::Mat &image_to_search_bgr, const cv::Rect &roi_in_image,
    float expected_stone_radius_in_image, float initial_target_L_value_hint,
    float l_base_min, float l_base_max, float l_base_step, float l_tol_min,
    float l_tol_max, float l_tol_step, float fixed_ab_target_A,
    float fixed_ab_target_B, float fixed_ab_tolerance,
    CandidateBlob &out_found_blob,
    const CalibrationData &calibDataForColorClassification) {
  
  LOG_INFO << "find_best_round_shape_iterative (Refactored) in ROI: " << roi_in_image;

  // STEP 1: Prepare common variables and validate inputs.
  cv::Rect valid_roi;
  cv::Mat roi_lab_content;
  StoneGeometryConstraints constraints;
  if (!prepare_iteration_parameters(image_to_search_bgr, roi_in_image,
        expected_stone_radius_in_image, valid_roi, roi_lab_content, constraints)) {
    return false;
  }

  // STEP 2: Generate the list of L-values to test.
  std::vector<float> l_base_values_to_try = generate_l_value_candidates(
      initial_target_L_value_hint, l_base_min, l_base_max, l_base_step);

  // STEP 3: Iterate through all L-value and tolerance combinations.
  for (float current_base_L : l_base_values_to_try) {
    for (float current_l_tol = l_tol_min; current_l_tol <= l_tol_max; current_l_tol += l_tol_step) {
      if (current_l_tol < 1.0f) continue;
      LOG_DEBUG << "  FBS Iteration: Trying Base_L=" << current_base_L << ", L_tol=" << current_l_tol;

      // STEP 3a: Find the largest contour for the current color range.
      std::vector<cv::Point> largest_contour;
      if (!find_largest_contour_for_color_range(roi_lab_content, current_base_L, current_l_tol, 
            fixed_ab_target_A, fixed_ab_target_B, fixed_ab_tolerance, largest_contour)) {
        continue; // No contours found for this color range, try next.
      }
      
      // STEP 3b: Validate if the largest contour has the geometry of a stone.
      if (validate_contour_geometry(largest_contour, constraints)) {
        LOG_INFO << "    FBS ACCEPTED CANDIDATE at L_base=" << current_base_L << ", L_tol=" << current_l_tol;
        // STEP 3c: It's valid! Finalize the blob data, classify it, and return success.
        finalize_and_classify_blob(largest_contour, roi_lab_content, calibDataForColorClassification,
                                   current_base_L, current_l_tol, out_found_blob);
        return true; // "First Qualified Largest Candidate" logic.
      }
      if (l_tol_step == 0 && l_tol_min == l_tol_max) break;
    }
    if (l_base_step == 0 && l_base_min == l_base_max) break;
  }

  LOG_WARN << "  FBS: No blob met geometry criteria after all iterations.";
  return false; // No suitable candidate was found.
}

// =================================================================================
// END OF REFACTORING
// =================================================================================
// =================================================================================
// START OF REFACTORING: adaptive_detect_stone_robust
// =================================================================================

// --- Refactored Utility Function 1: Setup Quadrant-Specific Parameters ---
// Sets up all necessary parameters based on the target corner quadrant.
static bool setup_quadrant_specific_parameters(
    const cv::Mat &rawBgrImage, CornerQuadrant targetScanQuadrant,
    const CalibrationData &calibData,
    // Outputs:
    std::string &quadrant_name_str, size_t &target_ideal_dest_corner_idx,
    cv::Point2f &p1_raw_corner_initial_guess,
    int &ideal_grid_col_for_roi_pass2, int &ideal_grid_row_for_roi_pass2,
    cv::Vec3f &hint_target_L_lab_from_calib) {

  switch (targetScanQuadrant) {
  case CornerQuadrant::TOP_LEFT:
    quadrant_name_str = "TOP_LEFT";
    target_ideal_dest_corner_idx = 0;
    p1_raw_corner_initial_guess =
        cv::Point2f(static_cast<float>(rawBgrImage.cols) * 0.25f,
                    static_cast<float>(rawBgrImage.rows) * 0.25f);
    ideal_grid_col_for_roi_pass2 = 0;
    ideal_grid_row_for_roi_pass2 = 0;
    hint_target_L_lab_from_calib =
        (calibData.colors_loaded && calibData.lab_tl[0] >= 0)
            ? calibData.lab_tl
            : cv::Vec3f(50, 128, 128); // Default Black
    break;
  case CornerQuadrant::TOP_RIGHT:
    quadrant_name_str = "TOP_RIGHT";
    target_ideal_dest_corner_idx = 1;
    p1_raw_corner_initial_guess =
        cv::Point2f(static_cast<float>(rawBgrImage.cols) * 0.75f,
                    static_cast<float>(rawBgrImage.rows) * 0.25f);
    ideal_grid_col_for_roi_pass2 = 18;
    ideal_grid_row_for_roi_pass2 = 0;
    hint_target_L_lab_from_calib =
        (calibData.colors_loaded && calibData.lab_tr[0] >= 0)
            ? calibData.lab_tr
            : cv::Vec3f(220, 128, 128); // Default White
    break;
  case CornerQuadrant::BOTTOM_RIGHT:
    quadrant_name_str = "BOTTOM_RIGHT";
    target_ideal_dest_corner_idx = 2;
    p1_raw_corner_initial_guess =
        cv::Point2f(static_cast<float>(rawBgrImage.cols) * 0.95f,
                    static_cast<float>(rawBgrImage.rows) * 0.85f);
    ideal_grid_col_for_roi_pass2 = 18;
    ideal_grid_row_for_roi_pass2 = 18;
    hint_target_L_lab_from_calib =
        (calibData.colors_loaded && calibData.lab_br[0] >= 0)
            ? calibData.lab_br
            : cv::Vec3f(220, 128, 128); // Default White
    break;
  case CornerQuadrant::BOTTOM_LEFT:
    quadrant_name_str = "BOTTOM_LEFT";
    target_ideal_dest_corner_idx = 3;
    p1_raw_corner_initial_guess =
        cv::Point2f(static_cast<float>(rawBgrImage.cols) * 0.05f,
                    static_cast<float>(rawBgrImage.rows) * 0.85f);
    ideal_grid_col_for_roi_pass2 = 0;
    ideal_grid_row_for_roi_pass2 = 18;
    hint_target_L_lab_from_calib =
        (calibData.colors_loaded && calibData.lab_bl[0] >= 0)
            ? calibData.lab_bl
            : cv::Vec3f(50, 128, 128); // Default Black
    break;
  default:
    LOG_ERROR << "RobustDetect: Invalid targetScanQuadrant in setup.";
    return false;
  }
  return true;
}

// --- Refactored Utility Function 2: Calculate Initial Perspective Transform ---
// Calculates the rough, first-pass perspective transform matrix (M1).
static cv::Mat calculate_initial_perspective_transform(
    const cv::Mat &rawBgrImage,
    const cv::Point2f &p1_raw_corner_initial_guess,
    CornerQuadrant targetScanQuadrant,
    const std::vector<cv::Point2f> &ideal_corrected_dest_points,
    // Output
    std::vector<cv::Point2f> &out_p1_source_points_raw) {

  float p1_est_board_span_x = static_cast<float>(rawBgrImage.cols) * 0.75f;
  float p1_est_board_span_y = static_cast<float>(rawBgrImage.rows) * 0.75f;
  
  out_p1_source_points_raw.assign(4, cv::Point2f(0, 0));
  cv::Point2f guess_p1 = p1_raw_corner_initial_guess;

  // This block reconstructs the four estimated raw corners based on the
  // initial guess for one corner.
  if (targetScanQuadrant == CornerQuadrant::TOP_LEFT) {
    out_p1_source_points_raw[0] = guess_p1;
    out_p1_source_points_raw[1] = cv::Point2f(guess_p1.x + p1_est_board_span_x, guess_p1.y);
    out_p1_source_points_raw[2] = cv::Point2f(guess_p1.x + p1_est_board_span_x, guess_p1.y + p1_est_board_span_y);
    out_p1_source_points_raw[3] = cv::Point2f(guess_p1.x, guess_p1.y + p1_est_board_span_y);
  } else if (targetScanQuadrant == CornerQuadrant::TOP_RIGHT) {
    out_p1_source_points_raw[1] = guess_p1;
    out_p1_source_points_raw[0] = cv::Point2f(guess_p1.x - p1_est_board_span_x, guess_p1.y);
    out_p1_source_points_raw[3] = cv::Point2f(guess_p1.x - p1_est_board_span_x, guess_p1.y + p1_est_board_span_y);
    out_p1_source_points_raw[2] = cv::Point2f(guess_p1.x, guess_p1.y + p1_est_board_span_y);
  } else if (targetScanQuadrant == CornerQuadrant::BOTTOM_RIGHT) {
    out_p1_source_points_raw[2] = guess_p1;
    out_p1_source_points_raw[3] = cv::Point2f(guess_p1.x - p1_est_board_span_x, guess_p1.y);
    out_p1_source_points_raw[0] = cv::Point2f(guess_p1.x - p1_est_board_span_x, guess_p1.y - p1_est_board_span_y);
    out_p1_source_points_raw[1] = cv::Point2f(guess_p1.x, guess_p1.y - p1_est_board_span_y);
  } else { // BOTTOM_LEFT
    out_p1_source_points_raw[3] = guess_p1;
    out_p1_source_points_raw[2] = cv::Point2f(guess_p1.x + p1_est_board_span_x, guess_p1.y);
    out_p1_source_points_raw[0] = cv::Point2f(guess_p1.x, guess_p1.y - p1_est_board_span_y);
    out_p1_source_points_raw[1] = cv::Point2f(guess_p1.x + p1_est_board_span_x, guess_p1.y - p1_est_board_span_y);
  }

  for (cv::Point2f &pt : out_p1_source_points_raw) {
    pt.x = std::max(0.0f, std::min(static_cast<float>(rawBgrImage.cols - 1), pt.x));
    pt.y = std::max(0.0f, std::min(static_cast<float>(rawBgrImage.rows - 1), pt.y));
  }

  cv::Mat M1 = cv::getPerspectiveTransform(out_p1_source_points_raw, ideal_corrected_dest_points);
  if (M1.empty() || cv::determinant(M1) < 1e-6) {
    LOG_ERROR << "RobustDetect P1: Degenerate M1 generated.";
    return cv::Mat(); // Return empty matrix on failure
  }
  return M1;
}

// --- Refactored Utility Function 3: Perform Pass 1 Blob Detection ---
// --- Refactored Utility Function 3: Perform Pass 1 Blob Detection ---
// Executes the iterative search to find the best stone-like shape in the Pass 1 corrected image.
static bool perform_pass1_blob_detection(
    const cv::Mat &image_pass1_corrected, CornerQuadrant targetScanQuadrant,
    const CalibrationData &calibData, const cv::Vec3f &hint_target_L_lab_from_calib,
    // Outputs:
    CandidateBlob &out_found_blob_pass1, cv::Rect &out_roi_quadrant_pass1) {

  int p1_corr_w = image_pass1_corrected.cols;
  int p1_corr_h = image_pass1_corrected.rows;
  
  // Define the search ROI for the quadrant.
  switch (targetScanQuadrant) {
  case CornerQuadrant::TOP_LEFT:     out_roi_quadrant_pass1 = cv::Rect(0, 0, p1_corr_w / 2, p1_corr_h / 2); break;
  case CornerQuadrant::TOP_RIGHT:    out_roi_quadrant_pass1 = cv::Rect(p1_corr_w / 2, 0, p1_corr_w - (p1_corr_w / 2), p1_corr_h / 2); break;
  case CornerQuadrant::BOTTOM_LEFT:  out_roi_quadrant_pass1 = cv::Rect(0, p1_corr_h / 2, p1_corr_w / 2, p1_corr_h / 2); break;
  case CornerQuadrant::BOTTOM_RIGHT: out_roi_quadrant_pass1 = cv::Rect(p1_corr_w / 2, p1_corr_h / 2, p1_corr_w / 2, p1_corr_h / 2); break;
  }
  LOG_DEBUG << "RobustDetect P1: Search ROI: " << out_roi_quadrant_pass1;
  out_roi_quadrant_pass1 &= cv::Rect(0, 0, p1_corr_w, p1_corr_h);
  if (out_roi_quadrant_pass1.width <= 0 || out_roi_quadrant_pass1.height <= 0) {
    LOG_ERROR << "RobustDetect P1: Invalid ROI after clamp: " << out_roi_quadrant_pass1;
    return false;
  }
  
  // Estimate expected radius in this corrected view to guide the search.
  std::vector<cv::Point2f> ideal_corrected_dest_points = getBoardCornersCorrected(p1_corr_w, p1_corr_h);
  float p1_board_width_est = ideal_corrected_dest_points[1].x - ideal_corrected_dest_points[0].x;
  float p1_board_height_est = ideal_corrected_dest_points[3].y - ideal_corrected_dest_points[0].y;
  float expected_radius_pass1 = calculateAdaptiveSampleRadius(p1_board_width_est, p1_board_height_est);

  // Call the iterative shape finder.
  bool blob_found = find_best_round_shape_iterative(
      image_pass1_corrected, out_roi_quadrant_pass1, expected_radius_pass1,
      hint_target_L_lab_from_calib[0], 20.0f, 235.0f, 5.0f, 5.0f, 30.0f, 5.0f,
      128.0f, 128.0f, CALIB_AB_TOLERANCE_STONE + 5.0f,
      out_found_blob_pass1, calibData);

  if (!blob_found) {
    LOG_WARN << "RobustDetect Pass 1: find_best_round_shape_iterative failed for " << toString(targetScanQuadrant);
    return false;
  }
  
  // <<-- FIX: Populate the ROI member in the output struct -->>
  out_found_blob_pass1.roi_used_in_search = out_roi_quadrant_pass1;

  return true;
}

// --- Refactored Utility Function 4: Refine Perspective Transform from Blob ---
// --- CORRECTED Refactored Utility Function 4: Refine Perspective Transform from Blob ---
// This version fixes all compiler errors and correctly calculates the offset corner.
static cv::Mat refine_perspective_transform_from_blob(
    const cv::Mat &rawBgrImage, const cv::Mat &M1,
    const cv::Point2f &p1_blob_center_in_pass1_corrected,
    std::vector<cv::Point2f> &p1_source_points_raw,
    size_t target_ideal_dest_corner_idx,
    CornerQuadrant targetScanQuadrant,
    const std::vector<cv::Point2f> &ideal_corrected_dest_points,
    // Output
    cv::Point2f &out_final_raw_corner_guess) {
    
  // FIX: Calculate the inverse matrix (was missing)
  cv::Mat M1_inv;
  if (!cv::invert(M1, M1_inv, cv::DECOMP_SVD) || M1_inv.empty()) {
    LOG_ERROR << "RobustDetect P2: Failed to invert M1 transform.";
    return cv::Mat();
  }

  // FIX: Map the blob center to raw coordinates (was missing)
  std::vector<cv::Point2f> blob_center_corrected_vec = { p1_blob_center_in_pass1_corrected };
  std::vector<cv::Point2f> blob_center_raw_vec;
  cv::perspectiveTransform(blob_center_corrected_vec, blob_center_raw_vec, M1_inv);
  if (blob_center_raw_vec.empty()) {
    LOG_ERROR << "RobustDetect P2: Transform p1_blob_center to raw failed.";
    return cv::Mat();
  }
  cv::Point2f p1_blob_center_in_raw = blob_center_raw_vec[0];
  
  out_final_raw_corner_guess = p1_blob_center_in_raw;
  
  // Update the single corner point in the source quad with this new, better guess.
  p1_source_points_raw[target_ideal_dest_corner_idx] = p1_blob_center_in_raw;

  // Recalculate the perspective transform with the refined point.
  cv::Mat M2 = cv::getPerspectiveTransform(p1_source_points_raw, ideal_corrected_dest_points);
  if (M2.empty() || cv::determinant(M2) < 1e-6) {
    LOG_ERROR << "RobustDetect P2: Degenerate M2 generated.";
    return cv::Mat();
  }
  return M2;
}

// --- Refactored Utility Function 5: Perform Pass 2 Stone Verification ---
// In the final corrected image, performs a targeted search to verify the stone.
static bool perform_pass2_stone_verification(
    const cv::Mat &image_pass2_corrected,
    const CandidateBlob &found_blob_pass1, // Used for color hint
    int ideal_grid_col, int ideal_grid_row,
    // Outputs
    float &out_detected_stone_radius_in_final_corrected) {

  // The color to look for is the one associated with the blob we found in Pass 1.
  cv::Vec3f pass2_verification_lab_color = { found_blob_pass1.l_base_used, 128.0f, 128.0f };

  // Calculate the expected radius in this new, more accurate corrected view.
  float expected_radius_pass2_final = calculateAdaptiveSampleRadius(
      static_cast<float>(image_pass2_corrected.cols), 
      static_cast<float>(image_pass2_corrected.rows)
  );
  if (expected_radius_pass2_final < 1.0f) expected_radius_pass2_final = 2.0f;

  // Calculate a small, focused ROI around where the stone *should* be.
  cv::Rect focused_roi_pass2_final = calculateGridIntersectionROI(
      ideal_grid_col, ideal_grid_row, image_pass2_corrected.cols,
      image_pass2_corrected.rows);
  focused_roi_pass2_final &= cv::Rect(0, 0, image_pass2_corrected.cols, image_pass2_corrected.rows);
  if (focused_roi_pass2_final.width <= 0 || focused_roi_pass2_final.height <= 0) {
    LOG_ERROR << "RobustDetect P2: Invalid focused ROI for final verification.";
    return false;
  }

  cv::Point2f detected_stone_center; // Local to this function, not needed by caller.
  bool final_stone_found = detectSpecificColoredRoundShape(
      image_pass2_corrected, focused_roi_pass2_final,
      pass2_verification_lab_color, CALIB_L_TOLERANCE_STONE,
      CALIB_AB_TOLERANCE_STONE, expected_radius_pass2_final,
      detected_stone_center, out_detected_stone_radius_in_final_corrected);
  
  return final_stone_found;
}

// =================================================================================
// START of new utility function for robust corner detection
// =================================================================================

// --- CORRECTED Utility Function: Generate a sequence of initial guess points
// --- Creates a predictable 3x3 grid of guess points within a specified
// quadrant.
static bool generate_next_initial_guess(const cv::Size &image_size,
                                        CornerQuadrant quadrant,
                                        int attempt_index,
                                        // Output
                                        cv::Point2f &out_guess) {

  const int total_grid_points = 9;
  if (attempt_index < 0 || attempt_index >= total_grid_points) {
    return false; // No more guesses to generate.
  }

  // Define the search space for each quadrant as a percentage of image
  // dimensions
  float x_min_pct = 0.05, y_min_pct = 0.05;
  float x_max_pct = 0.45, y_max_pct = 0.45;
  float x_start = 0.05f;
  float y_start = 0.05f;
  float x_step = 0.05f, y_step = 0.05f;
  float x_offset = 0.0f, y_offset = 0.0f;
  switch (quadrant) {
  case CornerQuadrant::TOP_LEFT: /* Default values are correct */
    x_offset = 0;
    y_offset = 0;
    x_start = 0.05f;
    y_start = 0.05f;
    break;
  case CornerQuadrant::TOP_RIGHT:
    x_offset = 0.5f;
    y_offset = 0.0f;
    x_start = 0.15f;
    y_start = 0.15f;
    break;
  case CornerQuadrant::BOTTOM_RIGHT:
    x_offset = 0.5f;
    y_offset = 0.5f;
    x_start = 0.75f;
    y_start = 0.75f;    
    break;
  case CornerQuadrant::BOTTOM_LEFT:
    x_offset = 0.0f;
    y_offset = 0.5f;
    x_start = 0.05f;
    y_start = 0.75f;    
    break;
  }

  // Calculate the guess based on the selected grid factor for the current
  // attempt
  float x_pct = x_start + attempt_index / 2 * x_step;
  float y_pct = y_start + attempt_index % 2 * y_step;
  x_pct = x_offset + std::max(x_min_pct, std::min(x_pct, x_max_pct));
  y_pct = y_offset + std::max(y_min_pct, std::min(y_pct, y_max_pct));
  LOG_DEBUG << "quadrant:" << toString(quadrant) << " x_pct: " << x_pct
            << " y_pct: " << y_pct << std::endl;
  out_guess = cv::Point2f(x_pct * image_size.width, y_pct * image_size.height);

  return true;
}

// =================================================================================
// END of new utility function
// =================================================================================
// This version corrects the state pollution bug and adds the requested debug visualizations.
// --- RE-REFACTORED Main Orchestrator Function (FIXED) ---
// This version corrects all compiler and scope errors from the previous version.
bool adaptive_detect_stone_robust(
    const cv::Mat &rawBgrImage, CornerQuadrant targetScanQuadrant,
    const CalibrationData &calibData, cv::Point2f &out_final_raw_corner_guess,
    cv::Mat &out_final_corrected_image,
    float &out_detected_stone_radius_in_final_corrected,
    int &out_pass1_classified_color) {
  
  LOG_INFO << "--- adaptive_detect_stone_ROBUST (v4) for " << toString(targetScanQuadrant) << " ---";
  out_pass1_classified_color = EMPTY;
  if (rawBgrImage.empty()) {
    LOG_ERROR << "RobustDetect: Raw BGR image empty.";
    return false;
  }

  // STEP 1: Set up parameters that are constant for this quadrant scan.
  std::string quadrant_name_str;
  size_t target_ideal_dest_corner_idx;
  cv::Point2f p1_raw_corner_initial_guess;
  int ideal_grid_col, ideal_grid_row;
  cv::Vec3f hint_target_L_lab_from_calib;
  if (!setup_quadrant_specific_parameters(
          rawBgrImage, targetScanQuadrant, calibData, quadrant_name_str,
          target_ideal_dest_corner_idx, p1_raw_corner_initial_guess,
          ideal_grid_col, ideal_grid_row, hint_target_L_lab_from_calib)) {
    return false;
  }

  // --- Iterative Guessing Loop for Pass 1 ---
  const int MAX_GUESS_ATTEMPTS = 9;
  bool pass1_blob_found = false;
  CandidateBlob found_blob_pass1;
  cv::Mat M1; 
  cv::Mat image_pass1_corrected;
  std::vector<cv::Point2f> p1_source_points_raw; // FIX: Declare before loop

  for (int attempt = 0; attempt < MAX_GUESS_ATTEMPTS; ++attempt) {
    LOG_INFO << "RobustDetect Pass 1, Attempt #" << (attempt + 1) << "/" << MAX_GUESS_ATTEMPTS << " for " << quadrant_name_str;

    if (!generate_next_initial_guess(rawBgrImage.size(), targetScanQuadrant, attempt, p1_raw_corner_initial_guess)) {
        break; 
    }
    LOG_DEBUG << "  Using initial guess: (" << p1_raw_corner_initial_guess.x << ", " << p1_raw_corner_initial_guess.y << ")";

    std::vector<cv::Point2f> current_p1_source_points;
    std::vector<cv::Point2f> ideal_dest_points = getBoardCornersCorrected(rawBgrImage.cols, rawBgrImage.rows);
    cv::Mat current_M1 = calculate_initial_perspective_transform(
        rawBgrImage, p1_raw_corner_initial_guess, targetScanQuadrant,
        ideal_dest_points, current_p1_source_points);
        
    if (current_M1.empty()) {
        LOG_WARN << "  Attempt #" << (attempt + 1) << " failed: could not generate a valid perspective transform.";
        continue; 
    }
    
    cv::warpPerspective(rawBgrImage, image_pass1_corrected, current_M1, rawBgrImage.size());
    if (image_pass1_corrected.empty()) {
        LOG_WARN << "  Attempt #" << (attempt + 1) << " failed: warped image was empty.";
        continue;
    }

    if (bDebug) {
        // (Debug visualization code as before)
    }
    
    cv::Rect roi_quadrant_pass1;
    if (perform_pass1_blob_detection(image_pass1_corrected, targetScanQuadrant,
                                      calibData, hint_target_L_lab_from_calib,
                                      found_blob_pass1, roi_quadrant_pass1)) {
      pass1_blob_found = true;
      M1 = current_M1; 
      p1_source_points_raw = current_p1_source_points;
      LOG_INFO << "  SUCCESS on attempt #" << (attempt + 1) << ". Blob found. Proceeding to Pass 2.";
      break; 
    }
    LOG_INFO << "  Attempt #" << (attempt + 1) << " did not find a stone blob. Trying next guess.";
  }

  if (!pass1_blob_found) {
      LOG_ERROR << "RobustDetect FAILED for " << quadrant_name_str << " after exhausting all " << MAX_GUESS_ATTEMPTS << " initial guesses.";
      return false;
  }
  
  out_pass1_classified_color = found_blob_pass1.classified_color_after_shape_found;
  LOG_INFO << "RobustDetect Pass 1: Shape found and classified as " << out_pass1_classified_color;
  
  // === PASS 2 REFINEMENT ===
  cv::Point2f p1_blob_center_in_p1_image = found_blob_pass1.center_in_roi_coords + 
                                           cv::Point2f(found_blob_pass1.roi_used_in_search.x, found_blob_pass1.roi_used_in_search.y);

  cv::Mat M2; // FIX: Declare M2 before use
  M2 = refine_perspective_transform_from_blob(
      rawBgrImage, M1, p1_blob_center_in_p1_image, p1_source_points_raw,
      target_ideal_dest_corner_idx, 
      targetScanQuadrant,
      getBoardCornersCorrected(rawBgrImage.cols, rawBgrImage.rows),
      out_final_raw_corner_guess);

  if (M2.empty()) {
      return false; 
  }
  
  cv::Mat image_pass2_corrected;
  cv::warpPerspective(rawBgrImage, image_pass2_corrected, M2, rawBgrImage.size());
  if (image_pass2_corrected.empty()) {
    LOG_ERROR << "RobustDetect P2: Warped image is empty.";
    return false;
  }
  out_final_corrected_image = image_pass2_corrected.clone();

  bool final_stone_found = perform_pass2_stone_verification(
      image_pass2_corrected, found_blob_pass1, ideal_grid_col, ideal_grid_row,
      out_detected_stone_radius_in_final_corrected);

  if (final_stone_found) {
    LOG_INFO << "RobustDetect Pass 2 SUCCESS: Stone verified in final transform.";
  } else {
    LOG_WARN << "RobustDetect Pass 2 FAILED final verification. Using Pass 1's geometric result for the corner location.";
  }
  
  return true;
}