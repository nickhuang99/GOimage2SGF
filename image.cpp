#include "common.h" // Includes logger.h
#include "logger.h"
#include <algorithm>
#include <cassert>
#include <cmath>
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
const double MIN_STONE_AREA_RATIO = 0.03;
const double MAX_STONE_AREA_RATIO = 0.85; // MODIFIED (was 0.70)
const double MIN_STONE_CIRCULARITY_WHITE = 0.65;
const double MIN_STONE_CIRCULARITY_BLACK =
    0.55; // More lenient for black stones
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

  if (detectSpecificColoredRoundShape(
          corrected_bgr_image, roi, ref_black_lab, CALIB_L_TOLERANCE_STONE,
          CALIB_AB_TOLERANCE_STONE, detected_center, detected_radius)) {
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
          CALIB_AB_TOLERANCE_STONE, detected_center, detected_radius)) {
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
                                     cv::Point2f &detectedCenter,
                                     float &detectedRadius) {

  LOG_DEBUG << "Detecting specific colored round shape in ROI: {x:"
            << regionOfInterest.x << ",y:" << regionOfInterest.y
            << ",w:" << regionOfInterest.width
            << ",h:" << regionOfInterest.height
            << "} Target Lab: " << expectedAvgLabColor[0] << ","
            << expectedAvgLabColor[1] << "," << expectedAvgLabColor[2]
            << " L_tol: " << l_tolerance << ", AB_tol: " << ab_tolerance
            << std::endl;

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

  if (Logger::getGlobalLogLevel() >= LogLevel::DEBUG) {
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

  double roiArea = static_cast<double>(roi.width * roi.height);
  // <<< MODIFIED: Use new constants for area and circularity >>>
  double minStoneAreaInRoi = roiArea * MIN_STONE_AREA_RATIO;
  double maxStoneAreaInRoi = roiArea * MAX_STONE_AREA_RATIO;
  double minCircularity = (expectedAvgLabColor[0] < 100.0f)
                              ? MIN_STONE_CIRCULARITY_BLACK
                              : MIN_STONE_CIRCULARITY_WHITE;

  LOG_DEBUG << "  For ROI at (" << roi.x << "," << roi.y
            << "): ROI Area=" << roiArea
            << ", MinStoneAreaInRoi=" << minStoneAreaInRoi
            << ", MaxStoneAreaInRoi=" << maxStoneAreaInRoi
            << ", MinCircularity=" << minCircularity << std::endl;

  std::vector<cv::Point> bestContour;
  double bestContourScore = 0.0f;
  int contour_idx = 0;
  cv::Mat roi_contour_vis_canvas;
  if (Logger::getGlobalLogLevel() >= LogLevel::DEBUG) {
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

    if (area < minStoneAreaInRoi || area > maxStoneAreaInRoi) {
      LOG_DEBUG
          << "      -> Contour " << contour_idx << " (ROI " << roi.x << ","
          << roi.y << ") REJECTED by area " << area
          << (area < minStoneAreaInRoi
                  ? " (too small, min=" + Num2Str(minStoneAreaInRoi).str() + ")"
                  : " (too large, max=" + Num2Str(maxStoneAreaInRoi).str() +
                        ")")
          << std::endl;
      if (Logger::getGlobalLogLevel() >= LogLevel::DEBUG)
        cv::drawContours(roi_contour_vis_canvas,
                         std::vector<std::vector<cv::Point>>{contour}, -1,
                         cv::Scalar(0, 165, 255), 1);
      continue;
    }
    if (circularity < minCircularity) {
      LOG_DEBUG << "      -> Contour " << contour_idx << " (ROI " << roi.x
                << "," << roi.y << ") REJECTED by circularity " << circularity
                << " (min_circ=" << minCircularity << ")" << std::endl;
      if (Logger::getGlobalLogLevel() >= LogLevel::DEBUG) {
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
    if (Logger::getGlobalLogLevel() >= LogLevel::DEBUG)
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

  if (Logger::getGlobalLogLevel() >= LogLevel::DEBUG && !contours.empty()) {
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

bool detectFourCornersGoBoard(
    const cv::Mat &corrected_bgr_image,
    std::vector<cv::Point2f> &detected_raw_centers_tl_tr_br_bl,
    std::vector<float> &detected_raw_radii_tl_tr_br_bl) {

  LOG_INFO
      << "Detecting four corner Go board stones using config-refined method..."
      << std::endl;
  detected_raw_centers_tl_tr_br_bl.assign(4, cv::Point2f(-1, -1));
  detected_raw_radii_tl_tr_br_bl.assign(4, -1.0f);

  if (corrected_bgr_image.empty()) {
    LOG_ERROR << "Input image is empty in detectFourCornersGoBoard."
              << std::endl;
    return false;
  }

  CalibrationData calib_data = loadCalibrationData(CALIB_CONFIG_PATH);
  if (!calib_data.corners_loaded || !calib_data.colors_loaded ||
      !calib_data.dimensions_loaded) {
    LOG_ERROR
        << "Essential calibration data (corners, colors, or dimensions) not "
           "loaded. Cannot perform refined detection for four corners."
        << std::endl;
    return false;
  }

  if (corrected_bgr_image.cols != calib_data.image_width ||
      corrected_bgr_image.rows != calib_data.image_height) {
    LOG_ERROR << "Input image dimensions (" << corrected_bgr_image.cols << "x"
              << corrected_bgr_image.rows
              << ") do not match calibration config dimensions ("
              << calib_data.image_width << "x" << calib_data.image_height
              << ") in detectFourCornersGoBoard." << std::endl;
    return false;
  }

  if (Logger::getGlobalLogLevel() >= LogLevel::DEBUG) {
    cv::imshow("detectFourCorners - Corrected BGR for ROI calc",
               corrected_bgr_image);
    cv::waitKey(1);
    // cv::destroyWindow("detectFourCorners - Corrected BGR for ROI calc"); //
    // Can be annoying if stepping through
  }

  bool all_found = true;
  cv::Point2f detected_center_corrected;
  float detected_radius_corrected;

  const int grid_lines = 19;
  std::vector<std::pair<int, int>> corner_coords_grid = {
      {0, 0},
      {grid_lines - 1, 0},
      {grid_lines - 1, grid_lines - 1},
      {0, grid_lines - 1}};
  std::vector<cv::Vec3f> corner_lab_refs = {
      calib_data.lab_tl, calib_data.lab_tr, calib_data.lab_br,
      calib_data.lab_bl};
  std::vector<std::string> corner_names = {"TL(Black)", "TR(White)",
                                           "BR(White)", "BL(Black)"};

  for (size_t i = 0; i < 4; ++i) {
    int target_c = corner_coords_grid[i].first;
    int target_r = corner_coords_grid[i].second;

    cv::Rect roi_corrected = calculateGridIntersectionROI(
        target_c, target_r, corrected_bgr_image.cols, corrected_bgr_image.rows,
        grid_lines);
    roi_corrected &=
        cv::Rect(0, 0, corrected_bgr_image.cols, corrected_bgr_image.rows);

    LOG_DEBUG << "  For " << corner_names[i] << ": Target Grid(" << target_r
              << "," << target_c
              << "), ROI in Corrected Img: {x:" << roi_corrected.x
              << ",y:" << roi_corrected.y << ",w:" << roi_corrected.width
              << ",h:" << roi_corrected.height << "}"
              << ", Using Lab Ref: " << corner_lab_refs[i][0] << ","
              << corner_lab_refs[i][1] << "," << corner_lab_refs[i][2]
              << std::endl;

    if (roi_corrected.width <= 0 || roi_corrected.height <= 0) {
      all_found = false;
      LOG_ERROR << "    ROI for " << corner_names[i]
                << " is invalid after clamping. Skipping detection."
                << std::endl;
      continue;
    }

    if (detectSpecificColoredRoundShape(
            corrected_bgr_image, roi_corrected, corner_lab_refs[i],
            CALIB_L_TOLERANCE_STONE, CALIB_AB_TOLERANCE_STONE,
            detected_center_corrected, detected_radius_corrected)) {

      std::vector<cv::Point2f> point_to_transform = {detected_center_corrected};

      detected_raw_centers_tl_tr_br_bl[i] = point_to_transform[0];

      cv::Point2f p1_roi_edge_corrected =
          cv::Point2f(roi_corrected.x, roi_corrected.y);
      cv::Point2f p2_roi_edge_corrected =
          cv::Point2f(roi_corrected.x + roi_corrected.width, roi_corrected.y);
      std::vector<cv::Point2f> roi_edge_points_corrected = {
          p1_roi_edge_corrected, p2_roi_edge_corrected};

      float roi_width_raw =
          cv::norm(roi_edge_points_corrected[0] - roi_edge_points_corrected[1]);
      float scale_factor_at_roi =
          (roi_corrected.width > 0.001f)
              ? (roi_width_raw / static_cast<float>(roi_corrected.width))
              : 1.0f;
      detected_raw_radii_tl_tr_br_bl[i] =
          detected_radius_corrected * scale_factor_at_roi;

      LOG_DEBUG << "    Found " << corner_names[i]
                << ": Corrected Center=" << detected_center_corrected.x << ","
                << detected_center_corrected.y
                << ", Raw Center=" << detected_raw_centers_tl_tr_br_bl[i].x
                << "," << detected_raw_centers_tl_tr_br_bl[i].y
                << ", Corrected Radius=" << detected_radius_corrected
                << ", Approx Raw Radius=" << detected_raw_radii_tl_tr_br_bl[i]
                << std::endl;
    } else {
      all_found = false;
      LOG_WARN << "    Failed to find " << corner_names[i]
               << " stone using specific Lab target." << std::endl;
    }
  }

  if (all_found) {
    LOG_INFO
        << "Refined config-guided detection successful for all four corners."
        << std::endl;
    if (Logger::getGlobalLogLevel() >= LogLevel::DEBUG) {
      LOG_DEBUG << "  Raw corner centers (TL, TR, BR, BL) and radii:"
                << std::endl;
      for (size_t i = 0; i < detected_raw_centers_tl_tr_br_bl.size(); ++i) {
        LOG_DEBUG << "    " << corner_names[i] << ": Center=("
                  << detected_raw_centers_tl_tr_br_bl[i].x << ","
                  << detected_raw_centers_tl_tr_br_bl[i].y
                  << "), Radius=" << detected_raw_radii_tl_tr_br_bl[i]
                  << std::endl;
      }
      cv::Mat debug_raw_img_final = corrected_bgr_image.clone();
      for (size_t i = 0; i < 4; ++i) {
        if (detected_raw_centers_tl_tr_br_bl[i].x >= 0) {
          cv::circle(debug_raw_img_final, detected_raw_centers_tl_tr_br_bl[i],
                     static_cast<int>(
                         std::max(5.0f, detected_raw_radii_tl_tr_br_bl[i])),
                     cv::Scalar(0, 255, 0), 2);
          cv::putText(debug_raw_img_final, corner_names[i].substr(0, 2),
                      detected_raw_centers_tl_tr_br_bl[i] - cv::Point2f(10, 10),
                      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255),
                      1);
        }
      }
      cv::imshow("detectFourCornersGoBoard - Final Refined Raw Points",
                 debug_raw_img_final);
      cv::waitKey(0);
      cv::destroyWindow("detectFourCornersGoBoard - Final Refined Raw Points");
    }
    return true;
  } else {
    LOG_ERROR
        << "Failed to pinpoint all four corner stones using refined method."
        << std::endl;
    return false;
  }
}