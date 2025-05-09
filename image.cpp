#include "common.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <ostream>
#include <regex> // Include the regex library
#include <set>
#include <vector>

using namespace std;
using namespace cv;
struct Line {
  double value; // y for horizontal, x for vertical
  double angle;
};

bool compareLines(const Line &a, const Line &b) { return a.value < b.value; }

// Helper structure for Lab classification
// struct ClusterInfoLab {
//   int index;
//   float l, a, b; // Lab components
//   float black_score;
//   float white_score;
//   float board_score;

//   ClusterInfoLab(int idx, float l_val, float a_val, float b_val)
//       : index(idx), l(l_val), a(a_val), b(b_val),
//         black_score(std::numeric_limits<float>::max()),
//         white_score(std::numeric_limits<float>::max()),
//         board_score(std::numeric_limits<float>::max()) {}
// };

// --- Existing static helper (slightly modified) ---
// Tries to load corners and checks dimensions. Returns true on success, false
// otherwise.
static bool loadCornersAndCheckDims(
    const std::string &config_path, int current_width, int current_height,
    std::vector<cv::Point2f> &board_corners_out) { // Renamed param
  // --- Load corners using the new function ---
  std::vector<cv::Point2f> loaded_corners =
      loadCornersFromConfigFile(config_path);
  if (loaded_corners.empty()) {
    // Error message printed by loadCornersFromConfigFile in debug mode
    return false; // Loading corners failed
  }

  // --- Now, perform the dimension check (requires reading config file again,
  // or enhancing loadCornersFromConfigFile) --- For simplicity now, let's
  // re-read just for dimensions. A more optimal way would be to have
  // loadCornersFromConfigFile also return dimensions, maybe via out-params or a
  // struct. Re-reading for now:
  std::ifstream configFile(config_path);
  std::map<std::string, std::string> config_data;
  std::string line;
  if (!configFile.is_open())
    return false; // Should not happen if corners loaded, but check anyway

  try {
    while (getline(configFile, line)) {
      // Simplified parsing just for dimensions
      line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
      line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);
      if (line.empty() || line[0] == '#')
        continue;
      size_t equals_pos = line.find('=');
      if (equals_pos != std::string::npos) {
        std::string key = line.substr(0, equals_pos);
        std::string value = line.substr(equals_pos + 1);
        // Quick trim
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        config_data[key] = value;
      }
    }
    configFile.close();

    if (config_data.count("ImageWidth") && config_data.count("ImageHeight")) {
      int config_width = std::stoi(config_data["ImageWidth"]);
      int config_height = std::stoi(config_data["ImageHeight"]);

      if (config_width == current_width && config_height == current_height) {
        if (bDebug)
          std::cout
              << "Debug (loadCornersAndCheckDims): Config dimensions match "
                 "current image ("
              << current_width << "x" << current_height << ")." << std::endl;
        board_corners_out =
            loaded_corners; // Assign loaded corners to output parameter
        return true;        // Success!
      } else {
        if (bDebug)
          std::cerr
              << "Warning (loadCornersAndCheckDims): Config file dimensions ("
              << config_width << "x" << config_height
              << ") do not match image dimensions (" << current_width << "x"
              << current_height << "). Ignoring config corners." << std::endl;
        return false; // Dimension mismatch
      }
    } else {
      if (bDebug)
        std::cerr << "Warning (loadCornersAndCheckDims): Config file missing "
                     "ImageWidth or ImageHeight."
                  << std::endl;
      return false; // Missing dimensions
    }
  } catch (const std::exception &e) {
    if (bDebug)
      std::cerr
          << "Warning (loadCornersAndCheckDims): Error parsing dimensions "
             "from config '"
          << config_path << "': " << e.what() << std::endl;
    if (configFile.is_open())
      configFile.close();
    return false; // Error during dimension check
  }
  return false; // Should not be reached
}

static int classifyIntersectionByCalibration(
    const cv::Vec3f &intersection_lab_color,
    const cv::Vec3f &avg_black_calib, // Pre-calculated average black from calib
    const cv::Vec3f &avg_white_calib, // Pre-calculated average white from calib
    const cv::Vec3f &board_calib_lab) // Direct board color from calib
{
  const float MAX_DIST_STONE_CALIB =
      35.0f; // Max Lab distance for a sample to be considered a stone matching
             // its calibrated color
  const float MAX_DIST_BOARD_CALIB =
      45.0f; // Max Lab distance for a sample to be considered board matching
             // its calibrated color

  float dist_b = cv::norm(intersection_lab_color, avg_black_calib, cv::NORM_L2);
  float dist_w = cv::norm(intersection_lab_color, avg_white_calib, cv::NORM_L2);
  float dist_empty =
      cv::norm(intersection_lab_color, board_calib_lab, cv::NORM_L2);

  float min_dist = std::min({dist_b, dist_w, dist_empty});
  int classification = 0; // Default to empty (board)

  if (min_dist == dist_b && dist_b < MAX_DIST_STONE_CALIB) {
    classification = 1; // Black
  } else if (min_dist == dist_w && dist_w < MAX_DIST_STONE_CALIB) {
    classification = 2; // White
  } else if (min_dist == dist_empty && dist_empty < MAX_DIST_BOARD_CALIB) {
    classification = 0; // Empty
  } else {
    // Uncertain or too far from all references, classify as empty
    classification = 0;
    if (bDebug) {
      // This case might be interesting to log if min_dist wasn't one of the
      // three, or if it matched but exceeded its threshold. For now, just
      // default to empty.
    }
  }

  if (bDebug && false) { // Set to true for very verbose per-intersection
                         // classification details
    std::cout << "    Sample Lab: " << intersection_lab_color
              << " D(B):" << std::setw(5) << dist_b << " D(W):" << std::setw(5)
              << dist_w << " D(E):" << std::setw(5) << dist_empty
              << " -> Class: " << classification << std::endl;
  }

  return classification;
}

// --- NEW HELPER FUNCTION to load corners from config ---
static bool loadCornersFromConfig(const std::string &config_path,
                                  int current_width, int current_height,
                                  std::vector<cv::Point2f> &board_corners) {
  ifstream configFile(config_path);
  if (!configFile.is_open()) {
    if (bDebug)
      cout << "Debug: Config file '" << config_path << "' not found." << endl;
    return false;
  }

  if (bDebug)
    cout << "Debug: Found config file: " << config_path
         << ". Attempting to parse." << endl;
  map<string, string> config_data;
  string line;
  try {
    while (getline(configFile, line)) {
      line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
      line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);
      if (line.empty() || line[0] == '#')
        continue;

      size_t equals_pos = line.find('=');
      if (equals_pos != string::npos) {
        string key = line.substr(0, equals_pos);
        string value = line.substr(equals_pos + 1);
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        config_data[key] = value;
      }
    }
    configFile.close();

    if (config_data.count("ImageWidth") && config_data.count("ImageHeight")) {
      int config_width = std::stoi(config_data["ImageWidth"]);
      int config_height = std::stoi(config_data["ImageHeight"]);

      if (config_width == current_width && config_height == current_height) {
        if (bDebug)
          cout << "Debug: Config dimensions match current image ("
               << current_width << "x" << current_height << ")." << endl;

        if (config_data.count("TL_X_PX") && config_data.count("TL_Y_PX") &&
            config_data.count("TR_X_PX") && config_data.count("TR_Y_PX") &&
            config_data.count("BL_X_PX") && config_data.count("BL_Y_PX") &&
            config_data.count("BR_X_PX") && config_data.count("BR_Y_PX")) {

          Point2f tl(std::stof(config_data["TL_X_PX"]),
                     std::stof(config_data["TL_Y_PX"]));
          Point2f tr(std::stof(config_data["TR_X_PX"]),
                     std::stof(config_data["TR_Y_PX"]));
          Point2f bl(std::stof(config_data["BL_X_PX"]),
                     std::stof(config_data["BL_Y_PX"]));
          Point2f br(std::stof(config_data["BR_X_PX"]),
                     std::stof(config_data["BR_Y_PX"]));

          board_corners = {tl, tr, br, bl};
          if (bDebug)
            cout << "Debug: Successfully loaded corners from config file."
                 << endl;
          return true;
        } else {
          if (bDebug)
            cerr << "Warning: Config file missing one or more pixel coordinate "
                    "keys (_PX)."
                 << endl;
        }
      } else {
        if (bDebug)
          cerr << "Warning: Config file dimensions (" << config_width << "x"
               << config_height << ") do not match image dimensions ("
               << current_width << "x" << current_height
               << "). Ignoring config." << endl;
      }
    } else {
      if (bDebug)
        cerr << "Warning: Config file missing ImageWidth or ImageHeight."
             << endl;
    }
  } catch (const std::exception &e) {
    if (bDebug)
      cerr << "Warning: Error parsing config file '" << config_path
           << "': " << e.what() << ". Using defaults." << endl;
    if (configFile.is_open())
      configFile.close();
    return false; // Ensure returns false if parsing fails
  }
  return false; // Default return if conditions not met
}

CalibrationData loadCalibrationData(const std::string &config_path) {
  CalibrationData data;
  std::ifstream configFile(config_path);
  if (!configFile.is_open()) {
    if (bDebug)
      std::cerr << "Debug (loadCalibrationData): Config file not found: "
                << config_path << std::endl;
    return data;
  }

  if (bDebug)
    std::cout << "Debug (loadCalibrationData): Parsing config file: "
              << config_path << std::endl;
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
      if (bDebug)
        std::cerr << "Warning (loadCalibrationData): Invalid line format (no "
                     "'=') at line "
                  << line_num << ": " << line << std::endl;
    }
  }
  configFile.close();

  try {
    // Parse Dimensions (as in your file)
    if (config_map.count("ImageWidth") && config_map.count("ImageHeight")) {
      data.image_width = std::stoi(config_map["ImageWidth"]);
      data.image_height = std::stoi(config_map["ImageHeight"]);
      data.dimensions_loaded = true;
      if (bDebug)
        std::cout << "  Debug: Loaded Dimensions: " << data.image_width << "x"
                  << data.image_height << std::endl;
    } else {
      if (bDebug)
        std::cerr << "  Warning: ImageWidth or ImageHeight missing from config."
                  << std::endl;
    }

    // Parse Corners (_PX) (as in your file)
    if (config_map.count("TL_X_PX") &&
        config_map.count("TL_Y_PX") && /* ... all 8 corner keys ... */
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
      if (bDebug)
        std::cout << "  Debug: Loaded Corners (TL, TR, BR, BL): " << tl << ", "
                  << tr << ", " << br << ", " << bl << std::endl;
    } else {
      if (bDebug)
        std::cerr << "  Warning: One or more corner pixel keys (_PX) missing."
                  << std::endl;
    }

    // Parse Corner Colors (L, A, B) (as in your file)
    if (config_map.count("TL_L") && config_map.count("TL_A") &&
        config_map.count("TL_B") &&
        /* ... all 12 corner color keys ... */
        config_map.count("TR_L") && config_map.count("TR_A") &&
        config_map.count("TR_B") && config_map.count("BL_L") &&
        config_map.count("BL_A") && config_map.count("BL_B") &&
        config_map.count("BR_L") && config_map.count("BR_A") &&
        config_map.count("BR_B")) {
      data.lab_tl[0] = std::stof(config_map["TL_L"]);
      data.lab_tl[1] = std::stof(config_map["TL_A"]);
      data.lab_tl[2] = std::stof(config_map["TL_B"]);
      data.lab_tr[0] = std::stof(config_map["TR_L"]);
      data.lab_tr[1] = std::stof(config_map["TR_A"]);
      data.lab_tr[2] = std::stof(config_map["TR_B"]);
      data.lab_bl[0] = std::stof(config_map["BL_L"]);
      data.lab_bl[1] = std::stof(config_map["BL_A"]);
      data.lab_bl[2] = std::stof(config_map["BL_B"]);
      data.lab_br[0] = std::stof(config_map["BR_L"]);
      data.lab_br[1] = std::stof(config_map["BR_A"]);
      data.lab_br[2] = std::stof(config_map["BR_B"]);
      data.colors_loaded = true; // For corner stone colors
      if (bDebug) {              /* ... debug print for corner colors ... */
      }
    } else {
      if (bDebug)
        std::cerr
            << "  Warning: One or more corner Lab color keys (L/A/B) missing."
            << std::endl;
    }

    // --- NEW: Parse Average Board Color ---
    if (config_map.count("BOARD_L_AVG") && config_map.count("BOARD_A_AVG") &&
        config_map.count("BOARD_B_AVG")) {
      data.lab_board_avg[0] = std::stof(config_map["BOARD_L_AVG"]);
      data.lab_board_avg[1] = std::stof(config_map["BOARD_A_AVG"]);
      data.lab_board_avg[2] = std::stof(config_map["BOARD_B_AVG"]);
      data.board_color_loaded = true;
      if (bDebug) {
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "  Debug: Loaded Average Board Lab: ["
                  << data.lab_board_avg[0] << "," << data.lab_board_avg[1]
                  << "," << data.lab_board_avg[2] << "]" << std::endl;
      }
    } else {
      if (bDebug)
        std::cerr << "  Warning: Average Board Lab color keys "
                     "(BOARD_L/A/B_AVG) missing."
                  << std::endl;
    }
    // --- End NEW ---

  } catch (const std::invalid_argument &ia) {
    std::cerr
        << "Error (loadCalibrationData): Invalid number format in config file '"
        << config_path << "'. " << ia.what() << std::endl;
    data.corners_loaded = false;
    data.colors_loaded = false;
    data.dimensions_loaded = false;
    data.board_color_loaded = false;
  } catch (const std::out_of_range &oor) {
    std::cerr
        << "Error (loadCalibrationData): Number out of range in config file '"
        << config_path << "'. " << oor.what() << std::endl;
    data.corners_loaded = false;
    data.colors_loaded = false;
    data.dimensions_loaded = false;
    data.board_color_loaded = false;
  } catch (const std::exception &e) {
    std::cerr
        << "Error (loadCalibrationData): Generic error parsing config file '"
        << config_path << "': " << e.what() << std::endl;
    data.corners_loaded = false;
    data.colors_loaded = false;
    data.dimensions_loaded = false;
    data.board_color_loaded = false;
  }
  return data;
}

// Function to find the corners of the Go board.
// Attempts to load from config file first, otherwise uses hardcoded
// percentages.
std::vector<cv::Point2f> getBoardCorners(const cv::Mat &inputImage) {
  std::vector<cv::Point2f> board_corners_result;
  bool use_config_corners = false;

  int current_width = inputImage.cols;
  int current_height = inputImage.rows;
  CalibrationData calib_data = loadCalibrationData(
      CALIB_CONFIG_PATH); // CALIB_CONFIG_PATH defined in camera.cpp

  if (calib_data.corners_loaded && calib_data.dimensions_loaded &&
      calib_data.image_width == current_width &&
      calib_data.image_height == current_height) {
    if (bDebug)
      std::cout << "Debug (getBoardCorners): Using corners from config file "
                   "(dimensions match)."
                << std::endl;
    board_corners_result = calib_data.corners;
    use_config_corners = true;
  } else if (calib_data.corners_loaded && calib_data.dimensions_loaded) {
    if (bDebug)
      std::cerr << "Warning (getBoardCorners): Config dimensions ("
                << calib_data.image_width << "x" << calib_data.image_height
                << ") mismatch image (" << current_width << "x"
                << current_height << "). Ignoring config corners." << std::endl;
  }
  if (!use_config_corners) {
    if (bDebug) {
      std::cout << "Debug (getBoardCorners): Falling back to hardcoded "
                   "percentage values for corners."
                << std::endl;
    }
    board_corners_result = {cv::Point2f(current_width * 20.0f / 100.0f,
                                        current_height * 8.0f / 100.0f),
                            cv::Point2f(current_width * 73.0f / 100.0f,
                                        current_height * 5.0f / 100.0f),
                            cv::Point2f(current_width * 97.0f / 100.0f,
                                        current_height * 45.0f / 100.0f),
                            cv::Point2f(current_width * 5.0f / 100.0f,
                                        current_height * 52.0f / 100.0f)};
  }
  return board_corners_result;
}

vector<Point2f> getBoardCornersCorrected(const Mat &image) {
  int width = image.cols;
  int height = image.rows;
  int dest_percent = 15; // Start with 15 and adjust if needed
  vector<Point2f> output_corners = {
      Point2f(width * dest_percent / 100.0f, height * dest_percent / 100.0f),
      Point2f(width * (100 - dest_percent) / 100.0f,
              height * dest_percent / 100.0f),
      Point2f(width * (100 - dest_percent) / 100.0f,
              height * (100 - dest_percent) / 100.0f),
      Point2f(width * dest_percent / 100.0f,
              height * (100 - dest_percent) / 100.0f)};
  return output_corners;
}
Mat correctPerspective(const Mat &image) {
  int width = image.cols;
  int height = image.rows;

  vector<Point2f> input_corners = getBoardCorners(image);

  vector<Point2f> output_corners = getBoardCornersCorrected(image);
  Mat perspective_matrix =
      getPerspectiveTransform(input_corners, output_corners);
  Mat corrected_image;
  warpPerspective(image, corrected_image, perspective_matrix,
                  Size(width, height));
  if (bDebug && false) {
    // Draw the input and output corners on the original and corrected images
    for (const auto &corner : input_corners) {
      circle(image, corner, 5, Scalar(0, 0, 255), -1); // Red circles
    }
    for (const auto &corner : output_corners) {
      circle(corrected_image, corner, 5, Scalar(0, 255, 0),
             -1); // Green circles
    }
    imshow("Original Image with Input Corners", image);
    imshow("Corrected Image with Output Corners", corrected_image);
    waitKey(0);
  }
  return corrected_image;
}

// 1. Preprocessing Function
Mat preprocessImage(const Mat &image, bool bDebug) {
  Mat gray, blurred, edges;
  cvtColor(image, gray, COLOR_BGR2GRAY);
  GaussianBlur(gray, blurred, Size(5, 5), 0); // Or Size(7, 7)

  if (bDebug && false) {
    imshow("Blurred", blurred);
    waitKey(0);
  }

  adaptiveThreshold(blurred, edges, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
                    THRESH_BINARY, 11, 2);

  if (bDebug && false) {
    imshow("Edges (Before Morph)", edges);
    waitKey(0);
  }

  // Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
  // morphologyEx(edges, edges, MORPH_CLOSE, kernel, Point(-1, -1), 1);
  // if (bDebug) {
  //   imshow("Morph", edges);
  //   waitKey(0);
  // }
  Mat result;
  Canny(edges, result, 50, 150, 3);
  if (bDebug && false) {
    imshow("Canny", result);
  }
  return result;
}

// 2. Line Segment Detection Function (Refactored)
vector<Vec4i> detectLineSegments(const Mat &edges, bool bDebug) {
  int width = edges.cols;
  int height = edges.rows;

  // 1. Get Board Corners (You'll need to implement this correctly)
  // Use getBoardCornersCorrected to define the region of interest for
  // HoughLinesP
  vector<Point2f> board_corners = getBoardCornersCorrected(edges);
  float board_height = board_corners[2].y - board_corners[0].y;
  float board_width = board_corners[1].x - board_corners[0].x;

  // Define a single mask for the entire board area
  // You might want to slightly expand this rectangle beyond the strict corners
  // to ensure lines right at the edge are detected.
  int margin = 10; // Pixels to expand the mask
  Rect board_rect(static_cast<int>(board_corners[0].x) - margin,
                  static_cast<int>(board_corners[0].y) - margin,
                  static_cast<int>(board_width) + 2 * margin,
                  static_cast<int>(board_height) + 2 * margin);

  // Ensure the rectangle is within image bounds
  board_rect = board_rect & Rect(0, 0, width, height);

  Mat board_mask = Mat::zeros(height, width, CV_8U);
  board_mask(board_rect) = 255;

  Mat masked_edges;
  bitwise_and(edges, board_mask, masked_edges);

  vector<Vec4i> all_segments;

  // HoughLinesP Parameters (TUNE THESE CAREFULLY)
  // Increased max_line_gap to connect fragmented Canny edges
  int hough_threshold = 30; // Adjust as needed
  int min_line_length = 40; // Adjust as needed
  int max_line_gap = 15;    // Increased significantly

  HoughLinesP(masked_edges, all_segments, 1, CV_PI / 180, hough_threshold,
              min_line_length, max_line_gap);

  if (bDebug && false) {
    cout << "Total detected line segments: " << all_segments.size() << endl;

    // Visualize Mask and Line Segments
    Mat mask_and_lines = edges.clone();
    Scalar mask_color = Scalar(128); // Gray
    int mask_thickness = 2;
    Scalar line_color = Scalar(0, 255, 255); // Yellow for all lines

    rectangle(mask_and_lines, board_rect, mask_color,
              mask_thickness); // Draw board mask

    // Draw all line segments
    cout << "\n----all line segments---\n";
    for (const auto &line : all_segments) {
      cv::line(mask_and_lines, cv::Point(line[0], line[1]),
               cv::Point(line[2], line[3]), line_color, 1);
      cout << "[" << line[0] << "," << line[1] << "]:" << "[" << line[2] << ","
           << line[3] << "]\n";
    }

    imshow("Board Mask and All Line Segments", mask_and_lines);
    waitKey(0);
  }

  return all_segments;
}

// 3. Convert and Classify Line Segments to Lines (Refactored)
pair<vector<Line>, vector<Line>>
convertSegmentsToLines(const vector<Vec4i> &all_segments, bool bDebug) {
  vector<Line> horizontal_lines_raw, vertical_lines_raw;

  // Angle tolerance for classifying lines as horizontal or vertical (in
  // radians)
  double angle_tolerance = CV_PI / 180.0 * 10; // 10 degrees tolerance

  for (const auto &segment : all_segments) {
    Point pt1(segment[0], segment[1]);
    Point pt2(segment[2], segment[3]);

    // Calculate angle in radians (-pi to pi)
    double angle = atan2(pt2.y - pt1.y, pt2.x - pt1.x);

    // Normalize angle to be within [0, PI)
    if (angle < 0) {
      angle += CV_PI;
    }

    // Determine if the line is horizontal or vertical based on angle
    bool is_horizontal = false;
    bool is_vertical = false;

    // Check for horizontal lines (angle near 0 or PI)
    if (abs(angle) < angle_tolerance || abs(angle - CV_PI) < angle_tolerance) {
      is_horizontal = true;
    }
    // Check for vertical lines (angle near PI/2)
    else if (abs(angle - CV_PI / 2.0) < angle_tolerance) {
      is_vertical = true;
    }

    if (is_horizontal) {
      double value = (pt1.y + pt2.y) / 2.0; // Average y-coordinate
      horizontal_lines_raw.push_back({value, angle});
    } else if (is_vertical) {
      double value = (pt1.x + pt2.x) / 2.0; // Average x-coordinate
      vertical_lines_raw.push_back({value, angle});
    }
    // Segments that are neither horizontal nor vertical within tolerance are
    // ignored
  }

  sort(horizontal_lines_raw.begin(), horizontal_lines_raw.end(), compareLines);
  sort(vertical_lines_raw.begin(), vertical_lines_raw.end(), compareLines);

  if (bDebug) {
    cout << "Raw horizontal lines count (after angle classification): "
         << horizontal_lines_raw.size() << endl;
    cout << "Raw vertical lines count (after angle classification): "
         << vertical_lines_raw.size() << endl;
  }
  return std::make_pair(horizontal_lines_raw, vertical_lines_raw);
}

// 4. Cluster and Average Lines
vector<double> clusterAndAverageLines(const vector<Line> &raw_lines,
                                      double threshold, bool bDebug) {
  vector<double> clustered_values;
  if (raw_lines.empty())
    return clustered_values;

  vector<bool> processed(raw_lines.size(), false);
  for (size_t i = 0; i < raw_lines.size(); ++i) {
    if (processed[i])
      continue;
    vector<double> current_cluster;
    current_cluster.push_back(raw_lines[i].value);
    processed[i] = true;
    for (size_t j = i + 1; j < raw_lines.size(); ++j) {
      if (!processed[j] &&
          abs(raw_lines[j].value - raw_lines[i].value) < threshold) {
        if (bDebug && false) {
          cout << "Clustering: " << raw_lines[j].value << " and "
               << raw_lines[i].value
               << " (diff: " << abs(raw_lines[j].value - raw_lines[i].value)
               << ")" << endl;
        }
        current_cluster.push_back(raw_lines[j].value);
        processed[j] = true;
      }
    }
    if (!current_cluster.empty()) {
      clustered_values.push_back(
          accumulate(current_cluster.begin(), current_cluster.end(), 0.0) /
          current_cluster.size());
    }
  }
  sort(clustered_values.begin(), clustered_values.end());
  return clustered_values;
}

// Helper struct to store matching information and define comparison
struct LineMatch {
  int matched_count;
  double score;
  double start_value;
  vector<pair<double, double>>
      matched_values; // Store (clustered_value, distance)

  // Custom comparison: prioritize matched_count (descending), then score
  // (ascending)
  bool operator<(const LineMatch &other) const {
    if (matched_count != other.matched_count) {
      return matched_count > other.matched_count; // Descending order by count
    }
    // Ascending order by score if counts are equal
    if (score != other.score) {
      return score < other.score;
    }
    return start_value < other.start_value;
  }
};

vector<double> findUniformGridLinesImproved(const vector<double> &values,
                                            double dominant_distance,
                                            int target_count, double tolerance,
                                            bool bDebug) {
  vector<double> uniform_lines;

  if (values.size() < 2) {
    if (bDebug) {
      cout << "findUniformGridLinesImproved2: Less than 2 values, cannot find "
              "uniform grid.\n";
    }
    return uniform_lines; // Return empty
  }

  double lower_limit = values.front();
  double upper_limit = values.back();

  set<LineMatch> match_data;

  // 1. Combined Iteration for Scoring and Matching
  for (double start_value : values) {
    int matched_lines_count = 0;
    double current_fit_score = 0.0;
    vector<pair<double, double>> current_matched_values; // Local matched_values

    vector<double> expected_lines;
    expected_lines.push_back(start_value);

    // Generate lines in both directions
    double current_line = start_value;
    while (current_line > lower_limit) {
      current_line -= dominant_distance;
      expected_lines.push_back(current_line);
    }

    current_line = start_value;
    while (current_line < upper_limit) {
      current_line += dominant_distance;
      expected_lines.push_back(current_line);
    }

    for (double expected_line : expected_lines) {
      double min_diff = numeric_limits<double>::max();
      double closest_value = numeric_limits<double>::quiet_NaN();

      for (double clustered_value : values) {
        double diff = abs(clustered_value - expected_line);
        if (diff < min_diff) {
          min_diff = diff;
          closest_value = clustered_value;
        }
      }
      current_matched_values.push_back(
          {closest_value, min_diff}); // Store clustered_value!
      if (min_diff < tolerance) {
        current_fit_score += min_diff;
        matched_lines_count++;
      }
    }
    match_data.insert({matched_lines_count, current_fit_score, start_value,
                       current_matched_values});
  }

  // 2. Basis Line Selection
  if (match_data.empty()) {
    if (bDebug) {
      cout << "findUniformGridLinesImproved2: No matching data found.\n";
    }
    return uniform_lines; // Return empty
  }

  const vector<pair<double, double>> &best_matched_values =
      match_data.begin()
          ->matched_values; // Get matched_values from best candidate

  // 3. Final Line Selection
  if (best_matched_values.size() > target_count) {
    vector<pair<double, double>> sorted_matched_values = best_matched_values;
    sort(sorted_matched_values.begin(), sorted_matched_values.end(),
         [](const pair<double, double> &a, const pair<double, double> &b) {
           return a.second < b.second; // Sort by distance
         });

    for (int i = 0; i < target_count; ++i) {
      uniform_lines.push_back(
          sorted_matched_values[i].first); // Store clustered_value
    }
  } else {
    for (const auto &pair : best_matched_values) {
      uniform_lines.push_back(pair.first); // Store clustered_value
    }
  }

  sort(uniform_lines.begin(), uniform_lines.end());

  // Final check: If we didn't find enough lines, return empty.
  if (uniform_lines.size() < target_count) {
    if (bDebug) {
      cout << "findUniformGridLinesImproved2: Could not find enough uniform "
              "lines. Found: "
           << uniform_lines.size() << ", Expected: " << target_count << "\n";
    }
    uniform_lines.clear();
  }

  return uniform_lines;
}

// 5. Find Uniform Grid Lines (Refactored - More Robust)
vector<double> findUniformGridLines(vector<double> &values, int target_count,
                                    double tolerance, bool bDebug) {
  // target_count is expected to be 19

  if (bDebug && !values.empty()) {
    cout << "Sorted clustered values of size: {" << values.size() << "}:\n";
    for (size_t i = 0; i < values.size() - 1; ++i) {
      cout << "value[" << i << "]: " << values[i]
           << " distance: " << values[i + 1] - values[i] << endl;
    }
    cout << "value[" << values.size() - 1 << "]:" << values[values.size() - 1]
         << endl;
  }
  vector<double> uniform_lines; // Declare uniform_lines here!
  if (values.size() < 2) {
    if (bDebug) {
      cout << "findUniformGridLines: Less than 2 clustered values ("
           << values.size() << "), cannot find uniform grid." << endl;
    }
    return vector<double>{}; // Return empty if too few lines
  }

  // 1. Calculate distances between adjacent clustered lines.
  vector<double> distances;
  for (size_t i = 0; i < values.size() - 1; ++i) {
    distances.push_back(values[i + 1] - values[i]);
  }

  // 2. Estimate the dominant grid spacing.
  // We'll do this by analyzing the distribution of distances.
  // A simple way is to use a histogram or cluster the distances.
  // Let's try clustering the distances to find the most frequent spacing.

  if (distances.empty()) {
    if (bDebug) {
      cout << "findUniformGridLines: No distances between clustered values."
           << endl;
    }
    return vector<double>{};
  }

  // Use a map to count the frequency of distances (with a small bin size)
  map<int, int> distance_hist;
  double bin_size = 1.0; // Bin size for the histogram of distances
  for (double d : distances) {
    distance_hist[static_cast<int>(d / bin_size)]++;
  }

  int max_freq = 0;
  int dominant_distance_bin = -1;
  for (auto const &[bin, freq] : distance_hist) {
    if (freq > max_freq) {
      max_freq = freq;
      dominant_distance_bin = bin;
    }
  }

  double estimated_dominant_distance = dominant_distance_bin * bin_size;

  // Refine the dominant distance by averaging the distances within the dominant
  // bin
  double sum_dominant_distances = 0;
  int count_dominant_distances = 0;
  for (double d : distances) {
    if (abs(d - estimated_dominant_distance) <
        bin_size) { // Check if distance is close to the estimated dominant
                    // distance
      sum_dominant_distances += d;
      count_dominant_distances++;
    }
  }

  double dominant_distance = estimated_dominant_distance;
  if (count_dominant_distances > 0) {
    dominant_distance = sum_dominant_distances / count_dominant_distances;
  }

  if (bDebug) {
    cout << "Estimated dominant distance: " << dominant_distance << endl;
  }

  // 3. Find the best starting line and generate the 19 uniform lines.
  return findUniformGridLinesImproved(values, dominant_distance, 19,
                                      dominant_distance * tolerance, bDebug);
}

// Helper function to find the optimal clustering for a given set of lines
pair<vector<double>, double> findOptimalClusteringForOrientation(
    const vector<Line> &raw_lines, int target_count,
    const string
        &orientation, // "horizontal" or "vertical" for debugging output
    bool bDebug) {

  double cluster_threshold = 1.0; // Starting threshold - needs tuning
  double threshold_step = 0.5;    // Step to increase threshold - needs tuning
  int max_iterations = 30;        // Limit iterations - needs tuning

  vector<double> clustered_lines;
  vector<double> prev_clustered_lines;
  double optimal_threshold = cluster_threshold;

  // Initialize previous results with clustering at a very low threshold
  prev_clustered_lines = clusterAndAverageLines(raw_lines, 0.1, bDebug);

  if (bDebug) {
    cout << "Initial Clustering (" << orientation
         << ") with threshold 0.1: " << prev_clustered_lines.size()
         << " lines\n";
  }

  clustered_lines =
      prev_clustered_lines; // Initialize with the initial clustering

  for (int i = 0; i < max_iterations; ++i) {
    // Store current results
    prev_clustered_lines = clustered_lines;

    // Perform clustering with the current threshold
    clustered_lines =
        clusterAndAverageLines(raw_lines, cluster_threshold, bDebug);

    if (bDebug) {
      cout << "Clustering Attempt (" << orientation << ") " << i + 1
           << " with threshold " << cluster_threshold << ": "
           << clustered_lines.size() << " lines\n";
    }

    if (clustered_lines.size() < target_count) {
      if (bDebug) {
        cout << "Clustered line count (" << orientation
             << ") dropped below target (" << target_count
             << "). Returning previous threshold's results.\n";
      }
      return make_pair(prev_clustered_lines,
                       optimal_threshold); // Return previous (better) result
    }

    if (clustered_lines.size() == target_count) {
      if (bDebug) {
        cout << "Found target number of clustered lines (" << target_count
             << ") for " << orientation << ".\n";
      }
      return make_pair(clustered_lines, cluster_threshold);
    }

    optimal_threshold = cluster_threshold; // Update optimal threshold as we are
                                           // still at or above target
    cluster_threshold +=
        threshold_step; // Increase threshold for the next attempt
  }

  if (bDebug) {
    cout << "Max iterations reached for " << orientation
         << " without finding target count or dropping below. Returning last "
            "iteration's results.\n";
  }
  return make_pair(clustered_lines, optimal_threshold);
}

// Main function to call the helper functions for horizontal and vertical lines
pair<vector<double>, vector<double>>
findOptimalClustering(const vector<Line> &horizontal_lines_raw,
                      const vector<Line> &vertical_lines_raw, int target_count,
                      bool bDebug) {

  pair<vector<double>, double> horizontal_result =
      findOptimalClusteringForOrientation(horizontal_lines_raw, target_count,
                                          "horizontal", bDebug);

  pair<vector<double>, double> vertical_result =
      findOptimalClusteringForOrientation(vertical_lines_raw, target_count,
                                          "vertical", bDebug);

  return make_pair(horizontal_result.first, vertical_result.first);
}

// Refactored detectUniformGrid to use the new findOptimalClustering function
pair<vector<double>, vector<double>> detectUniformGrid(const Mat &image) {
  Mat processed_image = preprocessImage(image, bDebug);
  vector<Vec4i> mixed_segments = detectLineSegments(processed_image, bDebug);
  auto [horizontal_lines_raw, vertical_lines_raw] =
      convertSegmentsToLines(mixed_segments, bDebug);

  // Use the new function to find the optimal clustering to get potentially 19
  // lines We pass 19 as the target count for clustering
  auto [clustered_horizontal_y, clustered_vertical_x] = findOptimalClustering(
      horizontal_lines_raw, vertical_lines_raw, 19, bDebug);

  vector<double> final_horizontal_y;
  vector<double> final_vertical_x;

  // Now, use findUniformGridLines to verify the uniformity of the clustered
  // lines findUniformGridLines expects exactly 19 lines as input in the ideal
  // case and verifies uniformity
  double uniformity_tolerance =
      0.1; // Tolerance for uniformity check - needs tuning
  // Call the robust findUniformGridLines
  final_horizontal_y = findUniformGridLines(clustered_horizontal_y, 19,
                                            uniformity_tolerance, bDebug);
  final_vertical_x = findUniformGridLines(clustered_vertical_x, 19,
                                          uniformity_tolerance, bDebug);

  // Final check: If we didn't end up with exactly 19 uniform lines in both
  // directions, throw an exception.
  if (final_horizontal_y.size() != 19 || final_vertical_x.size() != 19) {
    if (bDebug) {
      cout << "Final grid line detection failed: Expected 19x19 uniform grid, "
              "but found "
           << final_horizontal_y.size() << " horizontal and "
           << final_vertical_x.size()
           << " vertical uniform lines after findUniformGridLines." << endl;
    }
    // Throw an exception as the detection failed
    THROWGEMERROR(std::string("Failed to detect 19x19 Go board grid. Found ") +
                  Num2Str(final_horizontal_y.size()).str() +
                  " horizontal and " + Num2Str(final_vertical_x.size()).str() +
                  " vertical lines after uniform grid finding.");
  }

  // The lines are already sorted by clusterAndAverageLines and verified by
  // findUniformGridLines. No need to sort again here.

  if (bDebug) {
    cout << "Final detected uniform horizontal lines (y): ";
    for (double y : final_horizontal_y)
      cout << y << " ";
    cout << endl;
    cout << "Final detected uniform vertical lines (x): ";
    for (double x : final_vertical_x)
      cout << x << " ";
    cout << endl;
  }
  return make_pair(final_horizontal_y, final_vertical_x);
}

// Function to find intersection points of two sets of lines
vector<Point2f> findIntersections(const vector<double> &horizontal_lines,
                                  const vector<double> &vertical_lines) {
  vector<Point2f> intersections;
  for (double y : horizontal_lines) {
    for (double x : vertical_lines) {
      intersections.push_back(Point2f(x, y));
    }
  }
  return intersections;
}

// Function to calculate the weighted Euclidean distance between two HSV colors
float colorDistanceWeighted(const Vec3f &color1, const Vec3f &color2,
                            float weight_h, float weight_s, float weight_v) {
  const float max_h = 180.0f;
  const float max_s = 255.0f;
  const float max_v = 255.0f;
  const float epsilon = 1e-6f; // A small value to avoid division by zero
  const float max_distance =
      max_h * weight_h + max_s * weight_s + max_v * weight_v;

  float dh = (color2[0] > epsilon) ? abs(color1[0] - color2[0]) / max_h : 0.0f;
  float ds = (color2[1] > epsilon) ? abs(color1[1] - color2[1]) / max_s : 0.0f;
  float dv = (color2[2] > epsilon) ? abs(color1[2] - color2[2]) / max_v : 0.0f;

  float distance = dh * weight_h + ds * weight_s + dv * weight_v;
  return (max_distance > epsilon) ? distance / max_distance : 0.0f;
}

// Function to calculate the original Euclidean distance between two HSV colors
float colorDistance(const Vec3f &color1, const Vec3f &color2) {
  return sqrt(pow(color1[0] - color2[0], 2) + pow(color1[1] - color2[1], 2) +
              pow(color1[2] - color2[2], 2));
}

// Helper structure to store cluster index and its properties/scores
struct ClusterInfo {
  int index;
  float h, s, v;
  float black_score;
  float white_score;
  float board_score;

  ClusterInfo(int idx, float h_val, float s_val, float v_val)
      : index(idx), h(h_val), s(s_val), v(v_val),
        black_score(std::numeric_limits<float>::max()),
        white_score(std::numeric_limits<float>::max()),
        board_score(std::numeric_limits<float>::max()) {}
};

// Revised function to classify clusters using H, S, and V components
void classifyClusters(const Mat &centers, int &label_black, int &label_white,
                      int &label_board) {

  if (centers.rows != 3) {
    // Handle the case where k-means didn't return exactly 3 clusters
    // For now, throw an error or fallback to the old method if desired.
    // This implementation assumes 3 clusters as per the original k-means setup.
    std::cerr << "Error: Expected 3 cluster centers, but found " << centers.rows
              << std::endl;
    // Assign default/invalid values
    label_black = -1;
    label_white = -1;
    label_board = -1;
    // Optionally, you could try the old V-based logic here as a fallback
    // Or throw an exception:
    THROWGEMERROR(std::string("Expected 3 cluster centers, found ") +
                  Num2Str(centers.rows).str());
    return; // Or throw
  }

  std::vector<ClusterInfo> cluster_data;
  for (int i = 0; i < centers.rows; ++i) {
    cluster_data.emplace_back(i, centers.at<float>(i, 0),
                              centers.at<float>(i, 1), centers.at<float>(i, 2));
  }

  // --- Scoring Parameters (These may need tuning based on typical images) ---
  const float low_s_threshold =
      60.0f; // Saturation below this suggests black/white stone
  const float high_s_threshold = 40.0f; // Saturation above this suggests board
  const float low_v_threshold = 80.0f;  // Value below this suggests black stone
  const float high_v_threshold =
      170.0f; // Value above this suggests white stone

  // --- Calculate Scores for each cluster ---
  for (auto &cluster : cluster_data) {
    // Score for Black: Low V, Low S
    // Penalize high V and high S
    cluster.black_score = (cluster.v / 255.0f) + (cluster.s / 255.0f);
    // Add a larger penalty if V is clearly too high or S is too high
    if (cluster.v > low_v_threshold * 1.5)
      cluster.black_score += 1.0f; // Heavier penalty if too bright
    if (cluster.s > low_s_threshold)
      cluster.black_score += 0.5f; // Penalty if saturated

    // Score for White: High V, Low S
    // Penalize low V and high S
    cluster.white_score =
        ((255.0f - cluster.v) / 255.0f) + (cluster.s / 255.0f);
    // Add a larger penalty if V is clearly too low or S is too high
    if (cluster.v < high_v_threshold * 0.8)
      cluster.white_score += 1.0f; // Heavier penalty if too dark
    if (cluster.s > low_s_threshold)
      cluster.white_score += 0.5f; // Penalty if saturated

    // Score for Board: Mid V, Higher S (relative to stones)
    // Penalize very low/high V and very low S
    float v_mid_penalty = std::abs(cluster.v - 128.0f) /
                          128.0f; // Penalize distance from mid-value
    float s_low_penalty =
        (cluster.s < high_s_threshold)
            ? (high_s_threshold - cluster.s) / high_s_threshold
            : 0.0f; // Penalize low saturation
    cluster.board_score =
        v_mid_penalty + s_low_penalty * 1.5f; // Weight low S penalty more
  }

  // --- Assign Labels based on lowest scores ---

  // Find best candidate for Black
  std::sort(cluster_data.begin(), cluster_data.end(),
            [](const ClusterInfo &a, const ClusterInfo &b) {
              return a.black_score < b.black_score;
            });
  label_black = cluster_data[0].index;

  // Find best candidate for White (excluding the one chosen for Black)
  std::sort(cluster_data.begin(), cluster_data.end(),
            [&](const ClusterInfo &a, const ClusterInfo &b) {
              if (a.index == label_black)
                return false; // Ensure black label is not chosen
              if (b.index == label_black)
                return true; // Ensure black label is not chosen
              return a.white_score < b.white_score;
            });
  // The best white candidate (that isn't black) will be at the start now
  // Need to handle the edge case where the first element *is* the black label
  label_white = (cluster_data[0].index != label_black) ? cluster_data[0].index
                                                       : cluster_data[1].index;

  // Find best candidate for Board (excluding Black and White)
  std::sort(cluster_data.begin(), cluster_data.end(),
            [&](const ClusterInfo &a, const ClusterInfo &b) {
              if (a.index == label_black || a.index == label_white)
                return false; // Exclude black/white
              if (b.index == label_black || b.index == label_white)
                return true; // Exclude black/white
              return a.board_score < b.board_score;
            });
  // The best board candidate (that isn't black or white) will be at the start
  // Need to handle edge cases where the first/second elements are black/white
  if (cluster_data[0].index != label_black &&
      cluster_data[0].index != label_white) {
    label_board = cluster_data[0].index;
  } else if (cluster_data.size() > 1 && cluster_data[1].index != label_black &&
             cluster_data[1].index != label_white) {
    label_board = cluster_data[1].index;
  } else if (cluster_data.size() > 2 && cluster_data[2].index != label_black &&
             cluster_data[2].index != label_white) {
    label_board = cluster_data[2].index; // Should be the last one if size is 3
  } else {
    // This case should theoretically not happen if there are 3 distinct
    // clusters and labels are assigned correctly, but as a fallback:
    for (const auto &cluster : cluster_data) {
      if (cluster.index != label_black && cluster.index != label_white) {
        label_board = cluster.index;
        break;
      }
    }
  }

  // --- Debug Output (Optional) ---
  if (bDebug) { // Assuming bDebug is accessible here or passed as argument
    std::cout << "\n--- Cluster Classification Scores ---\n";
    std::cout << std::fixed << std::setprecision(3);
    for (const auto &cluster : cluster_data) {
      std::cout << "Cluster " << cluster.index << ": HSV(" << cluster.h << ", "
                << cluster.s << ", " << cluster.v << ")"
                << " Scores[Blk:" << cluster.black_score
                << ", Wht:" << cluster.white_score
                << ", Brd:" << cluster.board_score << "]\n";
    }
    std::cout << "\n--- Assigned Labels (Score Based) ---\n";
    std::cout << "Black Cluster ID: " << label_black << std::endl;
    std::cout << "White Cluster ID: " << label_white << std::endl;
    std::cout << "Board Cluster ID: " << label_board << std::endl;

    // Sanity check: ensure labels are unique and valid
    if (label_black == label_white || label_black == label_board ||
        label_white == label_board || label_black == -1 || label_white == -1 ||
        label_board == -1) {
      std::cerr << "Warning: Label assignment resulted in duplicate or invalid "
                   "labels!\n";
    }
  }
}

// Function to sample a region around a point and get the average HSV
Vec3f getAverageHSV(const Mat &image, Point2f center, int radius) {
  Vec3f sum(0, 0, 0);
  int count = 0;
  for (int y = center.y - radius; y <= center.y + radius; ++y) {
    for (int x = center.x - radius; x <= center.x + radius; ++x) {
      if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
        Vec3b bgr_color = image.at<Vec3b>(y, x);
        Mat bgr_pixel(1, 1, CV_8UC3,
                      bgr_color); // Create a 1x1 Mat from the pixel
        Mat hsv_pixel;
        cvtColor(bgr_pixel, hsv_pixel, COLOR_BGR2HSV);
        Vec3b hsv = hsv_pixel.at<Vec3b>(0, 0);
        sum[0] += hsv[0];
        sum[1] += hsv[1];
        sum[2] += hsv[2];
        count++;
      }
    }
  }
  if (count > 0) {
    return sum / count;
  } else {
    return Vec3f(0, 0, 0); // Return black HSV if no valid pixels
  }
}

double getSampleRadiusSize(const vector<double> &horizontal_lines,
                           const vector<double> &vertical_lines) {
  // range of [1,3]
  return min(3.0, max(2.0, (abs(horizontal_lines[1] - horizontal_lines[0]) +
                            abs(vertical_lines[1] - vertical_lines[0])) /
                               2.0f / 8.0f));
}

// Helper function to calculate the shortest distance between two hues (0-180
// range)
inline float hueDistance(float h1, float h2) {
  float diff = std::abs(h1 - h2);
  // OpenCV HSV uses H range 0-179
  return std::min(diff, 180.0f - diff);
}

// Revised function to find the closest cluster center using weighted Euclidean
// distance in HSV space
int findClosestCenter(const Vec3f &hsv, const Mat &centers) {
  const int num_clusters = centers.rows;
  if (num_clusters == 0) {
    // Handle case with no centers
    return -1; // Or throw an error
  }

  // --- Weights for HSV components (TUNABLE PARAMETERS) ---
  // These weights determine the relative importance of Hue, Saturation, and
  // Value. For Black/White/Board distinction:
  // - Saturation (S) is often important to separate stones (low S) from board
  // (higher S).
  // - Value (V) is crucial to separate Black (low V) from White (high V).
  // - Hue (H) might be less critical for stones but useful for board color.
  // Adjust these based on testing:
  const float weight_h = 0.5f; // Less weight for Hue initially
  const float weight_s = 1.5f; // More weight for Saturation
  const float weight_v = 1.5f; // More weight for Value

  // --- Normalization factors (Max values for OpenCV HSV) ---
  // We normalize the differences by the max possible range before applying
  // weights to make weights more comparable across components.
  const float max_h_diff = 90.0f; // Max possible hueDistance is 180/2 = 90
  const float max_s = 255.0f;
  const float max_v = 255.0f;

  float min_distance_sq =
      std::numeric_limits<float>::max(); // Compare squared distances to avoid
                                         // sqrt
  int closest_center_index = -1;

  for (int i = 0; i < num_clusters; ++i) {
    Vec3f cluster_center(centers.at<float>(i, 0), centers.at<float>(i, 1),
                         centers.at<float>(i, 2));

    // Calculate normalized differences for each component
    float dh_normalized = hueDistance(hsv[0], cluster_center[0]) / max_h_diff;
    float ds_normalized = std::abs(hsv[1] - cluster_center[1]) / max_s;
    float dv_normalized = std::abs(hsv[2] - cluster_center[2]) / max_v;

    // Calculate weighted squared Euclidean distance
    // d^2 = w_h * (normalized_dh)^2 + w_s * (normalized_ds)^2 + w_v *
    // (normalized_dv)^2
    float distance_sq = weight_h * std::pow(dh_normalized, 2) +
                        weight_s * std::pow(ds_normalized, 2) +
                        weight_v * std::pow(dv_normalized, 2);

    if (distance_sq < min_distance_sq) {
      min_distance_sq = distance_sq;
      closest_center_index = i;
    }
  }

  // Optional Debug Output
  // if (bDebug) {
  //     std::cout << "Sample HSV: [" << hsv[0] << "," << hsv[1] << "," <<
  //     hsv[2] << "] -> Closest Center: " << closest_center_index << " (Dist^2:
  //     " << min_distance_sq << ")" << std::endl;
  // }

  return closest_center_index;
}

// Function to sample a region around a point and get the average Lab
// NOTE: Assumes input image is already in Lab format (CV_8UC3)
Vec3f getAverageLab(const Mat &image_lab, Point2f center, int radius) {
  Vec3d sum(0.0, 0.0, 0.0);
  std::vector<uchar> l_values;
  std::vector<uchar> a_values;
  std::vector<uchar> b_values;

  // Define the square boundary for sampling
  int x_min = max(0, static_cast<int>(center.x - radius));
  int x_max = min(image_lab.cols - 1, static_cast<int>(center.x + radius));
  int y_min = max(0, static_cast<int>(center.y - radius));
  int y_max = min(image_lab.rows - 1, static_cast<int>(center.y + radius));

  for (int y = y_min; y <= y_max; ++y) {
    for (int x = x_min; x <= x_max; ++x) {
      // Optional: Check if the pixel is within the circular radius
      if (std::pow(x - center.x, 2) + std::pow(y - center.y, 2) <=
          std::pow(radius, 2)) {
        Vec3b lab = image_lab.at<Vec3b>(y, x);
        l_values.push_back(lab[0]); // L
        a_values.push_back(lab[1]); // a
        b_values.push_back(lab[2]); // b
      }
    }
  }

  size_t count = l_values.size(); // Number of pixels sampled

  if (count > 0) {
    // Sort each channel's values independently
    std::sort(l_values.begin(), l_values.end());
    std::sort(a_values.begin(), a_values.end());
    std::sort(b_values.begin(), b_values.end());

    // Find the median index
    size_t mid_index = count / 2;

    // Extract the median value for each channel
    // (Using the middle element - simple approach for even/odd counts)
    float median_l = static_cast<float>(l_values[mid_index]);
    float median_a = static_cast<float>(a_values[mid_index]);
    float median_b = static_cast<float>(b_values[mid_index]);

    return Vec3f(median_l, median_a, median_b);
  } else {
    // Return default Lab (e.g., mid-gray) if no valid pixels found
    return Vec3f(128.0f, 128.0f, 128.0f);
  }
}

const float MAX_DIST_STONE_CALIB_PHASE3 =
    35.0f; // Max Lab distance for a sample to be considered a stone matching
           // its calibrated color
const float MAX_DIST_BOARD_CALIB_PHASE3 =
    45.0f; // Max Lab distance for a sample to be considered board matching its
           // calibrated color

// --- NEW HELPER: Classify a single intersection's Lab color using calibration
// data --- This was the smaller helper from the previous step, still useful.
static int classifySingleIntersectionByDistance(
    const cv::Vec3f &intersection_lab_color, const cv::Vec3f &avg_black_calib,
    const cv::Vec3f &avg_white_calib, const cv::Vec3f &board_calib_lab) {
  if (intersection_lab_color[0] <
      0) {    // Check for invalid sample from getAverageLab
    return 0; // Default to empty
  }

  float dist_b = cv::norm(intersection_lab_color, avg_black_calib, cv::NORM_L2);
  float dist_w = cv::norm(intersection_lab_color, avg_white_calib, cv::NORM_L2);
  float dist_empty =
      cv::norm(intersection_lab_color, board_calib_lab, cv::NORM_L2);

  float min_dist = std::min({dist_b, dist_w, dist_empty});
  int classification = 0;

  if (min_dist == dist_b && dist_b < MAX_DIST_STONE_CALIB_PHASE3) {
    classification = 1; // Black
  } else if (min_dist == dist_w && dist_w < MAX_DIST_STONE_CALIB_PHASE3) {
    classification = 2; // White
  } else if (min_dist == dist_empty &&
             dist_empty < MAX_DIST_BOARD_CALIB_PHASE3) {
    classification = 0; // Empty
  } else {
    classification = 0; // Uncertain -> Empty
  }
  return classification;
}

// --- NEW HELPER: Perform direct classification for all intersections ---
static void performDirectClassification(
    const std::vector<cv::Vec3f>
        &average_lab_values,           // Lab samples for all intersections
    const CalibrationData &calib_data, // Loaded calibration data
    const std::vector<cv::Point2f>
        &intersection_points, // Coordinates for drawing
    int num_intersections,    // Typically 361
    const cv::Mat
        &original_bgr_image_for_drawing, // To clone for board_with_stones
    cv::Mat &board_state_output,         // Out: 19x19 matrix with 0,1,2
    cv::Mat &board_with_stones_output)   // Out: Image with stones drawn
{
  if (bDebug)
    std::cout << "  Debug (performDirectClassification): Starting..."
              << std::endl;

  cv::Vec3f avg_black_calib = (calib_data.lab_tl + calib_data.lab_bl) * 0.5f;
  cv::Vec3f avg_white_calib = (calib_data.lab_tr + calib_data.lab_br) * 0.5f;
  // calib_data.lab_board_avg is already the averaged board color

  board_state_output = cv::Mat(19, 19, CV_8U, cv::Scalar(0)); // Initialize
  board_with_stones_output = original_bgr_image_for_drawing.clone();

  if (bDebug) {
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "    Using Ref Black Lab: " << avg_black_calib << std::endl;
    std::cout << "    Using Ref White Lab: " << avg_white_calib << std::endl;
    std::cout << "    Using Ref Board Lab: " << calib_data.lab_board_avg
              << std::endl;
    std::cout << "    Using Thresholds: Stone=" << MAX_DIST_STONE_CALIB_PHASE3
              << ", Board=" << MAX_DIST_BOARD_CALIB_PHASE3 << std::endl;
  }

  for (int i = 0; i < num_intersections; ++i) {
    int row = i / 19;
    int col = i % 19;
    if (row >= 19 || col >= 19)
      continue;

    cv::Vec3f current_intersection_lab = average_lab_values[i];
    int classification;

    if (current_intersection_lab[0] < 0) { // Check if getAverageLab failed
      if (bDebug)
        std::cout << "    Intersection [" << row << "," << col
                  << "] - Invalid Lab sample (-1). Classifying as Empty."
                  << std::endl;
      classification = 0;
      cv::circle(board_with_stones_output, intersection_points[i], 8,
                 cv::Scalar(0, 128, 255), 2); // Orange for bad sample
    } else {
      classification = classifySingleIntersectionByDistance(
          current_intersection_lab, avg_black_calib, avg_white_calib,
          calib_data.lab_board_avg);
    }

    board_state_output.at<uchar>(row, col) = classification;

    if (bDebug) { // More detailed per-intersection logging
      float dist_b =
          cv::norm(current_intersection_lab, avg_black_calib, cv::NORM_L2);
      float dist_w =
          cv::norm(current_intersection_lab, avg_white_calib, cv::NORM_L2);
      float dist_empty = cv::norm(current_intersection_lab,
                                  calib_data.lab_board_avg, cv::NORM_L2);
      std::string stone_type =
          (classification == 1)
              ? "Black"
              : (classification == 2 ? "White" : "Empty/Board");
      if (current_intersection_lab[0] >=
          0) { // Only print distances if sample was valid
        std::cout << "    Int [" << row << "," << col
                  << "] Lab: " << current_intersection_lab
                  << " D(B):" << std::setw(5) << dist_b
                  << " D(W):" << std::setw(5) << dist_w
                  << " D(E):" << std::setw(5) << dist_empty
                  << " -> Class: " << stone_type << " (" << classification
                  << ")" << std::endl;
      }
    }

    if (classification == 1) {
      cv::circle(board_with_stones_output, intersection_points[i], 8,
                 cv::Scalar(0, 0, 0), -1);
    } else if (classification == 2) {
      cv::circle(board_with_stones_output, intersection_points[i], 8,
                 cv::Scalar(255, 255, 255), -1);
    } else if (current_intersection_lab[0] >=
               0) { // Only draw green if it wasn't an invalid sample initially
      cv::circle(board_with_stones_output, intersection_points[i], 8,
                 cv::Scalar(0, 255, 0), 2);
    }
  }

  if (bDebug) {
    imshow("Direct Classification Result (Helper)", board_with_stones_output);
    cv::waitKey(1); // Give GUI time to refresh
  }
  if (bDebug)
    std::cout << "  Debug (performDirectClassification): Finished."
              << std::endl;
}

// Function to process the Go board image and determine the board state
void processGoBoard(const cv::Mat &image_bgr_in, cv::Mat &board_state_out, // Renamed for clarity
  cv::Mat &board_with_stones_out,    // Renamed for clarity
  std::vector<cv::Point2f> &intersection_points_out) // Renamed for clarity
{
if (bDebug) std::cout << "Debug (processGoBoard): Starting board processing." << std::endl;

// 1. Load Calibration Data
CalibrationData calib_data = loadCalibrationData(CALIB_CONFIG_PATH);
if (!calib_data.corners_loaded || !calib_data.colors_loaded || !calib_data.board_color_loaded || !calib_data.dimensions_loaded) {
std::string err_msg = "ProcessGoBoard Error: Incomplete calibration data. ";
if (!calib_data.corners_loaded) err_msg += "Corners missing. ";
if (!calib_data.colors_loaded) err_msg += "Stone colors missing. ";
if (!calib_data.board_color_loaded) err_msg += "Board color missing. ";
if (!calib_data.dimensions_loaded) err_msg += "Dimensions missing. ";
err_msg += "Please run calibration (-b) ensuring stones and empty board are correctly placed.";
THROWGEMERROR(err_msg);
}
if (bDebug) std::cout << "  Debug: Full calibration data loaded." << std::endl;

// 2. Perspective Correction
cv::Mat image_bgr_corrected = correctPerspective(image_bgr_in);
if (image_bgr_corrected.empty()) { THROWGEMERROR("Corrected perspective image is empty."); }
if (bDebug) { imshow("Corrected Perspective", image_bgr_corrected); cv::waitKey(1); }

// 3. Convert to Lab and Detect Grid
cv::Mat image_lab;
cv::cvtColor(image_bgr_corrected, image_lab, cv::COLOR_BGR2Lab);
if (image_lab.empty()) { THROWGEMERROR("Lab converted image is empty."); }
if (bDebug) { imshow("Lab Image", image_lab); cv::waitKey(1); }

std::pair<std::vector<double>, std::vector<double>> grid_lines = detectUniformGrid(image_bgr_corrected);
std::vector<double> horizontal_lines = grid_lines.first;
std::vector<double> vertical_lines = grid_lines.second;

intersection_points_out = findIntersections(horizontal_lines, vertical_lines);
int num_intersections = intersection_points_out.size();
if (num_intersections != 361 && image_bgr_corrected.cols > 0 && image_bgr_corrected.rows > 0) {
std::cerr << "Warning (processGoBoard): Expected 361 intersections, found " << num_intersections << "." << std::endl;
if (num_intersections == 0) THROWGEMERROR("No intersection points found.");
} else if (num_intersections == 0) { 
THROWGEMERROR("No intersection points found (image might be invalid).");
}
if (bDebug) std::cout << "  Debug: Found " << num_intersections << " intersection points." << std::endl;

// 4. Sample Lab Color at Each Intersection
int sample_radius = getSampleRadiusSize(horizontal_lines, vertical_lines);
std::vector<cv::Vec3f> average_lab_values(num_intersections);
for (int i = 0; i < num_intersections; ++i) {
average_lab_values[i] = getAverageLab(image_lab, intersection_points_out[i], sample_radius);
}
if (bDebug) std::cout << "  Debug: Sampled Lab colors for all intersections." << std::endl;

// --- 5. Call the new helper function for Direct Classification ---
performDirectClassification(
average_lab_values,
calib_data,
intersection_points_out,
num_intersections,
image_bgr_corrected, // Pass the corrected BGR image for drawing
board_state_out,     // Output: board_state
board_with_stones_out // Output: board_with_stones
);
// The imshow for "Direct Classification Result" is now inside performDirectClassification if bDebug

// --- 6. Post-Processing Filter ---
if (bDebug) std::cout << "  Debug: Applying post-processing filter." << std::endl;
if (num_intersections == 361) { // Ensure we have the expected number of points for 19x19 indexing
cv::Mat temp_board_state = board_state_out.clone();
for (int r = 0; r < 19; ++r) {
for (int c = 0; c < 19; ++c) {
if (temp_board_state.at<uchar>(r, c) == 2) { // If white
  bool has_stone_neighbor = false;
  for (int dr = -1; dr <= 1; ++dr) {
      for (int dc = -1; dc <= 1; ++dc) {
          if (dr == 0 && dc == 0) continue;
          int nr = r + dr; int nc = c + dc;
          if (nr >= 0 && nr < 19 && nc >= 0 && nc < 19) {
              if (temp_board_state.at<uchar>(nr, nc) == 1 || temp_board_state.at<uchar>(nr, nc) == 2) {
                  has_stone_neighbor = true; break;
              }
          }
      }
      if (has_stone_neighbor) break;
  }
  if (!has_stone_neighbor) {
      board_state_out.at<uchar>(r, c) = 0;
      // Ensure index is valid before drawing
      int intersection_idx = r * 19 + c;
      if (intersection_idx < intersection_points_out.size()) {
         cv::circle(board_with_stones_out, intersection_points_out[intersection_idx], 8, cv::Scalar(0, 255, 0), 2);
      }
  }
}
}
}
} else if (bDebug) {
std::cout << "  Debug: Skipping post-processing filter due to non-standard number of intersections (" << num_intersections <<")." << std::endl;
}

if (bDebug) {
imshow("Filtered Stones (Final)", board_with_stones_out);
cv::waitKey(0);
}
if (bDebug) std::cout << "Debug (processGoBoard): Board processing finished." << std::endl;
}

std::vector<cv::Point2f>
loadCornersFromConfigFile(const std::string &config_path) {
  std::vector<cv::Point2f> corners;
  std::ifstream configFile(config_path);
  if (!configFile.is_open()) {
    if (bDebug)
      std::cout << "Debug (loadCornersFromConfigFile): Config file '"
                << config_path << "' not found." << std::endl;
    return corners; // Return empty vector
  }

  if (bDebug)
    std::cout << "Debug (loadCornersFromConfigFile): Found config file: "
              << config_path << ". Attempting to parse corners." << std::endl;
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

    // Check if all required pixel coordinate keys exist
    if (config_data.count("TL_X_PX") && config_data.count("TL_Y_PX") &&
        config_data.count("TR_X_PX") && config_data.count("TR_Y_PX") &&
        config_data.count("BL_X_PX") && config_data.count("BL_Y_PX") &&
        config_data.count("BR_X_PX") && config_data.count("BR_Y_PX")) {
      cv::Point2f tl(std::stof(config_data["TL_X_PX"]),
                     std::stof(config_data["TL_Y_PX"]));
      cv::Point2f tr(std::stof(config_data["TR_X_PX"]),
                     std::stof(config_data["TR_Y_PX"]));
      // IMPORTANT: Config file saves BL before BR based on previous logic, but
      // perspective expects TL, TR, BR, BL Let's load them based on key names
      // and return in the standard order.
      cv::Point2f bl(std::stof(config_data["BL_X_PX"]),
                     std::stof(config_data["BL_Y_PX"]));
      cv::Point2f br(std::stof(config_data["BR_X_PX"]),
                     std::stof(config_data["BR_Y_PX"]));

      corners = {tl, tr, br, bl}; // Standard order: TL, TR, BR, BL
      if (bDebug)
        std::cout << "Debug (loadCornersFromConfigFile): Successfully loaded "
                     "corners from config file."
                  << std::endl;

    } else {
      if (bDebug)
        std::cerr << "Warning (loadCornersFromConfigFile): Config file missing "
                     "one or more pixel coordinate keys (_PX)."
                  << std::endl;
    }

  } catch (const std::exception &e) {
    if (bDebug)
      std::cerr
          << "Warning (loadCornersFromConfigFile): Error parsing config file '"
          << config_path << "': " << e.what() << std::endl;
    if (configFile.is_open())
      configFile.close();
    corners.clear(); // Ensure empty vector on error
  }
  return corners;
}
