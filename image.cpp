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

bool compareLines(const Line &a, const Line &b) { return a.value < b.value; }

// Forward declarations for static helper functions if defined later

static int classifySingleIntersectionByDistance(
    const cv::Vec3f &intersection_lab_color, const cv::Vec3f &avg_black_calib,
    const cv::Vec3f &avg_white_calib, const cv::Vec3f &board_calib_lab);

//====================================================================//

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
    classification = BLACK; // Black
  } else if (min_dist == dist_w && dist_w < MAX_DIST_STONE_CALIB) {
    classification = WHITE; // White
  } else if (min_dist == dist_empty && dist_empty < MAX_DIST_BOARD_CALIB) {
    classification = EMPTY; // Empty
  } else {
    // Uncertain or too far from all references, classify as empty
    classification = EMPTY;
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
    if (config_map.count("DevicePath")) {
      data.device_path = config_map["DevicePath"];
      data.device_path_loaded = true;
      if (bDebug)
        std::cout << "  Debug: Loaded DevicePath: " << data.device_path
                  << std::endl;
    } else {
      if (bDebug)
        std::cerr << "  Warning: DevicePath missing from config." << std::endl;
    }
    // Parse Dimensions (as in your file)
    if (config_map.count("ImageWidth") && config_map.count("ImageHeight")) {
      data.image_width =
          std::stoi(config_map["ImageWidth"]); // Store in new distinct fields
      data.image_height = std::stoi(config_map["ImageHeight"]);
      data.dimensions_loaded = true;
      if (bDebug)
        std::cout << "  Debug: Loaded Dimensions at Calib: " << data.image_width
                  << "x" << data.image_height << std::endl;
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

vector<Point2f> getBoardCornersCorrected(int width, int height) {
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

  vector<Point2f> output_corners = getBoardCornersCorrected(width, height);
  Mat perspective_matrix =
      getPerspectiveTransform(input_corners, output_corners);
  Mat corrected_image;
  warpPerspective(image, corrected_image, perspective_matrix,
                  Size(width, height));
  if (bDebug) {
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
  vector<Point2f> board_corners = getBoardCornersCorrected(width, height);
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
std::vector<double> findUniformGridLines(
    int target_count,           // Expected number of lines (e.g., 19)
    bool bDebug,                // Debug flag
    int corrected_image_width,  // Width of the perspective-corrected image
    int corrected_image_height, // Height of the perspective-corrected image
    bool is_generating_horizontal_lines // True for horizontal (y-coords), false
                                        // for vertical (x-coords)
) {
  if (bDebug) {
    std::cout
        << "  Debug (findUniformGridLines - NEW CALIBRATED METHOD): Generating "
        << (is_generating_horizontal_lines ? "horizontal" : "vertical")
        << " lines for image size " << corrected_image_width << "x"
        << corrected_image_height << std::endl;
  }

  if (target_count <= 1) {
    THROWGEMERROR("findUniformGridLines: target_count must be greater than 1.");
  }
  if (corrected_image_width <= 0 || corrected_image_height <= 0) {
    THROWGEMERROR(
        "findUniformGridLines: corrected_image dimensions must be positive.");
  }

  // Get the ideal corner positions in the corrected image
  // getBoardCornersCorrected is from common.h (defined in image.cpp)
  std::vector<cv::Point2f> corrected_board_corners =
      getBoardCornersCorrected(corrected_image_width, corrected_image_height);

  if (corrected_board_corners.size() != 4) {
    THROWGEMERROR("findUniformGridLines: getBoardCornersCorrected did not "
                  "return 4 points.");
  }

  cv::Point2f tl = corrected_board_corners[0]; // Top-Left
  cv::Point2f tr = corrected_board_corners[1]; // Top-Right
  // cv::Point2f br = corrected_board_corners[2]; // Bottom-Right (not directly
  // used for main axis start/end)
  cv::Point2f bl = corrected_board_corners[3]; // Bottom-Left

  if (bDebug) {
    std::cout
        << "    Corrected board corners used for generation (TL, TR, BR, BL):"
        << std::endl;
    std::cout << "      TL: " << tl << ", TR: " << tr
              << ", BR: " << corrected_board_corners[2] << ", BL: " << bl
              << std::endl;
  }

  std::vector<double> lines;
  lines.reserve(target_count);

  if (is_generating_horizontal_lines) {
    // Generating Y-coordinates for horizontal lines
    float first_line_y = tl.y;
    float last_line_y =
        bl.y; // Use Bottom-Left Y for the extent of horizontal lines

    if (std::abs(last_line_y - first_line_y) <
        1.0f) { // Check for negligible height
      THROWGEMERROR("findUniformGridLines: Corrected board height is too small "
                    "for horizontal line generation.");
    }

    double spacing =
        static_cast<double>(last_line_y - first_line_y) / (target_count - 1.0);

    if (bDebug) {
      std::cout << "    Horizontal lines: start_y=" << first_line_y
                << ", end_y=" << last_line_y << ", spacing=" << spacing
                << std::endl;
    }

    for (int i = 0; i < target_count; ++i) {
      lines.push_back(static_cast<double>(first_line_y) + i * spacing);
    }
  } else {
    // Generating X-coordinates for vertical lines
    float first_line_x = tl.x;
    float last_line_x =
        tr.x; // Use Top-Right X for the extent of vertical lines

    if (std::abs(last_line_x - first_line_x) <
        1.0f) { // Check for negligible width
      THROWGEMERROR("findUniformGridLines: Corrected board width is too small "
                    "for vertical line generation.");
    }

    double spacing =
        static_cast<double>(last_line_x - first_line_x) / (target_count - 1.0);

    if (bDebug) {
      std::cout << "    Vertical lines: start_x=" << first_line_x
                << ", end_x=" << last_line_x << ", spacing=" << spacing
                << std::endl;
    }

    for (int i = 0; i < target_count; ++i) {
      lines.push_back(static_cast<double>(first_line_x) + i * spacing);
    }
  }

  if (lines.size() != static_cast<size_t>(target_count)) {
    // This should ideally not happen if logic is correct
    THROWGEMERROR(
        "findUniformGridLines: Failed to generate the target number of lines.");
  }

  return lines;
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
  // image here is expected to be the perspective-corrected BGR image
  if (bDebug) {
    std::cout << "Debug (detectUniformGrid): Using NEW calibrated method via "
                 "refactored findUniformGridLines."
              << std::endl;
  }

  if (image.empty()) {
    THROWGEMERROR("detectUniformGrid: Input image is empty.");
  }

  int corrected_width = image.cols;
  int corrected_height = image.rows;
  const int target_line_count = 19;

  // Call the refactored findUniformGridLines
  std::vector<double> final_horizontal_y = findUniformGridLines(
      target_line_count, bDebug, corrected_width, corrected_height,
      true); // true for is_generating_horizontal_lines

  std::vector<double> final_vertical_x = findUniformGridLines(
      target_line_count, bDebug, corrected_width, corrected_height,
      false); // false for is_generating_horizontal_lines

  // Basic validation
  if (final_horizontal_y.size() != target_line_count ||
      final_vertical_x.size() != target_line_count) {
    THROWGEMERROR(
        std::string("detectUniformGrid: Refactored findUniformGridLines failed "
                    "to return 19x19 lines. Got H:") +
        Num2Str(final_horizontal_y.size()).str() +
        ", V:" + Num2Str(final_vertical_x.size()).str());
  }

  if (bDebug) {
    std::cout << "  Debug (detectUniformGrid): Generated uniform horizontal "
                 "lines (y): ";
    for (double y : final_horizontal_y)
      std::cout << y << " ";
    std::cout << std::endl;
    std::cout << "  Debug (detectUniformGrid): Generated uniform vertical "
                 "lines (x): ";
    for (double x : final_vertical_x)
      std::cout << x << " ";
    std::cout << std::endl;

    // Optional: Draw these generated lines on a copy of the input image for
    // visualization
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
      cv::circle(grid_visualization_image, dbg_corrected_corners[0], 5,
                 cv::Scalar(0, 0, 255), -1); // TL Red
      cv::circle(grid_visualization_image, dbg_corrected_corners[1], 5,
                 cv::Scalar(0, 0, 255), -1); // TR Red
      cv::circle(grid_visualization_image, dbg_corrected_corners[2], 5,
                 cv::Scalar(0, 0, 255), -1); // BR Red
      cv::circle(grid_visualization_image, dbg_corrected_corners[3], 5,
                 cv::Scalar(0, 0, 255), -1); // BL Red
    }
    cv::imshow("Generated Grid on Corrected Image (detectUniformGrid)",
               grid_visualization_image);
    cv::waitKey(0);
  }

  return std::make_pair(final_horizontal_y, final_vertical_x);

  /*
  // --- OLD LOGIC THAT IS NOW BYPASSED ---
  Mat processed_image = preprocessImage(image, bDebug); // Now unused by this
  path vector<Vec4i> mixed_segments = detectLineSegments(processed_image,
  bDebug); // Now unused auto [horizontal_lines_raw, vertical_lines_raw] =
      convertSegmentsToLines(mixed_segments, bDebug); // Now unused

  auto [clustered_horizontal_y, clustered_vertical_x] = findOptimalClustering(
      horizontal_lines_raw, vertical_lines_raw, 19, bDebug); // Now unused

  // The old findUniformGridLines would have been called here with
  clustered_horizontal_y / clustered_vertical_x
  // e.g., findUniformGridLines(clustered_horizontal_y, 19,
  uniformity_tolerance, bDebug);
  // That call is now replaced by the direct calls to the NEW
  findUniformGridLines above.
  */
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

int calculateAdaptiveSampleRadius(float board_pixel_width,
                                  float board_pixel_height) {
  const float factor = 0.35f;
  if (board_pixel_width <= 0 || board_pixel_height <= 0) {
    if (bDebug)
      std::cerr << "Warning (calculateAdaptiveSampleRadius): Invalid board "
                   "dimensions ("
                << board_pixel_width << "x" << board_pixel_height
                << "). Defaulting radius to 3." << std::endl;
    return 3; // Default small radius
  }
  float avg_grid_spacing_x =
      board_pixel_width / 18.0f; // 18 spaces for 19 lines
  float avg_grid_spacing_y = board_pixel_height / 18.0f;
  float avg_grid_spacing = (avg_grid_spacing_x + avg_grid_spacing_y) * 0.5f;

  int radius = static_cast<int>(avg_grid_spacing * factor);
  radius = std::max(2, radius); // Ensure a minimum practical radius
  // Optional: Add a maximum cap if desired, e.g., radius = std::min(radius,
  // 10);

  if (bDebug) {
    std::cout << "  Debug (calculateAdaptiveSampleRadius): Board W="
              << board_pixel_width << ", H=" << board_pixel_height
              << ". Avg Grid Spacing X=" << avg_grid_spacing_x
              << ", Y=" << avg_grid_spacing_y << ". Calculated Radius (factor "
              << factor << "): " << radius << std::endl;
  }
  return radius;
}

// --- NEW HELPER: Classify a single intersection's Lab color using calibration
// data --- This was the smaller helper from the previous step, still useful.
static int classifySingleIntersectionByDistance(
    const cv::Vec3f &intersection_lab_color,
    const cv::Vec3f &avg_black_calib, // Calibrated avg L,a,b for black
    const cv::Vec3f &avg_white_calib, // Calibrated avg L,a,b for white
    const cv::Vec3f &board_calib_lab) // Calibrated avg L,a,b for board
{
  // --- Tunable Parameters LOCAL to this function ---
  // Weights for Lab distance calculation
  const float WEIGHT_L = 0.4f; // Your suggestion to de-emphasize L
  const float WEIGHT_A = 0.8f; // Your suggestion
  const float WEIGHT_B = 2.2f; // Your suggestion to emphasize B

  // Thresholds for WEIGHTED distances
  const float MAX_DIST_STONE_WEIGHTED =
      30.0f; // YOU WILL NEED TO TUNE THIS EXPERIMENTALLY
  const float MAX_DIST_BOARD_WEIGHTED =
      35.0f; // YOU WILL NEED TO TUNE THIS EXPERIMENTALLY

  // The black stones in test1.jpg had L ~28-36, ref_black_L ~73.5. Difference
  // is ~37-45. The white stones in test1.jpg had L ~168, ref_white_L ~200.
  // Difference is ~32.
  const float L_HEURISTIC_BLACK_OFFSET =
      30.0f; // If L is this much lower than avg_black_calib[0]
  const float L_HEURISTIC_WHITE_OFFSET =
      20.0f; // If L is this much higher than avg_white_calib[0]

  if (intersection_lab_color[0] < 0) { // Invalid sample
    THROWGEMERROR(
        "Error: Invalid Lab sample in classifySingleIntersectionByDistance.")
  }

  // --- Heuristic 1: Absolute L* checks (your suggestion) ---
  if (intersection_lab_color[0] <
      (avg_black_calib[0] - L_HEURISTIC_BLACK_OFFSET)) {
    if (bDebug)
      std::cout << "      L-Heuristic: Classified as BLACK (L="
                << intersection_lab_color[0]
                << " < BlackRefL=" << avg_black_calib[0] << " - "
                << L_HEURISTIC_BLACK_OFFSET << ")" << std::endl;
    return 1; // Black
  }
  if (intersection_lab_color[0] >
      (avg_white_calib[0] + L_HEURISTIC_WHITE_OFFSET)) {
    if (bDebug)
      std::cout << "      L-Heuristic: Classified as WHITE (L="
                << intersection_lab_color[0]
                << " > WhiteRefL=" << avg_white_calib[0] << " + "
                << L_HEURISTIC_WHITE_OFFSET << ")" << std::endl;
    return 2; // White
  }

  // --- Heuristic 2: Weighted Euclidean Distance ---
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
  int classification = 0;

  if (min_weighted_dist == dist_b_w && dist_b_w < MAX_DIST_STONE_WEIGHTED) {
    classification = 1;
  } else if (min_weighted_dist == dist_w_w &&
             dist_w_w < MAX_DIST_STONE_WEIGHTED) {
    classification = 2;
  } else if (min_weighted_dist == dist_empty_w &&
             dist_empty_w < MAX_DIST_BOARD_WEIGHTED) {
    classification = 0;
  } else {
    // If all weighted checks fail, it's uncertain.
    // You could add a fallback to unweighted if desired, or just default to
    // empty.
    classification = 0;
    if (bDebug) {
      std::cout << "      WeightedDist: All checks failed or distances too "
                   "high. MinWDist="
                << min_weighted_dist << ". Defaulting to Empty." << std::endl;
    }
  }
  return classification;
}

// --- NEW HELPER: Perform direct classification for all intersections (was
// performDirectClassification) --- Its debug output should be updated to show
// which thresholds and weights are being used.
static void classifyIntersectionsByCalibration(
    const std::vector<cv::Vec3f> &average_lab_values,
    const CalibrationData &calib_data,
    const std::vector<cv::Point2f> &intersection_points, int num_intersections,
    const cv::Mat &corrected_bgr_image, int adaptive_sample_radius_for_drawing,
    cv::Mat &board_state_output, cv::Mat &board_with_stones_output) {
  if (bDebug)
    std::cout << "  Debug (classifyIntersectionsByCalibration): Starting..."
              << std::endl;

  cv::Vec3f avg_black_calib = (calib_data.lab_tl + calib_data.lab_bl) * 0.5f;
  cv::Vec3f avg_white_calib = (calib_data.lab_tr + calib_data.lab_br) * 0.5f;

  board_state_output = cv::Mat(19, 19, CV_8U, cv::Scalar(0));
  board_with_stones_output = corrected_bgr_image.clone();

  if (bDebug) {
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "    Using Ref Black Lab: " << avg_black_calib << std::endl;
    std::cout << "    Using Ref White Lab: " << avg_white_calib << std::endl;
    std::cout << "    Using Ref Board Lab: " << calib_data.lab_board_avg
              << std::endl;
    // These constants are now inside classifySingleIntersectionByDistance,
    // but you can print their values here if you re-declare them or pass them.
    // For now, let's assume the user knows they are defined locally in the
    // helper. To print them, they would need to be accessible here (e.g.,
    // file-static const). For simplicity in this snippet, I'll omit printing
    // the local consts of the helper.
    std::cout << "    (Using L-heuristic and weighted distances with internal "
                 "thresholds/weights)"
              << std::endl;
  }

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

    if (bDebug && current_intersection_lab[0] >= 0) {
      // For logging, you might want to re-calculate the weighted distances
      // or modify classifySingleIntersectionByDistance to return them too for
      // logging. This is just for the unweighted for now.
      float dist_b_unweighted =
          cv::norm(current_intersection_lab, avg_black_calib, cv::NORM_L2);
      float dist_w_unweighted =
          cv::norm(current_intersection_lab, avg_white_calib, cv::NORM_L2);
      float dist_e_unweighted = cv::norm(current_intersection_lab,
                                         calib_data.lab_board_avg, cv::NORM_L2);
      std::string stone_type_str =
          (classification == 1)
              ? "Black"
              : (classification == 2 ? "White" : "Empty/Board");

      std::cout << "    Int [" << std::setw(2) << row << "," << std::setw(2)
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
    // ... (drawing circles as before, using adaptive_sample_radius_for_drawing)
    // ...
    if (static_cast<size_t>(i) < intersection_points.size()) {
      if (classification == 1) {
        cv::circle(board_with_stones_output, intersection_points[i],
                   adaptive_sample_radius_for_drawing, cv::Scalar(0, 0, 0), -1);
      } else if (classification == 2) {
        cv::circle(board_with_stones_output, intersection_points[i],
                   adaptive_sample_radius_for_drawing,
                   cv::Scalar(255, 255, 255), -1);
      } else if (current_intersection_lab[0] >=
                 0) { // Valid sample, classified as empty
        cv::circle(board_with_stones_output, intersection_points[i],
                   adaptive_sample_radius_for_drawing, cv::Scalar(0, 255, 0),
                   2); // Green for empty
      } else { // Invalid sample, already drawn orange if debug was on in
               // getAverageLab, or draw again
        cv::circle(board_with_stones_output, intersection_points[i],
                   adaptive_sample_radius_for_drawing, cv::Scalar(0, 165, 255),
                   2); // Orange for bad sample
      }
    }
  }
  if (bDebug) {
    imshow("Direct Classification Result (Helper)", board_with_stones_output);
    cv::waitKey(0);
  }
  if (bDebug)
    std::cout << "  Debug (classifyIntersectionsByCalibration): Finished."
              << std::endl;
}

// Function to process the Go board image and determine the board state
void processGoBoard(
    const cv::Mat &image_bgr_in,
    cv::Mat &board_state_out,                          // Renamed for clarity
    cv::Mat &board_with_stones_out,                    // Renamed for clarity
    std::vector<cv::Point2f> &intersection_points_out) // Renamed for clarity
{
  if (bDebug)
    std::cout << "Debug (processGoBoard): Starting board processing."
              << std::endl;

  // 1. Load Calibration Data
  CalibrationData calib_data = loadCalibrationData(CALIB_CONFIG_PATH);
  if (!calib_data.corners_loaded || !calib_data.colors_loaded ||
      !calib_data.board_color_loaded || !calib_data.dimensions_loaded) {
    std::string err_msg = "ProcessGoBoard Error: Incomplete calibration data. ";
    if (!calib_data.corners_loaded)
      err_msg += "Corners missing. ";
    if (!calib_data.colors_loaded)
      err_msg += "Stone colors missing. ";
    if (!calib_data.board_color_loaded)
      err_msg += "Board color missing. ";
    if (!calib_data.dimensions_loaded)
      err_msg += "Dimensions missing. ";
    err_msg += "Please run calibration (-b) ensuring stones and empty board "
               "are correctly placed.";
    THROWGEMERROR(err_msg);
  }
  if (bDebug)
    std::cout << "  Debug: Full calibration data loaded." << std::endl;

  // 2. Perspective Correction
  cv::Mat image_bgr_corrected = correctPerspective(image_bgr_in);
  if (image_bgr_corrected.empty()) {
    THROWGEMERROR("Corrected perspective image is empty.");
  }
  if (bDebug) {
    imshow("Corrected Perspective", image_bgr_corrected);
    cv::waitKey(0);
  }

  // 3. Convert to Lab and Detect Grid
  cv::Mat image_lab;
  cv::cvtColor(image_bgr_corrected, image_lab, cv::COLOR_BGR2Lab);
  if (image_lab.empty()) {
    THROWGEMERROR("Lab converted image is empty.");
  }
  if (bDebug) {
    imshow("Lab Image", image_lab);
    cv::waitKey(0);
  }

  std::pair<std::vector<double>, std::vector<double>> grid_lines =
      detectUniformGrid(image_bgr_corrected);
  std::vector<double> horizontal_lines = grid_lines.first;
  std::vector<double> vertical_lines = grid_lines.second;

  intersection_points_out = findIntersections(horizontal_lines, vertical_lines);
  int num_intersections = intersection_points_out.size();
  if (num_intersections != 361 && image_bgr_corrected.cols > 0 &&
      image_bgr_corrected.rows > 0) {
    std::cerr << "Warning (processGoBoard): Expected 361 intersections, found "
              << num_intersections << "." << std::endl;
    if (num_intersections == 0)
      THROWGEMERROR("No intersection points found.");
  } else if (num_intersections == 0) {
    THROWGEMERROR("No intersection points found (image might be invalid).");
  }
  if (bDebug)
    std::cout << "  Debug: Found " << num_intersections
              << " intersection points." << std::endl;

  // 4. Sample Lab Color at Each Intersection
  float board_pixel_width_corrected = 0;
  if (!vertical_lines.empty()) {
    board_pixel_width_corrected =
        std::abs(vertical_lines.back() - vertical_lines.front());
  }
  float board_pixel_height_corrected = 0;
  if (!horizontal_lines.empty()) {
    board_pixel_height_corrected =
        std::abs(horizontal_lines.back() - horizontal_lines.front());
  }

  int adaptive_sample_radius = calculateAdaptiveSampleRadius(
      board_pixel_width_corrected, board_pixel_height_corrected);

  if (bDebug)
    std::cout << "width:" << board_pixel_width_corrected
              << "\theight:" << board_pixel_height_corrected
              << "  Debug: Image processing using adaptive_sample_radius: "
              << adaptive_sample_radius << std::endl;

  std::vector<cv::Vec3f> average_lab_values(num_intersections);
  for (int i = 0; i < num_intersections; ++i) {
    average_lab_values[i] = getAverageLab(image_lab, intersection_points_out[i],
                                          adaptive_sample_radius);
  }
  if (bDebug)
    std::cout << "  Debug: Sampled Lab colors for all intersections."
              << std::endl;

  // --- 5. Call the new helper function for Direct Classification ---
  classifyIntersectionsByCalibration(
      average_lab_values, calib_data, intersection_points_out,
      num_intersections,
      image_bgr_corrected, // Pass the corrected BGR image for drawing
      adaptive_sample_radius,
      board_state_out,      // Output: board_state
      board_with_stones_out // Output: board_with_stones
  );
  // The imshow for "Direct Classification Result" is now inside
  // performDirectClassification if bDebug

  if (bDebug) {
    imshow("board_with_stones_out (Final)", board_with_stones_out);
    cv::waitKey(0);
  }
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
    std::cerr << "Error (drawSimulatedGoBoard): Canvas size " << canvas_size_px
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
    std::cout << "Debug (drawSimulatedGoBoard): Displaying board state after "
              << display_up_to_move_idx << " B/W moves." << std::endl;
    if (highlight_this_move_idx != -1) {
      std::cout << "Debug (drawSimulatedGoBoard): Highlighting move index "
                << highlight_this_move_idx << std::endl;
    }
  }
}

static std::vector<cv::Point2f>
getStoneCandidateCenters(const cv::Mat &image_for_size_ref,
                         const std::vector<std::vector<cv::Point>> &contours,
                         double min_area_ratio, double max_area_ratio,
                         double min_circularity) {

  std::vector<cv::Point2f> centers;
  if (contours.empty()) {
    return centers;
  }

  double image_area =
      static_cast<double>(image_for_size_ref.rows * image_for_size_ref.cols);
  double min_abs_area = image_area * min_area_ratio;
  double max_abs_area = image_area * max_area_ratio;

  if (bDebug) {
    std::cout << "    getCandidateCenters: Image Area=" << image_area
              << ", MinStoneArea=" << min_abs_area
              << ", MaxStoneArea=" << max_abs_area << std::endl;
  }

  for (const auto &contour : contours) {
    double area = cv::contourArea(contour);
    if (area < min_abs_area || area > max_abs_area) {
      if (bDebug && area > 10)
        std::cout << "      Contour rejected by area: " << area << std::endl;
      continue;
    }

    if (contour.size() < 5) {
      if (bDebug)
        std::cout << "      Contour rejected by insufficient points: "
                  << contour.size() << std::endl;
      continue;
    }

    double perimeter = cv::arcLength(contour, true);
    if (perimeter < 1.0)
      continue;

    double circularity = (4 * CV_PI * area) / (perimeter * perimeter);
    if (circularity < min_circularity) {
      if (bDebug)
        std::cout << "      Contour rejected by circularity: " << circularity
                  << " (Area: " << area << ")" << std::endl;
      continue;
    }

    cv::Moments M = cv::moments(contour);
    if (M.m00 > 0) {
      cv::Point2f center(static_cast<float>(M.m10 / M.m00),
                         static_cast<float>(M.m01 / M.m00));
      centers.push_back(center);
      if (bDebug)
        std::cout << "      Contour accepted: Center=" << center
                  << ", Area=" << area << ", Circ=" << circularity << std::endl;
    }
  }
  return centers;
}

bool detectSpecificColoredRoundShape(
    const cv::Mat &inputBgrImage, const cv::Rect &regionOfInterest,
    const cv::Vec3f &expectedAvgLabColor, // Specific target Lab color
    float l_tolerance,                    // Tolerance for L channel
    float ab_tolerance,                   // Tolerance for A and B channels
    cv::Point2f &detectedCenter, float &detectedRadius) {

  // ... (initial checks, Lab conversion, color masking, morphology -
  // unchanged) ...
  if (inputBgrImage.empty()) {
    if (bDebug)
      std::cerr << "Error (detectSpecificColoredRoundShape): Input BGR image "
                   "is empty."
                << std::endl;
    return false;
  }

  cv::Rect roi =
      regionOfInterest & cv::Rect(0, 0, inputBgrImage.cols, inputBgrImage.rows);
  if (roi.width <= 0 || roi.height <= 0) {
    if (bDebug)
      std::cerr << "Error (detectSpecificColoredRoundShape): Region of "
                   "interest is invalid or outside image bounds. ROI: "
                << regionOfInterest << ", Image: " << inputBgrImage.cols << "x"
                << inputBgrImage.rows << std::endl;
    return false;
  }

  cv::Mat roiImageBgr = inputBgrImage(roi);
  cv::Mat roiImageLab;
  cv::cvtColor(roiImageBgr, roiImageLab, cv::COLOR_BGR2Lab);

  cv::Mat colorMask;
  cv::Scalar labLower =
      cv::Scalar(std::max(0.f, expectedAvgLabColor[0] - l_tolerance),
                 std::max(0.f, expectedAvgLabColor[1] - ab_tolerance),
                 std::max(0.f, expectedAvgLabColor[2] - ab_tolerance));
  cv::Scalar labUpper =
      cv::Scalar(std::min(255.f, expectedAvgLabColor[0] + l_tolerance),
                 std::min(255.f, expectedAvgLabColor[1] + ab_tolerance),
                 std::min(255.f, expectedAvgLabColor[2] + ab_tolerance));

  if (bDebug) {
    std::cout << "  Debug (detectSpecificColoredRoundShape for ROI at "
              << roi.tl() << "): Target Lab: " << expectedAvgLabColor
              << ", L_tol: " << l_tolerance << ", AB_tol: " << ab_tolerance
              << std::endl;
    std::cout << "    Calculated Lab Lower: " << labLower
              << ", Lab Upper: " << labUpper << std::endl;
  }

  cv::inRange(roiImageLab, labLower, labUpper, colorMask);

  cv::Mat open_kernel = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, cv::Size(5, 5)); // Try a smaller kernel first
  cv::Mat close_kernel =
      cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));

  // MORPH_OPEN to remove small noise and break thin connections
  cv::morphologyEx(colorMask, colorMask, cv::MORPH_OPEN, open_kernel,
                   cv::Point(-1, -1), 2); // Try 1 or 2 iterations

  // MORPH_CLOSE to fill small holes within the stone
  cv::morphologyEx(colorMask, colorMask, cv::MORPH_CLOSE, close_kernel,
                   cv::Point(-1, -1),
                   1); // Maybe reduce to 1 iteration if opening is aggressive

  if (bDebug) {
    std::string roi_window_name =
        "Color Mask ROI (L:" +
        std::to_string(static_cast<int>(expectedAvgLabColor[0])) + " X" +
        std::to_string(roi.x) + ")";
    cv::imshow(roi_window_name, colorMask);
    waitKey(0);
  }

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(colorMask, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  if (bDebug) {
    std::cout << "  Debug (detectSpecificColoredRoundShape for ROI at "
              << roi.tl() << "): Found " << contours.size()
              << " raw contours in ROI (" << roi.x << "," << roi.y << " "
              << roi.width << "x" << roi.height << ")." << std::endl;
  }

  if (contours.empty()) {
    if (bDebug)
      std::cout << "  Debug (detectSpecificColoredRoundShape for ROI at "
                << roi.tl() << "): No contours " << "found after color masking."
                << std::endl;
    return false;
  }

  double roiArea = static_cast<double>(roi.width * roi.height);
  double minStoneAreaInRoi = roiArea * 0.03;
  double maxStoneAreaInRoi = roiArea * 0.70;
  double minCircularity = 0.65; // Current threshold

  if (bDebug) {
    std::cout << "  Debug (detectSpecificColoredRoundShape for ROI at "
              << roi.tl() << "): ROI Area=" << roiArea
              << ", MinStoneAreaInRoi=" << minStoneAreaInRoi
              << ", MaxStoneAreaInRoi=" << maxStoneAreaInRoi
              << ", MinCircularity=" << minCircularity << std::endl;
  }

  std::vector<cv::Point> bestContour;
  double bestContourScore = 0.0f;
  int contour_idx = 0;

  // Prepare a visualization canvas for this ROI's contours if debugging
  cv::Mat roi_contour_vis_canvas;
  if (bDebug) {
    roi_contour_vis_canvas = roiImageBgr.clone();
  }

  for (const auto &contour : contours) {
    contour_idx++;
    if (contour.size() < 5) {
      if (bDebug)
        std::cout << "    Contour " << contour_idx << " (ROI " << roi.tl()
                  << ") rejected: too few points (" << contour.size() << ")."
                  << std::endl;
      continue;
    }

    double area = cv::contourArea(contour);
    double perimeter = cv::arcLength(contour, true);

    if (perimeter < 1.0) {
      if (bDebug)
        std::cout << "    Contour " << contour_idx << " (ROI " << roi.tl()
                  << ") rejected: perimeter too small (" << perimeter << ")."
                  << std::endl;
      continue;
    }

    double circularity = (4 * CV_PI * area) / (perimeter * perimeter);

    if (bDebug) {
      std::cout << "    Contour " << contour_idx << " (ROI " << roi.tl()
                << ") evaluating: " << "Area=" << area
                << " (Thresh: " << minStoneAreaInRoi << "-" << maxStoneAreaInRoi
                << "), " << "Circularity=" << circularity
                << " (Thresh: >=" << minCircularity << ")." << std::endl;
    }

    if (area < minStoneAreaInRoi || area > maxStoneAreaInRoi) {
      if (bDebug) {
        std::cout << "      -> Contour " << contour_idx << " (ROI " << roi.tl()
                  << ") REJECTED by area."
                  << (area < minStoneAreaInRoi ? " (too small)"
                                               : " (too large)")
                  << std::endl;
        cv::drawContours(roi_contour_vis_canvas,
                         std::vector<std::vector<cv::Point>>{contour}, -1,
                         cv::Scalar(0, 165, 255), 1); // Orange for area fail
      }
      continue;
    }

    if (circularity < minCircularity) {
      if (bDebug) {
        std::cout << "      -> Contour " << contour_idx << " (ROI " << roi.tl()
                  << ") REJECTED by circularity." << std::endl;
        // Draw the contour that failed circularity check in blue
        cv::drawContours(roi_contour_vis_canvas,
                         std::vector<std::vector<cv::Point>>{contour}, -1,
                         cv::Scalar(255, 0, 0), 1); // Blue for circ fail
        cv::putText(roi_contour_vis_canvas,
                    "C:" + std::to_string(circularity).substr(0, 4),
                    contour[0] - cv::Point(0, 5), cv::FONT_HERSHEY_SIMPLEX, 0.3,
                    cv::Scalar(255, 100, 100));
      }
      continue;
    }

    if (bDebug) {
      std::cout << "      => Contour " << contour_idx << " (ROI " << roi.tl()
                << ") PASSED filters. Current best score: " << bestContourScore
                << std::endl;
      // Draw passed contours in a tentative color like yellow
      cv::drawContours(roi_contour_vis_canvas,
                       std::vector<std::vector<cv::Point>>{contour}, -1,
                       cv::Scalar(0, 255, 255), 1);
    }

    if (area > bestContourScore) {
      bestContourScore = area;
      bestContour = contour;
      if (bDebug)
        std::cout << "        ==> New best contour " << contour_idx << " (ROI "
                  << roi.tl() << ") with area " << area << std::endl;
    }
  }

  if (bDebug && !contours.empty()) { // Show the canvas if we had contours
    std::string roi_contours_win_name = "Evaluated Contours (ROI X" +
                                        std::to_string(roi.x) + " Y" +
                                        std::to_string(roi.y) + ")";
    if (!bestContour.empty()) {
      // Highlight best contour in Green
      cv::drawContours(roi_contour_vis_canvas,
                       std::vector<std::vector<cv::Point>>{bestContour}, -1,
                       cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow(roi_contours_win_name, roi_contour_vis_canvas);
    waitKey(0);
  }

  if (!bestContour.empty()) {
    cv::Point2f centerInRoi;
    float radiusInRoi;
    cv::minEnclosingCircle(bestContour, centerInRoi, radiusInRoi);
    detectedCenter.x = centerInRoi.x + roi.x;
    detectedCenter.y = centerInRoi.y + roi.y;
    detectedRadius = radiusInRoi;
    if (bDebug)
      std::cout << "  Debug (detectSpecificColoredRoundShape for ROI at "
                << roi.tl() << "): Found specific "
                << "stone. Center (orig img): " << detectedCenter
                << ", Radius: " << detectedRadius << std::endl;
    return true;
  }
  if (bDebug)
    std::cout << "  Debug (detectSpecificColoredRoundShape for ROI at "
              << roi.tl() << "): No suitable "
              << "specific stone contour found in ROI after filtering."
              << std::endl;
  return false;
}

// Utility function to detect a single colored round shape in a region
// Utility function to detect a single colored round shape in a region
// Output order for detected_corners_tl_tr_br_bl: TL, TR, BR, BL
// image.cpp
// ... (other functions like loadCalibrationData, getBoardCornersCorrected,
// detectSpecificColoredRoundShape etc.)

bool detectFourCornersGoBoard(
    const cv::Mat &input_bgr_image, // Expected to be CALIB_SNAPSHOT_RAW_PATH
    std::vector<cv::Point2f> &found_corrected_centers) {
  if (bDebug)
    std::cout
        << "Debug (detectFourCornersGoBoard - Config-Refined): Starting..."
        << std::endl;

  // ... (Initial checks, load calib_data, dimension check, perspective
  // transform - as in your current image.cpp) ...
  found_corrected_centers.clear();

  if (input_bgr_image.empty()) {
    std::cerr << "Error (detectFourCornersGoBoard): Input image is empty."
              << std::endl;
    return false;
  }

  CalibrationData calib_data = loadCalibrationData(CALIB_CONFIG_PATH);
  if (!calib_data.corners_loaded || !calib_data.colors_loaded ||
      !calib_data.dimensions_loaded) {
    std::cerr << "Error (detectFourCornersGoBoard): Essential calibration data "
                 "(corners, colors, or dimensions) not loaded from config.txt. "
                 "Cannot perform refined detection."
              << std::endl;
    return false;
  }

  if (input_bgr_image.cols != calib_data.image_width ||
      input_bgr_image.rows != calib_data.image_height) {
    std::cerr << "Error (detectFourCornersGoBoard): Input image dimensions ("
              << input_bgr_image.cols << "x" << input_bgr_image.rows
              << ") do not match calibration config dimensions ("
              << calib_data.image_width << "x" << calib_data.image_height
              << ")." << std::endl;
    return false;
  }

  std::vector<cv::Point2f> dst_points =
      getBoardCornersCorrected(input_bgr_image.cols, input_bgr_image.rows);

  cv::Mat corrected_bgr_image = input_bgr_image.clone();

  bool all_found = true;

  float line_spacing_corrected =
      (dst_points[1].x - dst_points[0].x) /
      18.0f; // Using x-spacing from corrected TL to TR
  float expected_stone_radius_corrected = line_spacing_corrected * 0.45f;
  int correct_board_width_px = dst_points[1].x - dst_points[0].x;
  int correct_board_height_px = dst_points[3].y - dst_points[0].y;
  int roi_half_width =
      1.5f * calculateAdaptiveSampleRadius(correct_board_width_px,
                                           correct_board_height_px);

  // Default Lab tolerances from previous version
  const float L_TOLERANCE_STONE = 35.0f;
  const float AB_TOLERANCE_STONE = 30.0f;

  cv::Point2f temp_center;
  float temp_radius;

  // --- TL (Black) using calib_data.lab_tl ---
  cv::Rect roi_tl(static_cast<int>(dst_points[0].x - roi_half_width),
                  static_cast<int>(dst_points[0].y - roi_half_width),
                  static_cast<int>(roi_half_width * 2),
                  static_cast<int>(roi_half_width * 2));
  if (detectSpecificColoredRoundShape(
          corrected_bgr_image, roi_tl, calib_data.lab_tl, L_TOLERANCE_STONE,
          AB_TOLERANCE_STONE, temp_center, temp_radius)) {
    found_corrected_centers[0] = temp_center;
  } else {
    all_found = false;
    if (bDebug)
      std::cerr << "  Failed to find TL stone using specific Lab target: "
                << calib_data.lab_tl << std::endl;
  }

  // --- TR (White) using calib_data.lab_tr ---
  cv::Rect roi_tr(static_cast<int>(dst_points[1].x - roi_half_width),
                  static_cast<int>(dst_points[1].y - roi_half_width),
                  static_cast<int>(roi_half_width * 2),
                  static_cast<int>(roi_half_width * 2));
  if (all_found &&
      detectSpecificColoredRoundShape(
          corrected_bgr_image, roi_tr, calib_data.lab_tr, L_TOLERANCE_STONE,
          AB_TOLERANCE_STONE, temp_center, temp_radius)) {
    found_corrected_centers[1] = temp_center;
  } else {
    all_found = false; // Keep this logic to ensure all_found becomes false if
                       // any step fails
    if (bDebug)
      std::cerr << "  Failed to find TR stone using specific Lab target: "
                << calib_data.lab_tr << std::endl;
  }

  // --- BR (White) using calib_data.lab_br ---
  cv::Rect roi_br(static_cast<int>(dst_points[2].x - roi_half_width),
                  static_cast<int>(dst_points[2].y - roi_half_width),
                  static_cast<int>(roi_half_width * 2),
                  static_cast<int>(roi_half_width * 2));

  // ***** ADDED DEBUG FOR BR TARGET COLOR VERIFICATION *****
  if (bDebug &&
      all_found) { // Only sample if prior steps were okay, to avoid confusion
    cv::Point2f ideal_br_center_corrected = dst_points[2]; // BR is index 2
    // Use a smaller radius for sampling to be more precise on the stone
    int sample_radius_debug =
        std::max(1, static_cast<int>(expected_stone_radius_corrected * 0.20f));

    cv::Mat corrected_lab_temp_for_br_sample;
    cv::cvtColor(corrected_bgr_image, corrected_lab_temp_for_br_sample,
                 cv::COLOR_BGR2Lab);

    cv::Vec3f actual_br_stone_lab_sample =
        Vec3f(-1, -1, -1); // Default to invalid
    if (ideal_br_center_corrected.x - sample_radius_debug >= 0 &&
        ideal_br_center_corrected.y - sample_radius_debug >= 0 &&
        ideal_br_center_corrected.x + sample_radius_debug <
            corrected_lab_temp_for_br_sample.cols &&
        ideal_br_center_corrected.y + sample_radius_debug <
            corrected_lab_temp_for_br_sample.rows) {
      actual_br_stone_lab_sample =
          getAverageLab(corrected_lab_temp_for_br_sample,
                        ideal_br_center_corrected, sample_radius_debug);
    }

    std::cout << "  DEBUG_BR_COLOR_VERIFY: Config BR Lab (calib_data.lab_br): "
              << calib_data.lab_br
              << ", Current Sampled at ideal BR in corrected_bgr_image (radius "
              << sample_radius_debug << "): " << actual_br_stone_lab_sample
              << std::endl;

    cv::Mat br_roi_display_for_sample_check =
        corrected_bgr_image.clone(); // Draw on a fresh clone
    cv::rectangle(br_roi_display_for_sample_check, roi_br,
                  cv::Scalar(255, 0, 0),
                  1); // Draw the actual ROI used for detection
    cv::circle(br_roi_display_for_sample_check, ideal_br_center_corrected,
               sample_radius_debug, cv::Scalar(0, 255, 0),
               1); // Green circle for sampling area
    cv::circle(br_roi_display_for_sample_check, ideal_br_center_corrected, 2,
               cv::Scalar(0, 0, 255), -1); // Red dot for center
    cv::imshow("BR ROI - Ideal Center Sampling Check",
               br_roi_display_for_sample_check);
    // cv::waitKey(0); // Add waitKey if you want to pause here specifically
  }
  // ***** END OF ADDED DEBUG *****

  if (all_found &&
      detectSpecificColoredRoundShape(
          corrected_bgr_image, roi_br, calib_data.lab_br,
          L_TOLERANCE_STONE, // Still using general tolerances for now
          AB_TOLERANCE_STONE, temp_center, temp_radius)) {
    found_corrected_centers[2] = temp_center;
  } else {
    all_found = false;
    if (bDebug)
      std::cerr << "  Failed to find BR stone using specific Lab target: "
                << calib_data.lab_br << std::endl;
  }

  // --- BL (Black) using calib_data.lab_bl ---
  cv::Rect roi_bl(static_cast<int>(dst_points[3].x - roi_half_width),
                  static_cast<int>(dst_points[3].y - roi_half_width),
                  static_cast<int>(roi_half_width * 2),
                  static_cast<int>(roi_half_width * 2));
  if (all_found &&
      detectSpecificColoredRoundShape(
          corrected_bgr_image, roi_bl, calib_data.lab_bl, L_TOLERANCE_STONE,
          AB_TOLERANCE_STONE, temp_center, temp_radius)) {
    found_corrected_centers[3] = temp_center;
  } else {
    all_found = false;
    if (bDebug)
      std::cerr << "  Failed to find BL stone using specific Lab target: "
                << calib_data.lab_bl << std::endl;
  }

  if (all_found) {
    if (bDebug) {
      std::cout
          << "  Refined config-guided detection successful. Raw corners (TL, "
             "TR, BR, BL):"
          << std::endl;
      for (size_t i = 0; i < found_corrected_centers.size(); ++i) {
        std::cout << "    Corner " << i << ": " << found_corrected_centers[i]
                  << std::endl;
      }
      cv::Mat debug_raw_img_final = input_bgr_image.clone();
      cv::circle(debug_raw_img_final, found_corrected_centers[0], 7,
                 cv::Scalar(0, 255, 0), 2); // TL
      cv::circle(debug_raw_img_final, found_corrected_centers[1], 7,
                 cv::Scalar(0, 255, 0), 2); // TR
      cv::circle(debug_raw_img_final, found_corrected_centers[2], 7,
                 cv::Scalar(0, 255, 0), 2); // BR
      cv::circle(debug_raw_img_final, found_corrected_centers[3], 7,
                 cv::Scalar(0, 255, 0), 2); // BL
      cv::imshow("detectFourCornersGoBoard - Final Refined Raw Points",
                 debug_raw_img_final);
    }
    return true;
  } else {
    std::cerr << "Error (detectFourCornersGoBoard - Config-Refined): Failed to "
                 "pinpoint all four corner stones using specific Lab targets."
              << std::endl;
    return false;
  }
}

// ... (rest of image.cpp, including detectSpecificColoredRoundShape,
// processGoBoard etc. from previous versions)