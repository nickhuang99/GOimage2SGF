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


const float MAX_DIST_STONE_CALIB_PHASE3 =
    20.0f; // Max Lab distance for a sample to be considered a stone matching
           // its calibrated color
const float MAX_DIST_BOARD_CALIB_PHASE3 =
    40.0f; // Max Lab distance for a sample to be considered board matching its
           // calibrated color

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

int calculateAdaptiveSampleRadius(float board_pixel_width,
                                  float board_pixel_height, float factor) {
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
    const cv::Vec3f &intersection_lab_color, const cv::Vec3f &avg_black_calib,
    const cv::Vec3f &avg_white_calib, const cv::Vec3f &board_calib_lab) {
  if (intersection_lab_color[0] <
      0) { // Check for invalid sample from getAverageLab
    THROWGEMERROR(std::string("color cannot be negative ") +
                  Num2Str(intersection_lab_color[0]).str());
  }

  float dist_b = cv::norm(intersection_lab_color, avg_black_calib, cv::NORM_L2);
  float dist_w = cv::norm(intersection_lab_color, avg_white_calib, cv::NORM_L2);
  float dist_empty =
      cv::norm(intersection_lab_color, board_calib_lab, cv::NORM_L2);

  float min_dist = std::min({dist_b, dist_w, dist_empty});  

  // Is it a black stone? (Closest to black AND within black threshold AND
  // significantly far from others)
  if (dist_b < MAX_DIST_STONE_CALIB_PHASE3 && dist_b <= dist_w &&
      dist_b <= dist_empty) {
    // Optional: add a further check if it's too close to board e.g. dist_b <
    // dist_empty * 0.7
    return 1; // Black
  }
  // Is it a white stone? (Closest to white AND within white threshold AND
  // significantly far from others)
  if (dist_w < MAX_DIST_STONE_CALIB_PHASE3 && dist_w <= dist_b &&
      dist_w <= dist_empty) {
    // Optional: add a further check if it's too close to board e.g. dist_w <
    // dist_empty * 0.7
    return 2; // White
  }
  // Is it an empty board point? (Closest to board AND within board threshold)
  if (dist_empty < MAX_DIST_BOARD_CALIB_PHASE3 && dist_empty <= dist_b &&
      dist_empty <= dist_w) {
    return 0; // Empty
  }
  return 0;
}

// --- NEW HELPER: Perform direct classification for all intersections (was performDirectClassification) ---
static void classifyIntersectionsByCalibration( // Renamed as per your request
  const std::vector<cv::Vec3f>& average_lab_values,
  const CalibrationData& calib_data,
  const std::vector<cv::Point2f>& intersection_points,
  int num_intersections,
  const cv::Mat& corrected_bgr_image,  // Changed from original_bgr_image_for_drawing
  cv::Mat& board_state_output,
  cv::Mat& board_with_stones_output)
{
  if (bDebug)
      std::cout << "  Debug (classifyIntersectionsByCalibration): Starting..." << std::endl;

  cv::Vec3f avg_black_calib = (calib_data.lab_tl + calib_data.lab_bl) * 0.5f;
  cv::Vec3f avg_white_calib = (calib_data.lab_tr + calib_data.lab_br) * 0.5f;
  // calib_data.lab_board_avg is already the averaged board color

  board_state_output = cv::Mat(19, 19, CV_8U, cv::Scalar(0)); 
  board_with_stones_output = corrected_bgr_image.clone(); // Use the corrected image as base

  if (bDebug) {
      std::cout << std::fixed << std::setprecision(1);
      std::cout << "    Using Ref Black Lab: " << avg_black_calib << std::endl;
      std::cout << "    Using Ref White Lab: " << avg_white_calib << std::endl;
      std::cout << "    Using Ref Board Lab: " << calib_data.lab_board_avg << std::endl;
      std::cout << "    Using Thresholds: Stone=" << MAX_DIST_STONE_CALIB_PHASE3
                << ", Board=" << MAX_DIST_BOARD_CALIB_PHASE3 << std::endl;
  }

  for (int i = 0; i < num_intersections; ++i) {
      int row = i / 19;
      int col = i % 19;
      if (row >= 19 || col >= 19) continue;

      cv::Vec3f current_intersection_lab = average_lab_values[i];
      int classification;

      if (current_intersection_lab[0] < 0) { 
          if (bDebug) std::cout << "    Intersection ["<< std::setw(2) << row << "," << std::setw(2) << col << "] - Invalid Lab sample (-1). Classifying as Empty." << std::endl;
          classification = 0; 
          if (!intersection_points.empty() && static_cast<size_t>(i) < intersection_points.size()) {
               cv::circle(board_with_stones_output, intersection_points[i], 8, cv::Scalar(0, 128, 255), 2); 
          }
      } else {
          classification = classifySingleIntersectionByDistance(
              current_intersection_lab,
              avg_black_calib,
              avg_white_calib,
              calib_data.lab_board_avg
          );
      }
      
      board_state_output.at<uchar>(row, col) = classification;

      if (bDebug && current_intersection_lab[0] >= 0) {
          float dist_b = cv::norm(current_intersection_lab, avg_black_calib, cv::NORM_L2);
          float dist_w = cv::norm(current_intersection_lab, avg_white_calib, cv::NORM_L2);
          float dist_empty = cv::norm(current_intersection_lab, calib_data.lab_board_avg, cv::NORM_L2);
          std::string stone_type_str = (classification == 1) ? "Black" : (classification == 2 ? "White" : "Empty/Board");
          
          std::cout << "    Int [" << std::setw(2) << row << "," << std::setw(2) << col
                    << "] Lab: [" << std::setw(5) << std::fixed << std::setprecision(1) << current_intersection_lab[0] 
                    << "," << std::setw(5) << current_intersection_lab[1] 
                    << "," << std::setw(5) << current_intersection_lab[2] << "]"
                    << " D(B):" << std::setw(5) << std::setprecision(1) << dist_b  // Added precision for distances
                    << " D(W):" << std::setw(5) << std::setprecision(1) << dist_w 
                    << " D(E):" << std::setw(5) << std::setprecision(1) << dist_empty
                    << " -> Class: " << stone_type_str << " (" << classification << ")" << std::endl;
      }

      if (static_cast<size_t>(i) < intersection_points.size()) { 
          if (classification == 1) {
              cv::circle(board_with_stones_output, intersection_points[i], 8, cv::Scalar(0, 0, 0), -1);
          } else if (classification == 2) {
              cv::circle(board_with_stones_output, intersection_points[i], 8, cv::Scalar(255, 255, 255), -1);
          } else if (current_intersection_lab[0] >= 0) { 
              cv::circle(board_with_stones_output, intersection_points[i], 8, cv::Scalar(0, 255, 0), 2);
          }
      }
  }

  if (bDebug) {
      imshow("Direct Classification Result (Helper)", board_with_stones_output);
      cv::waitKey(1); 
  }
  if (bDebug) std::cout << "  Debug (classifyIntersectionsByCalibration): Finished." << std::endl;
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
    cv::waitKey(1);
  }

  // 3. Convert to Lab and Detect Grid
  cv::Mat image_lab;
  cv::cvtColor(image_bgr_corrected, image_lab, cv::COLOR_BGR2Lab);
  if (image_lab.empty()) {
    THROWGEMERROR("Lab converted image is empty.");
  }
  if (bDebug) {
    imshow("Lab Image", image_lab);
    cv::waitKey(1);
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
      board_pixel_width_corrected, board_pixel_height_corrected,
      0.25f); // Use factor 0.25
  if (bDebug)
    std::cout << "  Debug: Image processing using adaptive_sample_radius: "
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
      image_bgr_corrected,  // Pass the corrected BGR image for drawing
      board_state_out,      // Output: board_state
      board_with_stones_out // Output: board_with_stones
  );
  // The imshow for "Direct Classification Result" is now inside
  // performDirectClassification if bDebug

  // --- 6. Post-Processing Filter ---
  if (bDebug)
    std::cout << "  Debug: Applying post-processing filter." << std::endl;
  // Ensure we have the expected number of points for 19x19 indexing
  if (num_intersections == 361) {
    cv::Mat temp_board_state = board_state_out.clone();
    for (int r = 0; r < 19; ++r) {
      for (int c = 0; c < 19; ++c) {
        if (temp_board_state.at<uchar>(r, c) == 2) { // If white
          bool has_stone_neighbor = false;
          for (int dr = -1; dr <= 1; ++dr) {
            for (int dc = -1; dc <= 1; ++dc) {
              if (dr == 0 && dc == 0)
                continue;
              int nr = r + dr;
              int nc = c + dc;
              if (nr >= 0 && nr < 19 && nc >= 0 && nc < 19) {
                if (temp_board_state.at<uchar>(nr, nc) == 1 ||
                    temp_board_state.at<uchar>(nr, nc) == 2) {
                  has_stone_neighbor = true;
                  break;
                }
              }
            }
            if (has_stone_neighbor)
              break;
          }
          if (!has_stone_neighbor) {
            board_state_out.at<uchar>(r, c) = 0;
            // Ensure index is valid before drawing
            int intersection_idx = r * 19 + c;
            if (intersection_idx < intersection_points_out.size()) {
              cv::circle(board_with_stones_out,
                         intersection_points_out[intersection_idx], 8,
                         cv::Scalar(0, 255, 0), 2);
            }
          }
        }
      }
    }
  } else if (bDebug) {
    std::cout << "  Debug: Skipping post-processing filter due to non-standard "
                 "number of intersections ("
              << num_intersections << ")." << std::endl;
  }

  if (bDebug) {
    imshow("Filtered Stones (Final)", board_with_stones_out);
    cv::waitKey(0);
  }
  if (bDebug)
    std::cout << "Debug (processGoBoard): Board processing finished."
              << std::endl;
}

