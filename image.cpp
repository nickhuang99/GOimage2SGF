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

extern bool bDebug;
struct Line {
  double value; // y for horizontal, x for vertical
  double angle;
};

bool compareLines(const Line &a, const Line &b) { return a.value < b.value; }


// Function to find the corners of the Go board
vector<Point2f> getBoardCorners(const Mat &inputImage) {

  float tl_x_percent = 20.0f; // Original: 19.0f
  float tl_y_percent = 8.0f;  // Original: 19.0f  (Slightly lower)
  float tr_x_percent = 73.0f; // Original: 86.0f
  float tr_y_percent = 5.0f;  // Original: 19.0f  (Slightly lower)
  float br_x_percent = 97.0f; // Original: 91.0f (Slightly inwards)
  float br_y_percent = 45.0f; // Original: 81.0f  (Slightly lower)
  float bl_x_percent = 5.0f;  // Original: 14.0f  (Slightly inwards)
  float bl_y_percent = 52.0f; // Original: 81.0f  (Slightly lower)
  int width = inputImage.cols;
  int height = inputImage.rows;
  vector<Point2f> board_corners = {
      Point2f(width * tl_x_percent / 100.0f, height * tl_y_percent / 100.0f),
      Point2f(width * tr_x_percent / 100.0f, height * tr_y_percent / 100.0f),
      Point2f(width * br_x_percent / 100.0f, height * br_y_percent / 100.0f),
      Point2f(width * bl_x_percent / 100.0f, height * bl_y_percent / 100.0f)};
  return board_corners;
}


vector<Point2f> getBoardCornersCorrected(const Mat& image){
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

  if (bDebug) {
    imshow("Blurred", blurred);
    waitKey(0);
  }

  adaptiveThreshold(blurred, edges, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
                    THRESH_BINARY, 11, 2);

  if (bDebug) {
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
  if (bDebug) {
    imshow("Canny", result);
  }
  return result;
}

// 2. Line Segment Detection Function (Refactored)
vector<Vec4i> detectLineSegments(const Mat &edges, bool bDebug) {
  int width = edges.cols;
  int height = edges.rows;

  // 1. Get Board Corners (You'll need to implement this correctly)
  // Use getBoardCornersCorrected to define the region of interest for HoughLinesP
  vector<Point2f> board_corners = getBoardCornersCorrected(edges);
  float board_height = board_corners[2].y - board_corners[0].y;
  float board_width = board_corners[1].x - board_corners[0].x;

  // Define a single mask for the entire board area
  // You might want to slightly expand this rectangle beyond the strict corners
  // to ensure lines right at the edge are detected.
  int margin = 10; // Pixels to expand the mask
   Rect board_rect(
      static_cast<int>(board_corners[0].x) - margin,
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
  int max_line_gap = 15;   // Increased significantly

  HoughLinesP(masked_edges, all_segments, 1, CV_PI / 180,
              hough_threshold, min_line_length, max_line_gap);

  if (bDebug) {
    cout << "Total detected line segments: " << all_segments.size() << endl;

    // Visualize Mask and Line Segments
    Mat mask_and_lines = edges.clone();
    Scalar mask_color = Scalar(128); // Gray
    int mask_thickness = 2;
    Scalar line_color = Scalar(0, 255, 255); // Yellow for all lines

    rectangle(mask_and_lines, board_rect, mask_color, mask_thickness); // Draw board mask

    // Draw all line segments
    cout << "\n----all line segments---\n";
    for (const auto &line : all_segments) {
      cv::line(mask_and_lines, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]),
           line_color, 1);
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

  // Angle tolerance for classifying lines as horizontal or vertical (in radians)
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
    // Segments that are neither horizontal nor vertical within tolerance are ignored
  }

  sort(horizontal_lines_raw.begin(), horizontal_lines_raw.end(), compareLines);
  sort(vertical_lines_raw.begin(), vertical_lines_raw.end(), compareLines);

  if (bDebug) {
    cout << "Raw horizontal lines count (after angle classification): " << horizontal_lines_raw.size()
         << endl;
    cout << "Raw vertical lines count (after angle classification): " << vertical_lines_raw.size()
         << endl;
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
  vector<pair<double, double>> matched_values; // Store (clustered_value, distance)

  // Custom comparison: prioritize matched_count (descending), then score (ascending)
  bool operator<(const LineMatch& other) const {
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

vector<double> findUniformGridLinesImproved(const vector<double>& values,
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
      current_matched_values.push_back({closest_value, min_diff}); // Store clustered_value!
      if (min_diff < tolerance) {
        current_fit_score += min_diff;
        matched_lines_count++;        
      }
    }
    match_data.insert({matched_lines_count, current_fit_score, start_value, current_matched_values});
  }

  // 2. Basis Line Selection
  if (match_data.empty()) {
    if (bDebug) {
      cout << "findUniformGridLinesImproved2: No matching data found.\n";
    }
    return uniform_lines; // Return empty
  }

  const vector<pair<double, double>>& best_matched_values = match_data.begin()->matched_values; // Get matched_values from best candidate

  // 3. Final Line Selection
  if (best_matched_values.size() > target_count) {
    vector<pair<double, double>> sorted_matched_values = best_matched_values;
    sort(sorted_matched_values.begin(), sorted_matched_values.end(),
         [](const pair<double, double>& a, const pair<double, double>& b) {
           return a.second < b.second; // Sort by distance
         });

    for (int i = 0; i < target_count; ++i) {
      uniform_lines.push_back(sorted_matched_values[i].first); // Store clustered_value
    }
  } else {
    for (const auto& pair : best_matched_values) {
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
                                      dominant_distance * tolerance,
                                      bDebug);
}

// Helper function to find the optimal clustering for a given set of lines
pair<vector<double>, double> findOptimalClusteringForOrientation(
  const vector<Line>& raw_lines,
  int target_count,
  const string& orientation, // "horizontal" or "vertical" for debugging output
  bool bDebug) {

  double cluster_threshold = 1.0; // Starting threshold - needs tuning
  double threshold_step = 0.5;   // Step to increase threshold - needs tuning
  int max_iterations = 20;       // Limit iterations - needs tuning

  vector<double> clustered_lines;
  vector<double> prev_clustered_lines;
  double optimal_threshold = cluster_threshold;

  // Initialize previous results with clustering at a very low threshold
  prev_clustered_lines = clusterAndAverageLines(raw_lines, 0.1, bDebug);

  if (bDebug) {
      cout << "Initial Clustering (" << orientation << ") with threshold 0.1: " << prev_clustered_lines.size() << " lines\n";
  }

  clustered_lines = prev_clustered_lines; // Initialize with the initial clustering

  for (int i = 0; i < max_iterations; ++i) {
      // Store current results
      prev_clustered_lines = clustered_lines;

      // Perform clustering with the current threshold
      clustered_lines = clusterAndAverageLines(raw_lines, cluster_threshold, bDebug);

      if (bDebug) {
          cout << "Clustering Attempt (" << orientation << ") " << i + 1 << " with threshold " << cluster_threshold << ": " << clustered_lines.size() << " lines\n";
      }

      if (clustered_lines.size() < target_count) {
          if (bDebug) {
              cout << "Clustered line count (" << orientation << ") dropped below target (" << target_count << "). Returning previous threshold's results.\n";
          }
          return make_pair(prev_clustered_lines, optimal_threshold); // Return previous (better) result
      }

      if (clustered_lines.size() == target_count) {
          if (bDebug) {
              cout << "Found target number of clustered lines (" << target_count << ") for " << orientation << ".\n";
          }
          return make_pair(clustered_lines, cluster_threshold);
      }

      optimal_threshold = cluster_threshold; // Update optimal threshold as we are still at or above target
      cluster_threshold += threshold_step;   // Increase threshold for the next attempt
  }

  if (bDebug) {
      cout << "Max iterations reached for " << orientation << " without finding target count or dropping below. Returning last iteration's results.\n";
  }
  return make_pair(clustered_lines, optimal_threshold);
}

// Main function to call the helper functions for horizontal and vertical lines
pair<vector<double>, vector<double>> findOptimalClustering(
  const vector<Line>& horizontal_lines_raw,
  const vector<Line>& vertical_lines_raw,
  int target_count,
  bool bDebug) {

  pair<vector<double>, double> horizontal_result =
      findOptimalClusteringForOrientation(horizontal_lines_raw, target_count, "horizontal", bDebug);

  pair<vector<double>, double> vertical_result =
      findOptimalClusteringForOrientation(vertical_lines_raw, target_count, "vertical", bDebug);

  return make_pair(horizontal_result.first, vertical_result.first);
}

// Refactored detectUniformGrid to use the new findOptimalClustering function
pair<vector<double>, vector<double>> detectUniformGrid(const Mat &image) {
  Mat processed_image = preprocessImage(image, bDebug);
  vector<Vec4i> mixed_segments =
      detectLineSegments(processed_image, bDebug);
  auto [horizontal_lines_raw, vertical_lines_raw] =
      convertSegmentsToLines(mixed_segments, bDebug);

  // Use the new function to find the optimal clustering to get potentially 19 lines
  // We pass 19 as the target count for clustering
  auto [clustered_horizontal_y, clustered_vertical_x] =
      findOptimalClustering(horizontal_lines_raw, vertical_lines_raw, 19, bDebug);

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
                  Int2Str(final_horizontal_y.size()).str() +
                  " horizontal and " + Int2Str(final_vertical_x.size()).str() +
                  " vertical lines after uniform grid finding.");
  }

  // The lines are already sorted by clusterAndAverageLines and verified by findUniformGridLines.
  // No need to sort again here.

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
  return sqrt(pow((color1[0] - color2[0]) * weight_h, 2) +
              pow((color1[1] - color2[1]) * weight_s, 2) +
              pow((color1[2] - color2[2]) * weight_v, 2));
}

// Function to calculate the original Euclidean distance between two HSV colors
float colorDistance(const Vec3f &color1, const Vec3f &color2) {
  return sqrt(pow(color1[0] - color2[0], 2) + pow(color1[1] - color2[1], 2) +
              pow(color1[2] - color2[2], 2));
}

// New function to classify clusters as Black, White, and Board
void classifyClusters(const Mat &centers, int &label_black, int &label_white,
                      int &label_board) {
  float min_v = numeric_limits<float>::max();
  float max_v = numeric_limits<float>::min();
  int index_min_v = -1;
  int index_max_v = -1;

  for (int i = 0; i < centers.rows;
       ++i) { // Use centers.rows for number of clusters
    float v = centers.at<float>(i, 2);
    if (v < min_v) {
      min_v = v;
      index_min_v = i;
    }
    if (v > max_v) {
      max_v = v;
      index_max_v = i;
    }
  }

  label_black = index_min_v;
  label_white = index_max_v;

  for (int i = 0; i < centers.rows; ++i) { // Iterate through all clusters
    if (i != label_black && i != label_white) {
      label_board = i;
      break; // No need to continue once board is found
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

// New function for testing perspective transform

// Function to process the Go board image and determine the board state
void processGoBoard(const Mat &image_bgr_in, Mat &board_state,
                    Mat &board_with_stones,
                    vector<Point2f> &intersection_points) {
  Mat image_bgr = correctPerspective(image_bgr_in);
  // imshow("image int", image_bgr_in);
  //       waitKey(0);
  // imshow("image correct perspective", image_bgr);
  //       waitKey(0);
  Mat image_hsv;
  cvtColor(image_bgr, image_hsv, COLOR_BGR2HSV);

  pair<vector<double>, vector<double>> grid_lines =
      detectUniformGrid(image_bgr);
  vector<double> horizontal_lines = grid_lines.first;
  vector<double> vertical_lines = grid_lines.second;
  if (bDebug) {
    Mat debug_lines = image_bgr.clone(); // Create a copy for drawing lines
    // Draw Horizontal Lines
    for (double y : horizontal_lines) {
      line(debug_lines, Point(0, y), Point(image_bgr.cols - 1, y),
           Scalar(0, 0, 255), 2); // Red lines
    }

    // Draw Vertical Lines
    for (double x : vertical_lines) {
      line(debug_lines, Point(x, 0), Point(x, image_bgr.rows - 1),
           Scalar(0, 255, 0), 2); // Green lines
    }
    imshow("Detected Grid Lines", debug_lines);
    waitKey(0);
  }
  intersection_points = findIntersections(horizontal_lines, vertical_lines);
  int num_intersections = intersection_points.size();
  int sample_radius = 8;

  Mat samples(num_intersections, 3, CV_32F);
  vector<Vec3f> average_hsv_values(num_intersections);
  for (int i = 0; i < num_intersections; ++i) {
    Vec3f avg_hsv =
        getAverageHSV(image_hsv, intersection_points[i], sample_radius);
    samples.at<float>(i, 0) = avg_hsv[0];
    samples.at<float>(i, 1) = avg_hsv[1];
    samples.at<float>(i, 2) = avg_hsv[2];
    average_hsv_values[i] = avg_hsv;
  }

  int num_clusters = 3;
  Mat labels;
  Mat centers;
  kmeans(samples, num_clusters, labels,
         TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 100, 1.0), 3,
         KMEANS_PP_CENTERS, centers);
  if (bDebug) {
    cout << "\n--- K-Means Cluster Centers (HSV) ---\n" << centers << endl;
    cout << "\n--- Raw K-Means Labels (first 20) ---\n";
    for (int i = 0; i < min(20, num_intersections); ++i) {
      cout << labels.at<int>(i, 0) << " ";
    }
    cout << "...\n";
  }
  int label_black = -1, label_white = -1, label_board = -1;
  classifyClusters(centers, label_black, label_white, label_board);
  if (bDebug) {
    cout << "\n--- Assigned Labels (Direct Value Based) ---\n";
    cout << "Black Cluster ID: " << label_black << endl;
    cout << "White Cluster ID: " << label_white << endl;
    cout << "Board Cluster ID: " << label_board << endl;
  }

  board_state = Mat(19, 19, CV_8U, Scalar(0));
  board_with_stones = image_bgr.clone();
  if (bDebug) {
    cout << "\n--- Intersection HSV and Assigned Cluster (Weighted Distance)"
         << "-- -" << endl;
  }

  cout << fixed << setprecision(2);

  // float weight_h = 0.10f;
  // float weight_s = 0.45f;
  // float weight_v = 0.45f;

  float weight_h = 0.60f; // Example: Increased Hue weight
  float weight_s = 0.20f; // Example: Decreased Saturation weight
  float weight_v = 0.20f; // Example: Decreased Value weight

  for (int i = 0; i < num_intersections; ++i) {
    int row = i / 19;
    int col = i % 19;
    Vec3f hsv = average_hsv_values[i];

    float min_distance = numeric_limits<float>::max();
    int closest_cluster = -1;
    for (int j = 0; j < num_clusters; ++j) {
      Vec3f cluster_center(centers.at<float>(j, 0), centers.at<float>(j, 1),
                           centers.at<float>(j, 2));
      float distance = colorDistanceWeighted(hsv, cluster_center, weight_h,
                                             weight_s, weight_v);
      if (distance < min_distance) {
        min_distance = distance;
        closest_cluster = j;
      }
    }
    if (bDebug && false) {
      cout << "[" << row << "," << col << "] HSV: [" << hsv[0] << ", " << hsv[1]
           << ", " << hsv[2] << "] Cluster (Weighted): " << closest_cluster
           << std::endl;
    }

    if (closest_cluster == label_black) {
      board_state.at<uchar>(row, col) = 1; // Black
      circle(board_with_stones, intersection_points[i], 8, Scalar(0, 0, 0), -1);
      // cout << " (Black)" << endl;
    } else if (closest_cluster == label_white) {
      board_state.at<uchar>(row, col) = 2; // White
      circle(board_with_stones, intersection_points[i], 8,
             Scalar(255, 255, 255), -1);
      // cout << " (White)" << endl;
    } else if (closest_cluster == label_board) {
      board_state.at<uchar>(row, col) = 0; // Empty
      circle(board_with_stones, intersection_points[i], 8, Scalar(0, 255, 0),
             2);
      // cout << " (Board)" << endl;
    } else {
      circle(board_with_stones, intersection_points[i], 8, Scalar(255, 0, 255),
             2); // Magenta for unclassified
                 // cout << " (Unclassified - Error?)" << endl;
    }
  }
  if (bDebug) {
    imshow("processGoBoard", board_with_stones);
    waitKey(0);
  }
  if (bDebug) {
    Mat debug_lines = image_bgr.clone();
    for (double y : vertical_lines) {
      line(debug_lines, Point(0, y), Point(debug_lines.cols - 1, y),
           Scalar(255, 0, 0), 2); // Red for horizontal
    }
    for (double x : horizontal_lines) {
      line(debug_lines, Point(x, 0), Point(x, debug_lines.rows - 1),
           Scalar(0, 0, 255), 2); // Blue for vertical
    }
    imshow("Detected Grid Lines", debug_lines);

    Mat debug_intersections = image_bgr.clone();
    for (const auto &p : intersection_points) {
      circle(debug_intersections, p, 10, Scalar(0, 255, 0),
             2); // Green for detected intersections
    }
    imshow("Detected Intersections (Raw)", debug_intersections);
  }
}
