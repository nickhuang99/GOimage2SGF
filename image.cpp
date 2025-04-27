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


// Helper function to order corners (top-left, top-right, bottom-right, bottom-left)
void orderCorners(vector<Point2f>& corners) {
  // Calculate centroid
  Point2f centroid(0, 0);
  for (const auto& p : corners) {
    centroid.x += p.x;
    centroid.y += p.y;
  }
  centroid.x /= corners.size();
  centroid.y /= corners.size();

  std::sort(corners.begin(), corners.end(),
            [centroid](const Point2f& a, const Point2f& b) {
              // Sort by angle from centroid
              return atan2(a.y - centroid.y, a.x - centroid.x) <
                     atan2(b.y - centroid.y, b.x - centroid.x);
            });
}

// Function to find the corners of the Go board
vector<Point2f> getBoardCorners(const Mat &inputImage) {

  int TOPWIDTH = 20, TOPHEIGHT = 5, BOTTOMWIDTH = 5, BOTTOMHEIGHT = 50;
  vector<Point2f> board_corners = {
      Point2f(inputImage.cols * TOPWIDTH / 100,
              inputImage.rows * TOPHEIGHT / 100),
      Point2f(inputImage.cols * (100 - TOPWIDTH) / 100,
              inputImage.rows * TOPHEIGHT / 100),
      Point2f(inputImage.cols * (100 - BOTTOMWIDTH) / 100,
              inputImage.rows * (100 - BOTTOMHEIGHT) / 100),
      Point2f(inputImage.cols * BOTTOMHEIGHT / 100,
              inputImage.rows * (100 - BOTTOMWIDTH) / 100)};
  return board_corners;
}

// Function to correct perspective using hardcoded percentage-based corners
Mat correctPerspectiveHardcoded(const Mat& inputImage, int dest_percent) {
  int width = inputImage.cols;
  int height = inputImage.rows;

  // Hardcoded corner percentages (You MUST fine-tune these)
  float tl_x_percent = 20.0f; // Top-left x
  float tl_y_percent = 20.0f; // Top-left y
  float tr_x_percent = 85.0f; // Top-right x
  float tr_y_percent = 20.0f; // Top-right y
  float br_x_percent = 90.0f; // Bottom-right x
  float br_y_percent = 80.0f; // Bottom-right y
  float bl_x_percent = 15.0f; // Bottom-left x
  float bl_y_percent = 80.0f; // Bottom-left y

  // Calculate corner coordinates
  vector<Point2f> input_corners = {
      Point2f(width * tl_x_percent / 100.0f, height * tl_y_percent / 100.0f),
      Point2f(width * tr_x_percent / 100.0f, height * tr_y_percent / 100.0f),
      Point2f(width * br_x_percent / 100.0f, height * br_y_percent / 100.0f),
      Point2f(width * bl_x_percent / 100.0f, height * bl_y_percent / 100.0f)
  };

  vector<Point2f> output_corners = {
      Point2f(width * dest_percent / 100.0f, height * dest_percent / 100.0f),
      Point2f(width * (100 - dest_percent) / 100.0f, height * dest_percent / 100.0f),
      Point2f(width * (100 - dest_percent) / 100.0f, height * (100 - dest_percent) / 100.0f),
      Point2f(width * dest_percent / 100.0f, height * (100 - dest_percent) / 100.0f)
  };


  Mat perspective_matrix = getPerspectiveTransform(input_corners, output_corners);
  Mat outputImage;
  warpPerspective(inputImage, outputImage, perspective_matrix, Size(width, height));

  return outputImage;
}

Mat correctPerspective(const Mat &image) {
  float tl_x_percent = 20.0f; // Original: 19.0f
  float tl_y_percent = 6.0f;  // Original: 19.0f  (Slightly lower)
  float tr_x_percent = 73.0f; // Original: 86.0f
  float tr_y_percent = 5.0f;  // Original: 19.0f  (Slightly lower)
  float br_x_percent = 97.0f; // Original: 91.0f (Slightly inwards)
  float br_y_percent = 45.0f; // Original: 81.0f  (Slightly lower)
  float bl_x_percent = 5.0f;  // Original: 14.0f  (Slightly inwards)
  float bl_y_percent = 52.0f; // Original: 81.0f  (Slightly lower)

  int width = image.cols;
  int height = image.rows;

  vector<Point2f> input_corners = {
      Point2f(width * tl_x_percent / 100.0f, height * tl_y_percent / 100.0f),
      Point2f(width * tr_x_percent / 100.0f, height * tr_y_percent / 100.0f),
      Point2f(width * br_x_percent / 100.0f, height * br_y_percent / 100.0f),
      Point2f(width * bl_x_percent / 100.0f, height * bl_y_percent / 100.0f)};

  int dest_percent = 15; // Start with 15 and adjust if needed
  vector<Point2f> output_corners = {
      Point2f(width * dest_percent / 100.0f, height * dest_percent / 100.0f),
      Point2f(width * (100 - dest_percent) / 100.0f,
              height * dest_percent / 100.0f),
      Point2f(width * (100 - dest_percent) / 100.0f,
              height * (100 - dest_percent) / 100.0f),
      Point2f(width * dest_percent / 100.0f,
              height * (100 - dest_percent) / 100.0f)};

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

pair<vector<double>, vector<double>> detectUniformGrid(const Mat &image) {
  // Mat gray, blurred, edges;
  // cvtColor(image, gray, COLOR_BGR2GRAY);
  // GaussianBlur(gray, blurred, Size(5, 5), 0);
  
  // //Canny(blurred, edges, 50, 150);
  // /*Experiment with ADAPTIVE_THRESH_MEAN_C vs. ADAPTIVE_THRESH_GAUSSIAN_C, 
  // and carefully tune the blockSize (e.g., 11, 15, 21 - must be odd) and C
  // (a constant subtracted from the mean/weighted sum) parameters.*/  
  // adaptiveThreshold(gray, edges, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);

  // Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3)); // Adjust kernel size
  // morphologyEx(edges, edges, MORPH_CLOSE, kernel);

  Mat gray, blurred, edges, morph; // Add morph

  cvtColor(image, gray, COLOR_BGR2GRAY);

  // Stronger Blur (Experiment - But Don't Over-Blur)
  GaussianBlur(gray, blurred, Size(5, 5), 0); // Or Size(7, 7)

  // Debug: Show blurred
  if (bDebug) {
    imshow("Blurred", blurred);
    waitKey(0);
  }

  // Adaptive Thresholding (CRITICAL - TUNE CAREFULLY)
  adaptiveThreshold(blurred, edges, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);

  // Debug: Show edges (before morphology)
  if (bDebug) {
    imshow("Edges (Before Morph)", edges);
    waitKey(0);
  }

  // Morphological Operations (NUANCED - TUNE VERY CAREFULLY)
  Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3)); // Smaller kernel
  morphologyEx(edges, morph, MORPH_CLOSE, kernel, Point(-1,-1), 1); // 1 iteration

  // Debug: Show morphological output
  if (bDebug) {
    imshow("Morph", morph);
    waitKey(0);
  }

  edges = morph.clone(); // Use the morphological output for HoughLinesP


//  // HoughLinesP Parameters (TUNE THESE CAREFULLY)
//   int threshold = 10;       // Start low, adjust as needed
//   int minLineLength = 20;   // Moderate, adjust as needed
//   int maxLineGap = 15;      // Start higher, adjust as needed

//   vector<Vec4i> line_segments;
//   HoughLinesP(edges, line_segments, 1, CV_PI / 180, threshold, minLineLength, maxLineGap);

//   if (bDebug) {
//     cout << "Number of line segments detected: " << line_segments.size() << endl;
//   }
int width = edges.cols;
int height = edges.rows;

// Masks for horizontal and vertical line detection
Mat horizontal_mask = Mat::zeros(height, width, CV_8U);
horizontal_mask(Rect(0, height / 2 - 5, width, 10)) = 255; // Horizontal strip

Mat vertical_mask = Mat::zeros(height, width, CV_8U);
vertical_mask(Rect(width / 2 - 5, 0, 10, height)) = 255;   // Vertical strip

Mat masked_edges;
vector<Vec4i> horizontal_lines_segments, vertical_lines_segments;

// Detect Horizontal Lines
bitwise_and(edges, horizontal_mask, masked_edges);
HoughLinesP(masked_edges, horizontal_lines_segments, 1, CV_PI / 180, 10, 20, 5); // TUNE
if (bDebug) {
  cout << "Horizontal line segments: " << horizontal_lines_segments.size() << endl;
}

// Detect Vertical Lines
bitwise_and(edges, vertical_mask, masked_edges);
HoughLinesP(masked_edges, vertical_lines_segments, 1, CV_PI / 180, 10, 20, 5); // TUNE
if (bDebug) {
  cout << "Vertical line segments: " << vertical_lines_segments.size() << endl;
}

  vector<Line> horizontal_lines_raw, vertical_lines_raw;

  auto process_segment_labda = [&horizontal_lines_raw,
                                &vertical_lines_raw](const vector<Vec4i> &in) {
    for (const auto &segment : in) {
      Point pt1(segment[0], segment[1]);
      Point pt2(segment[2], segment[3]);
      double angle = atan2(pt2.y - pt1.y, pt2.x - pt1.x);
      double center_y = (pt1.y + pt2.y) / 2.0;
      double center_x = (pt1.x + pt2.x) / 2.0;

      if (abs(angle) < CV_PI / 18 || abs(abs(angle) - CV_PI) < CV_PI / 18) {
        horizontal_lines_raw.push_back({center_y, angle});
      } else if (abs(abs(angle) - CV_PI / 2) < CV_PI / 18) {
        vertical_lines_raw.push_back({center_x, angle});
      }
    }
  };
  process_segment_labda(horizontal_lines_segments);
  process_segment_labda(vertical_lines_segments);
  // for (const auto &segment : line_segments) {
  //   Point pt1(segment[0], segment[1]);
  //   Point pt2(segment[2], segment[3]);
  //   double angle = atan2(pt2.y - pt1.y, pt2.x - pt1.x);
  //   double center_y = (pt1.y + pt2.y) / 2.0;
  //   double center_x = (pt1.x + pt2.x) / 2.0;

  //   if (abs(angle) < CV_PI / 18 || abs(abs(angle) - CV_PI) < CV_PI / 18) {
  //     horizontal_lines_raw.push_back({center_y, angle});
  //   } else if (abs(abs(angle) - CV_PI / 2) < CV_PI / 18) {
  //     vertical_lines_raw.push_back({center_x, angle});
  //   }
  // }



  sort(horizontal_lines_raw.begin(), horizontal_lines_raw.end(), compareLines);
  sort(vertical_lines_raw.begin(), vertical_lines_raw.end(), compareLines);
  if (bDebug) {
    cout << "Raw horizontal lines count: " << horizontal_lines_raw.size()
         << endl;
    cout << "Raw vertical lines count: " << vertical_lines_raw.size() << endl;
  }

  auto cluster_and_average_lines = [](const vector<Line> &raw_lines,
                                      double threshold) {
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
  };

  double cluster_threshold = 5.0;
  vector<double> clustered_horizontal_y =
      cluster_and_average_lines(horizontal_lines_raw, cluster_threshold);
  vector<double> clustered_vertical_x =
      cluster_and_average_lines(vertical_lines_raw, cluster_threshold);
  if (bDebug) {
    cout << "Clustered horizontal lines count: "
         << clustered_horizontal_y.size() << endl;
    cout << "Clustered vertical lines count: " << clustered_vertical_x.size()
         << endl;
    cout << "Clustered horizontal lines (y): ";
    for (double y : clustered_horizontal_y)
      cout << y << " ";
    cout << endl;
    cout << "Clustered vertical lines (x): ";
    for (double x : clustered_vertical_x)
      cout << x << " ";
    cout << endl;
  }

  int imageHeight = image.rows;
  int imageWidth = image.cols;

  auto find_uniform_grid_lines = [](vector<double> values, int target_count,
                                    double tolerance) {
    if (values.size() < target_count / 2) {
      return vector<double>{}; // Return empty if too few lines
    }
    sort(values.begin(), values.end());

    if (bDebug && !values.empty()) {
      cout << "Sorted clustered values of size: {" << values.size() << "}:\n";
      for (size_t i = 0; i < values.size() - 1; ++i) {
        cout << "value[" << i << "]: " << values[i]
             << " distance: " << values[i + 1] - values[i] << endl;
      }
      cout << "value: " << values[values.size() - 1] << endl;
    }

    if (values.size() < 2) {
      return values;
    }

    vector<double> distances;
    for (size_t i = 0; i < values.size() - 1; ++i) {
      distances.push_back(values[i + 1] - values[i]);
    }

    vector<double> sorted_distances = distances;
    sort(sorted_distances.begin(), sorted_distances.end());

    double average_distance = 0;
    if (!sorted_distances.empty()) {
      size_t i = 0;
      size_t j = sorted_distances.size() - 1;
      while (j - i > target_count / 2 && i < j &&
             abs(sorted_distances[i] - sorted_distances[j]) /
                     sorted_distances[i] >
                 tolerance) {
        j--;
        i++;
      }

      if (i <= j) {
        double sum_middle_distances = 0;
        for (size_t k = i; k <= j; ++k) {
          sum_middle_distances += sorted_distances[k];
        }
        average_distance = sum_middle_distances / (j - i + 1);
      }
    }

    if (average_distance <= 0) {
      if (bDebug) {
        cout << "average_distance is negative:" << average_distance << endl;
      }
      return values; // Fallback
    }

    int best_continuous_count = 0;
    int best_start_index = -1;

    for (size_t i = 0; i < distances.size(); ++i) {
      int current_continuous_count = 0;
      for (size_t j = i; j < distances.size(); ++j) {
        if (abs(distances[j] - average_distance) / average_distance <=
            tolerance) {
          current_continuous_count++;
        } else {
          break;
        }
      }
      if (current_continuous_count >= target_count / 2.0 &&
          current_continuous_count > best_continuous_count) {
        best_continuous_count = current_continuous_count;
        best_start_index = i;
      }
    }
    if (bDebug) {
      cout << "best_start_index: " << best_start_index << endl
           << "best_continuous_count: " << best_continuous_count << endl;
    }
    if (best_start_index == -1) {
      return values; // Could not find a good continuous group with average
                     // distance
    }

    vector<double> uniform_lines;
    double lowest_val = values[best_start_index];
    double highest_val = values[best_start_index + best_continuous_count - 1];
    double lo_boundary = values.front();
    double hi_boundary = values.back();
    int expand_needed = target_count - best_continuous_count;

    for (int i = 0; i < best_continuous_count; ++i) {
      uniform_lines.push_back(values[best_start_index + i]);
    }
    sort(uniform_lines.begin(), uniform_lines.end());
    int i = 0;
    while (i < expand_needed) {
      if (uniform_lines.front() - average_distance >=
          lo_boundary - tolerance * average_distance) {
        uniform_lines.insert(uniform_lines.begin(),
                             uniform_lines.front() - average_distance);
        i++;
        if (i < expand_needed)
          break;
      }
      if (uniform_lines.back() + average_distance <=
          hi_boundary + tolerance * average_distance) {
        uniform_lines.push_back(uniform_lines.back() + average_distance);
        i++;
        if (i < expand_needed)
          break;
      }
    }
    sort(uniform_lines.begin(), uniform_lines.end());
    if (bDebug) {
      cout << "uniform_lines:" << uniform_lines.size() << endl;
    }
    if (uniform_lines.size() > target_count) {
      size_t start = (uniform_lines.size() - target_count) / 2;
      uniform_lines.assign(uniform_lines.begin() + start,
                           uniform_lines.begin() + start + target_count);
    } else if (uniform_lines.size() < target_count && !values.empty()) {
      if (bDebug) {
        cout << "uniform_lines is less than target: " << uniform_lines.size()
             << endl;
      }
      return values; // Fallback
    }

    return uniform_lines;
  };
  double spacing_tolerance = 0.4;
  vector<double> final_horizontal_y =
      find_uniform_grid_lines(clustered_horizontal_y, 19, spacing_tolerance);
  vector<double> final_vertical_x =
      find_uniform_grid_lines(clustered_vertical_x, 19, spacing_tolerance);
  if (final_vertical_x.size() == 19) {
    THROWGEMERROR(
        std::string("find_uniform_grid_lines find final_vertical_x ") +
        Int2Str(final_vertical_x.size()).str());
  }
  if (final_horizontal_y.size() == 19) {
    THROWGEMERROR(
        std::string("find_uniform_grid_lines find final_horizontal_y ") +
        Int2Str(final_horizontal_y.size()).str());
  }  
  sort(final_horizontal_y.begin(), final_horizontal_y.end());
  sort(final_vertical_x.begin(), final_vertical_x.end());
  if (bDebug) {
    cout << "Final sorted horizontal lines (y): ";
    for (double y : final_horizontal_y)
      cout << y << " ";
    cout << endl;
    cout << "Final sorted vertical lines (x): ";
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

  float weight_h = 0.10f;
  float weight_s = 0.45f;
  float weight_v = 0.45f;

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
    if (bDebug) {
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
