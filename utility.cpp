#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace cv;
using namespace std;

struct Line {
    double value; // y for horizontal, x for vertical
    double angle;
};

bool compareLines(const Line& a, const Line& b) {
    return a.value < b.value;
}

pair<vector<double>, vector<double>> detectUniformGrid(const Mat& image) {
    Mat gray, blurred, edges;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurred, Size(5, 5), 0);
    Canny(blurred, edges, 50, 150);

    vector<Vec4i> line_segments;
    HoughLinesP(edges, line_segments, 1, CV_PI / 180, 50, 30, 10);

    vector<Line> horizontal_lines_raw, vertical_lines_raw;

    for (const auto& segment : line_segments) {
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

    sort(horizontal_lines_raw.begin(), horizontal_lines_raw.end(), compareLines);
    sort(vertical_lines_raw.begin(), vertical_lines_raw.end(), compareLines);

    auto cluster_and_average_lines = [](const vector<Line>& raw_lines, double threshold) {
        vector<double> clustered_values;
        if (raw_lines.empty()) return clustered_values;

        vector<bool> processed(raw_lines.size(), false);
        for (size_t i = 0; i < raw_lines.size(); ++i) {
            if (processed[i]) continue;
            vector<double> current_cluster;
            current_cluster.push_back(raw_lines[i].value);
            processed[i] = true;
            for (size_t j = i + 1; j < raw_lines.size(); ++j) {
                if (!processed[j] && abs(raw_lines[j].value - raw_lines[i].value) < threshold) {
                    current_cluster.push_back(raw_lines[j].value);
                    processed[j] = true;
                }
            }
            if (!current_cluster.empty()) {
                clustered_values.push_back(accumulate(current_cluster.begin(), current_cluster.end(), 0.0) / current_cluster.size());
            }
        }
        sort(clustered_values.begin(), clustered_values.end());
        return clustered_values;
    };

    double cluster_threshold = 15.0;
    vector<double> clustered_horizontal_y = cluster_and_average_lines(horizontal_lines_raw, cluster_threshold);
    vector<double> clustered_vertical_x = cluster_and_average_lines(vertical_lines_raw, cluster_threshold);

    int imageHeight = image.rows;

    auto find_uniform_spacing = [imageHeight](vector<double> values, int target_count, double tolerance) {
        if (target_count != 19 || values.size() < 5) return values; // Need enough lines to estimate spacing

        sort(values.begin(), values.end());
        int center_start = values.size() / 3;
        int center_end = 2 * values.size() / 3;
        vector<double> central_lines;
        for (int i = center_start; i < center_end; ++i) {
            central_lines.push_back(values[i]);
        }
        if (central_lines.size() < 2) return values;

        double total_spacing = 0;
        for (size_t i = 1; i < central_lines.size(); ++i) {
            total_spacing += central_lines[i] - central_lines[i - 1];
        }
        double estimated_spacing = total_spacing / (central_lines.size() - 1);

        vector<double> extrapolated_lines;
        double middle_line = central_lines[central_lines.size() / 2];
        int middle_index = 9; // For 19 lines, the middle is at index 9

        for (int i = 0; i < target_count; ++i) {
            extrapolated_lines.push_back(middle_line + (i - middle_index) * estimated_spacing);
        }
        sort(extrapolated_lines.begin(), extrapolated_lines.end());

        vector<double> final_lines;
        vector<bool> used(values.size(), false);
        for (double extrapolated_y : extrapolated_lines) {
            double min_diff = 1e9;
            int best_index = -1;
            for (size_t i = 0; i < values.size(); ++i) {
                if (!used[i]) {
                    double diff = abs(values[i] - extrapolated_y);
                    if (diff < min_diff) {
                        min_diff = diff;
                        best_index = i;
                    }
                }
            }
            if (best_index != -1) {
                final_lines.push_back(values[best_index]);
                used[best_index] = true;
            }
        }
        sort(final_lines.begin(), final_lines.end());
        return final_lines;
    };

    auto find_uniform_spacing_vertical = [](vector<double> values, int target_count, double tolerance) {
        vector<double> best_group;
        double min_deviation = 1e9;

        sort(values.begin(), values.end());

        for (size_t i = 0; i <= values.size() - target_count; ++i) {
            vector<double> current_group;
            for (int k = 0; k < target_count; ++k) {
                current_group.push_back(values[i + k]);
            }

            if (current_group.size() < 2) continue;

            double initial_spacing = current_group[1] - current_group[0];
            double max_deviation = 0;
            for (size_t j = 2; j < current_group.size(); ++j) {
                max_deviation = max(max_deviation, abs((current_group[j] - current_group[j - 1]) - initial_spacing));
            }

            if (max_deviation <= tolerance * initial_spacing) {
                if (current_group.size() == target_count && max_deviation < min_deviation) {
                    min_deviation = max_deviation;
                    best_group = current_group;
                } else if (best_group.empty() && current_group.size() >= target_count / 2 && max_deviation <= 2 * tolerance * initial_spacing) {
                    best_group = current_group;
                }
            }
        }
        if (!best_group.empty()) {
            sort(best_group.begin(), best_group.end());
            return best_group;
        }
        return values; // Fallback
    };


    double spacing_tolerance = 0.4;
    vector<double> final_horizontal_y = find_uniform_spacing(clustered_horizontal_y, 19, spacing_tolerance);
    vector<double> final_vertical_x = find_uniform_spacing_vertical(clustered_vertical_x, 19, spacing_tolerance); // Use a separate function for vertical

    sort(final_horizontal_y.begin(), final_horizontal_y.end());
    sort(final_vertical_x.begin(), final_vertical_x.end());

    return make_pair(final_horizontal_y, final_vertical_x);
}


// Function to find intersection points of two sets of lines
vector<Point2f> findIntersections(const vector<double>& horizontal_lines, const vector<double>& vertical_lines) {
    vector<Point2f> intersections;
    for (double y : horizontal_lines) {
        for (double x : vertical_lines) {
            intersections.push_back(Point2f(x, y));
        }
    }
    return intersections;
}


// Function to calculate the weighted Euclidean distance between two HSV colors
float colorDistanceWeighted(const Vec3f& color1, const Vec3f& color2, float weight_h, float weight_s, float weight_v) {
    return sqrt(pow((color1[0] - color2[0]) * weight_h, 2) +
                pow((color1[1] - color2[1]) * weight_s, 2) +
                pow((color1[2] - color2[2]) * weight_v, 2));
}

// Function to calculate the original Euclidean distance between two HSV colors
float colorDistance(const Vec3f& color1, const Vec3f& color2) {
    return sqrt(pow(color1[0] - color2[0], 2) +
                pow(color1[1] - color2[1], 2) +
                pow(color1[2] - color2[2], 2));
}

// New function to classify clusters as Black, White, and Board
void classifyClusters(const Mat& centers, int& label_black, int& label_white, int& label_board) {
    float min_v = numeric_limits<float>::max();
    float max_v = numeric_limits<float>::min();
    int index_min_v = -1;
    int index_max_v = -1;

    for (int i = 0; i < centers.rows; ++i) { // Use centers.rows for number of clusters
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
Vec3f getAverageHSV(const Mat& image, Point2f center, int radius) {
    Vec3f sum(0, 0, 0);
    int count = 0;
    for (int y = center.y - radius; y <= center.y + radius; ++y) {
        for (int x = center.x - radius; x <= center.x + radius; ++x) {
            if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
                Vec3b bgr_color = image.at<Vec3b>(y, x);
                Mat bgr_pixel(1, 1, CV_8UC3, bgr_color); // Create a 1x1 Mat from the pixel
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

// Function to process the Go board image and determine the board state
void processGoBoard(const Mat& image_bgr, Mat& board_state, Mat& board_with_stones) {
    Mat image_hsv;
    cvtColor(image_bgr, image_hsv, COLOR_BGR2HSV);

    pair<vector<double>, vector<double>> grid_lines = detectUniformGrid(image_bgr);
    vector<double> horizontal_lines = grid_lines.first;
    vector<double> vertical_lines = grid_lines.second;

    vector<Point2f> intersection_points = findIntersections(horizontal_lines, vertical_lines);
    int num_intersections = intersection_points.size();
    int sample_radius = 8;

    Mat samples(num_intersections, 3, CV_32F);
    vector<Vec3f> average_hsv_values(num_intersections);
    for (int i = 0; i < num_intersections; ++i) {
        Vec3f avg_hsv = getAverageHSV(image_hsv, intersection_points[i], sample_radius);
        samples.at<float>(i, 0) = avg_hsv[0];
        samples.at<float>(i, 1) = avg_hsv[1];
        samples.at<float>(i, 2) = avg_hsv[2];
        average_hsv_values[i] = avg_hsv;
    }

    int num_clusters = 3;
    Mat labels;
    Mat centers;
    kmeans(samples, num_clusters, labels, TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 100, 1.0), 3, KMEANS_PP_CENTERS, centers);

    //cout << "\n--- K-Means Cluster Centers (HSV) ---\n" << centers << endl;
    //cout << "\n--- Raw K-Means Labels (first 20) ---\n";
    //for (int i = 0; i < min(20, num_intersections); ++i) {
    //    cout << labels.at<int>(i, 0) << " ";
    //}
    //cout << "...\n";

    int label_black = -1, label_white = -1, label_board = -1;
    classifyClusters(centers, label_black, label_white, label_board);

    //cout << "\n--- Assigned Labels (Direct Value Based) ---\n";
    //cout << "Black Cluster ID: " << label_black << endl;
    //cout << "White Cluster ID: " << label_white << endl;
    //cout << "Board Cluster ID: " << label_board << endl;

    board_state = Mat(19, 19, CV_8U, Scalar(0));
    board_with_stones = image_bgr.clone();
    //cout << "\n--- Intersection HSV and Assigned Cluster (Weighted Distance) ---" << endl;
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
            Vec3f cluster_center(centers.at<float>(j, 0), centers.at<float>(j, 1), centers.at<float>(j, 2));
            float distance = colorDistanceWeighted(hsv, cluster_center, weight_h, weight_s, weight_v);
            if (distance < min_distance) {
                min_distance = distance;
                closest_cluster = j;
            }
        }

        //cout << "[" << row << "," << col << "] HSV: [" << hsv[0] << ", " << hsv[1] << ", " << hsv[2] << "] Cluster (Weighted): " << closest_cluster;

        if (closest_cluster == label_black) {
            board_state.at<uchar>(row, col) = 1; // Black
            circle(board_with_stones, intersection_points[i], 8, Scalar(0, 0, 0), -1);
            //cout << " (Black)" << endl;
        } else if (closest_cluster == label_white) {
            board_state.at<uchar>(row, col) = 2; // White
            circle(board_with_stones, intersection_points[i], 8, Scalar(255, 255, 255), -1);
            //cout << " (White)" << endl;
        } else if (closest_cluster == label_board) {
            board_state.at<uchar>(row, col) = 0; // Empty
            circle(board_with_stones, intersection_points[i], 8, Scalar(0, 255, 0), 2);
            //cout << " (Board)" << endl;
        } else {
            circle(board_with_stones, intersection_points[i], 8, Scalar(255, 0, 255), 2); // Magenta for unclassified
            //cout << " (Unclassified - Error?)" << endl;
        }
    }
}


// Function to determine the SGF move between two board states
string determineSGFMove(const Mat& before_board_state, const Mat& next_board_state) {
    // 1. Find the differences between the two board states.
    vector<Point> black_diff_add;
    vector<Point> white_diff_add;
    vector<Point> black_diff_remove;
    vector<Point> white_diff_remove;

    for (int row = 0; row < 19; ++row) {
        for (int col = 0; col < 19; ++col) {
            int before_stone = before_board_state.at<uchar>(row, col);
            int next_stone = next_board_state.at<uchar>(row, col);

            if (before_stone != next_stone) {
                if (next_stone == 1) { // Black stone added
                    black_diff_add.push_back(Point(col, row)); // Store as (x, y)
                } else if (next_stone == 2) { // White stone added
                    white_diff_add.push_back(Point(col, row));
                } else if (before_stone == 1) { // Black stone removed
                    black_diff_remove.push_back(Point(col, row));
                } else if (before_stone == 2) { // White stone removed
                    white_diff_remove.push_back(Point(col, row));
                }
            }
        }
    }

    // 2. Analyze the differences to determine the move.
    string sgf_move = "";
    if (black_diff_add.size() == 1 && white_diff_add.size() == 0) {
        // Black played a stone.
        char sgf_col = 'a' + black_diff_add[0].x;
        char sgf_row = 'a' + black_diff_add[0].y;
        sgf_move = ";B[" + string(1, sgf_col) + string(1, sgf_row) + "]";
        //  Captures are optional
        for (Point p : white_diff_remove) {
            char sgf_col_remove = 'a' + p.x;
            char sgf_row_remove = 'a' + p.y;
            sgf_move += "AE[" + string(1, sgf_col_remove) + string(1, sgf_row_remove) + "]";
        }

    } else if (white_diff_add.size() == 1 && black_diff_add.size() == 0) {
        // White played a stone.
        char sgf_col = 'a' + white_diff_add[0].x;
        char sgf_row = 'a' + white_diff_add[0].y;
        sgf_move = ";W[" + string(1, sgf_col) + string(1, sgf_row) + "]";
        // Captures are optional
        for (Point p : black_diff_remove) {
            char sgf_col_remove = 'a' + p.x;
            char sgf_row_remove = 'a' + p.y;
            sgf_move += "AE[" + string(1, sgf_col_remove) + string(1, sgf_row_remove) + "]";
        }
    } else {
        sgf_move = ";ERROR: Invalid move detected!"; // Handle as an error
    }

    return sgf_move;
}

// Function to generate SGF for the current board state
string generateSGF(const Mat& board_state,  const vector<Point2f>& intersections) {
    ostringstream sgf;
    sgf << "(;FF[4]GM[1]SZ[19]AP[GoBoardAnalyzer:1.0]\n"; // SGF Header

    vector<Point> black_stones;
    vector<Point> white_stones;

    for (int row = 0; row < 19; ++row) {
        for (int col = 0; col < 19; ++col) {
            int stone = board_state.at<uchar>(row, col);
            if (stone == 1) { // Black stone
                black_stones.push_back(Point(col, row));
            } else if (stone == 2) { // White stone
                white_stones.push_back(Point(col, row));
            }
            // 0 (empty) is skipped
        }
    }
     // Add black stones using AB property
    if (!black_stones.empty()) {
        sgf << ";AB";
        for (const auto& stone : black_stones) {
            char sgf_col = 'a' + stone.x;
            char sgf_row = 'a' + stone.y;
            sgf << "[" << sgf_col << sgf_row << "]";
        }
    }

    // Add white stones using AW property
    if (!white_stones.empty()) {
        sgf << ";AW";
        for (const auto& stone : white_stones) {
            char sgf_col = 'a' + stone.x;
            char sgf_row = 'a' + stone.y;
            sgf << "[" << sgf_col << sgf_row << "]";
        }
    }
    sgf << ")\n";
    return sgf.str();
}