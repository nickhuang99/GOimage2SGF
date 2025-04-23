#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "utility.h"
#include <iomanip>
#include <limits>
#include <fstream> // Required for file output
#include <sstream> // Required for stringstream

using namespace cv;
using namespace std;

int main() {
    Mat image_bgr = imread("go_board.jpg");
    if (image_bgr.empty()) {
        cerr << "Error loading image\n";
        return -1;
    }

    Mat board_state;
    Mat board_with_stones;
    processGoBoard(image_bgr, board_state, board_with_stones);

    imshow("Detected Board State (K-Means HSV - Weighted Distance)", board_with_stones);
    waitKey(0);

    pair<vector<double>, vector<double>> grid_lines = detectUniformGrid(image_bgr);
    vector<double> horizontal_lines = grid_lines.first;
    vector<double> vertical_lines = grid_lines.second;
    vector<Point2f> intersection_points = findIntersections(horizontal_lines, vertical_lines);
    int num_intersections = intersection_points.size();

    cout << "Final horizontal lines (uniform spacing): " << horizontal_lines.size() << "\n";
    cout << "Final vertical lines (uniform spacing): " << vertical_lines.size() << "\n";
    cout << "Detected " << num_intersections << " intersection points.\n";
    cout << "Detected board state (K-Means HSV - Weighted Distance):\n" << board_state << endl;

    // Generate SGF file
    ofstream sgf_file("go_board.sgf"); // Open file for writing
    if (!sgf_file.is_open()) {
        cerr << "Error opening SGF file for writing!\n";
        return -1;
    }

    // SGF Header
    sgf_file << "(;FF[4]GM[1]SZ[19]AP[GoBoardAnalyzer:1.0]\n"; // Basic SGF header

    // Add moves based on board_state
    for (int row = 0; row < 19; ++row) {
        for (int col = 0; col < 19; ++col) {
            int stone = board_state.at<uchar>(row, col);
            if (stone == 1) { // Black stone
                char sgf_col = 'a' + col; // Convert column index to SGF letter (a-s)
                char sgf_row = 'a' + row; // Convert row index to SGF letter
                sgf_file << ";B[" << sgf_col << sgf_row << "]"; // Add black move
            } else if (stone == 2) { // White stone
                char sgf_col = 'a' + col;
                char sgf_row = 'a' + row;
                sgf_file << ";W[" << sgf_col << sgf_row << "]"; // Add white move
            } // 0 = empty, so no move recorded
        }
    }

    sgf_file << ")\n"; // Close the SGF file
    sgf_file.close();

    cout << "SGF file (go_board.sgf) generated successfully.\n";

    return 0;
}

