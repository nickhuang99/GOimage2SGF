#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "utility.h"
#include <iomanip>
#include <limits>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

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
    if (black_diff_add.size() == 1 && white_diff_add.size() == 0 && black_diff_remove.size() == 0 && white_diff_remove.size() == 0) {
        // Black played a stone.
        char sgf_col = 'a' + black_diff_add[0].x;
        char sgf_row = 'a' + black_diff_add[0].y;
        return ";B[" + string(1, sgf_col) + string(1, sgf_row) + "]";
    } else if (white_diff_add.size() == 1 && black_diff_add.size() == 0 && black_diff_remove.size() == 0 && white_diff_remove.size() == 0) {
        // White played a stone.
        char sgf_col = 'a' + white_diff_add[0].x;
        char sgf_row = 'a' + white_diff_add[0].y;
        return ";W[" + string(1, sgf_col) + string(1, sgf_row) + "]";
    } else if (white_diff_add.size() == 0 && black_diff_add.size() == 0) {
        // Handle captures.
        string captureString = "";
        for (Point p : black_diff_remove) {
            char sgf_col = 'a' + p.x;
            char sgf_row = 'a' + p.y;
            captureString += "DB[" + string(1, sgf_col) + string(1, sgf_row) + "]";
        }
        for (Point p : white_diff_remove) {
            char sgf_col = 'a' + p.x;
            char sgf_row = 'a' + p.y;
            captureString += "DW[" + string(1, sgf_col) + string(1, sgf_row) + "]";
        }
        return captureString;
    } else if (black_diff_add.size() > 1 || white_diff_add.size() > 1 || black_diff_remove.size() > 0 || white_diff_remove.size() > 0) {
        // More than one stone changed, which is an error for a single move.
        return ";ERROR: Multiple stone changes detected!";
    } else {
        // No changes detected
        return ";No move detected";
    }
}

// Function to generate SGF for the current board state
string generateSGF(const Mat& board_state) {
    ostringstream sgf;
    sgf << "(;FF[4]GM[1]SZ[19]AP[GoBoardAnalyzer:1.0]\n"; // SGF Header

    for (int row = 0; row < 19; ++row) {
        for (int col = 0; col < 19; ++col) {
            int stone = board_state.at<uchar>(row, col);
            char sgf_col = 'a' + col;
            char sgf_row = 'a' + row;
            if (stone == 1) { // Black stone
                sgf << "B[" << sgf_col << sgf_row << "]";
            } else if (stone == 2) { // White stone
                sgf << "W[" << sgf_col << sgf_row << "]";
            }
            // 0 (empty) is skipped
        }
    }
    sgf << ")\n";
    return sgf.str();
}

// Function to visually verify SGF data on the original image
void verifySGF(const Mat& image, const string& sgf_data) {
    Mat verification_image = image.clone(); // Create a copy to draw on

    // Parse the SGF data to extract stone positions.  A very basic parser is implemented.
    size_t pos = sgf_data.find("AP[GoBoardAnalyzer");
    if (pos == string::npos){
         pos = sgf_data.find("SZ[19]");
    }
    pos = sgf_data.find("]");
    if(pos == string::npos){
        cout << "not a valid sgf" << endl;
        return;
    }
    pos = 0;
    while (pos != string::npos) {
        size_t black_pos = sgf_data.find(";B[", pos);
        size_t white_pos = sgf_data.find(";W[", pos);

        if (black_pos == string::npos && white_pos == string::npos)
            break; // No more stones found

        if (black_pos != string::npos && (white_pos == string::npos || black_pos < white_pos)) {
            // Black stone
            string coord = sgf_data.substr(black_pos + 3, 2);
            int col = coord[0] - 'a';
            int row = coord[1] - 'a';
            if (row >= 0 && row < 19 && col >= 0 && col < 19)
                putText(verification_image, "B", Point(col * 35 + 10, row * 35 + 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2); // Mark with 'B'
            pos = black_pos + 5; // Move past this 'B[..]'
        } else {
            // White stone
            string coord = sgf_data.substr(white_pos + 3, 2);
            int col = coord[0] - 'a';
            int row = coord[1] - 'a';
            if (row >= 0 && row < 19 && col >= 0 && col < 19)
                putText(verification_image, "W", Point(col * 35 + 10, row * 35 + 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2); // Mark with 'W'
            pos = white_pos + 5; // Move past this 'W[..]'
        }
    }

    imshow("SGF Verification", verification_image);
    waitKey(0);
}

int main() {
    // 1. Load the Go board image.
    Mat image_bgr = imread("go_board.jpg"); // Hardcoded file name
    if (image_bgr.empty()) {
        cerr << "Error loading image\n";
        return -1;
    }

    // 2. Process the image to get the board state.
    Mat board_state;
    Mat board_with_stones;
    processGoBoard(image_bgr, board_state, board_with_stones);

    // 3. Generate the SGF for the current board state.
    string initial_sgf = generateSGF(board_state);

    // 4. Write the SGF to a file.
    ofstream sgf_file("current_board_state.sgf"); // Changed filename
    if (!sgf_file.is_open()) {
        cerr << "Error opening SGF file\n";
        return -1;
    }
    sgf_file << initial_sgf;
    sgf_file.close();

    cout << "SGF file (current_board_state.sgf) generated.\n";

    // 5. Verify the SGF data.
    verifySGF(image_bgr, initial_sgf);

    return 0;
}

