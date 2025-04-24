#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "utility.h"
#include <iomanip>
#include <limits>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <set>
#include <map>

using namespace cv;
using namespace std;

// SGF Header structure
struct SGFHeader {
    int gm;     // Game
    int ff;     // File Format
    string ca;  // Character Set
    string ap;  // Application
    int sz;     // Size of the board
};

// Structure to represent a single move, including captured stones
struct Move {
    int player; // 1 for Black, 2 for White
    int row;
    int col;
    vector<pair<int, int>> capturedStones; // Coordinates of captured stones
};

// Function to parse SGF header from a string
SGFHeader parseSGFHeader(const string& sgf_content) {
    SGFHeader header = {0}; // Initialize all fields to default values.

    // Use a stringstream to parse the SGF content.
    stringstream ss(sgf_content);
    string token;

    // Helper function to extract property values
    auto extractValue = [&](const string& propertyName) -> string {
        size_t pos = sgf_content.find(propertyName);
        if (pos != string::npos) {
            size_t start = sgf_content.find('[', pos) + 1;
            size_t end = sgf_content.find(']', pos);
            if (end > start)
                return sgf_content.substr(start, end - start);
        }
        return "";
    };

    // Parse GM property
    string gm_value = extractValue("GM");
    if (!gm_value.empty()) {
        try {
            header.gm = stoi(gm_value);
        } catch (const invalid_argument& e) {
            cerr << "Invalid GM value: " << gm_value << endl;
            header.gm = 0; //set to default
        }
    }

    // Parse FF property
    string ff_value = extractValue("FF");
    if (!ff_value.empty()) {
        try {
            header.ff = stoi(ff_value);
        } catch (const invalid_argument& e) {
            cerr << "Invalid FF value: " << ff_value << endl;
            header.ff = 0;
        }
    }

    // Parse CA property
    header.ca = extractValue("CA");

    // Parse AP property
    header.ap = extractValue("AP");

    // Parse SZ property
    string sz_value = extractValue("SZ");
    if (!sz_value.empty()) {
        try {
            header.sz = stoi(sz_value);
        } catch (const invalid_argument& e) {
            cerr << "Invalid SZ value: " << sz_value << endl;
            header.sz = 0;
        }
    }
    return header;
}

// Function to parse SGF game moves, differentiating setup and moves, and handling captures
void parseSGFGame(const string& sgfContent, vector<pair<int, int>>& setupBlack, vector<pair<int, int>>& setupWhite, vector<Move>& moves) {
    size_t pos = 0;
    setupBlack.clear();
    setupWhite.clear();
    moves.clear();

    // Helper function to convert SGF coordinates (e.g., "ab") to row and column
    auto sgfCoordToRowCol = [](const string& coord) -> pair<int, int> {
        if (coord.size() != 2 || !islower(coord[0]) || !islower(coord[1])) {
            return {-1, -1}; // Invalid coordinate
        }
        int col = coord[0] - 'a';
        int row = coord[1] - 'a';
        return {row, col};
    };

    while (pos < sgfContent.length()) {
        size_t blackSetupPos = sgfContent.find(";AB[", pos);
        size_t whiteSetupPos = sgfContent.find(";AW[", pos);
        size_t blackMovePos = sgfContent.find(";B[", pos);
        size_t whiteMovePos = sgfContent.find(";W[", pos);
        size_t removeStonesPos = sgfContent.find(";AE[", pos);

        size_t minPos = string::npos;
        int type = 0; // 1: AB, 2: AW, 3: B, 4: W, 5: AE

        if (blackSetupPos != string::npos && (minPos == string::npos || blackSetupPos < minPos)) {
            minPos = blackSetupPos;
            type = 1;
        }
        if (whiteSetupPos != string::npos && (minPos == string::npos || whiteSetupPos < minPos)) {
            minPos = whiteSetupPos;
            type = 2;
        }
        if (blackMovePos != string::npos && (minPos == string::npos || blackMovePos < minPos)) {
            minPos = blackMovePos;
            type = 3;
        }
        if (whiteMovePos != string::npos && (minPos == string::npos || whiteMovePos < minPos)) {
            minPos = whiteMovePos;
            type = 4;
        }
        if (removeStonesPos != string::npos && (minPos == string::npos || removeStonesPos < minPos)) {
            minPos = removeStonesPos;
            type = 5;
        }

        if (minPos == string::npos) {
            break; // No more moves or setup
        }

        pos = minPos;

        switch (type) {
            case 1: { // AB - Black setup stone
                size_t start = pos + 4;
                size_t end = sgfContent.find("]", start);
                if (end != string::npos) {
                    string coordStr = sgfContent.substr(start, end - start);
                    pair<int, int> coord = sgfCoordToRowCol(coordStr);
                    if (coord.first != -1 && coord.second != -1) {
                        setupBlack.push_back(coord);
                    }
                    pos = end + 1;
                } else {
                    pos = sgfContent.length(); // Invalid SGF, stop parsing
                }
                break;
            }
            case 2: { // AW - White setup stone
                size_t start = pos + 4;
                size_t end = sgfContent.find("]", start);
                if (end != string::npos) {
                    string coordStr = sgfContent.substr(start, end - start);
                    pair<int, int> coord = sgfCoordToRowCol(coordStr);
                    if (coord.first != -1 && coord.second != -1) {
                        setupWhite.push_back(coord);
                    }
                    pos = end + 1;
                } else {
                    pos = sgfContent.length(); // Invalid SGF, stop parsing
                }
                break;
            }
            case 3: { // B - Black move
                size_t start = pos + 3;
                size_t end = sgfContent.find("]", start);
                if (end != string::npos) {
                    string coordStr = sgfContent.substr(start, end - start);
                    pair<int, int> coord = sgfCoordToRowCol(coordStr);
                    if (coord.first != -1 && coord.second != -1) {
                        Move move;
                        move.player = 1;
                        move.row = coord.first;
                        move.col = coord.second;
                        // TODO: Parse captured stones if present (not in basic SGF)
                        moves.push_back(move);
                    }
                    pos = end + 1;
                } else {
                    pos = sgfContent.length(); // Invalid SGF, stop parsing
                }
                break;
            }
            case 4: { // W - White move
                size_t start = pos + 3;
                size_t end = sgfContent.find("]", start);
                if (end != string::npos) {
                    string coordStr = sgfContent.substr(start, end - start);
                    pair<int, int> coord = sgfCoordToRowCol(coordStr);
                    if (coord.first != -1 && coord.second != -1) {
                        Move move;
                        move.player = 2;
                        move.row = coord.first;
                        move.col = coord.second;
                        // TODO: Parse captured stones if present (not in basic SGF)
                        moves.push_back(move);
                    }
                    pos = end + 1;
                } else {
                    pos = sgfContent.length(); // Invalid SGF, stop parsing
                }
                break;
            }
             case 5: { // AE - Remove Stones.
                size_t start = pos + 4;
                size_t end = sgfContent.find("]", start);
                if (end != string::npos) {
                    string coordStr = sgfContent.substr(start, end - start);
                    pair<int, int> coord = sgfCoordToRowCol(coordStr);
                    if (coord.first != -1 && coord.second != -1) {
                        Move move;
                        move.player = 0;  // 0 indicates removal.
                        move.row = coord.first;
                        move.col = coord.second;
                        moves.push_back(move);
                    }
                    pos = end + 1;
                } else {
                    pos = sgfContent.length(); // Invalid SGF, stop parsing.
                }
                break;
            }
            default:
                pos = sgfContent.length(); // Should not happen, but handle it to prevent infinite loop
        }
    }
}


// Function to visually verify SGF data on the original image
void verifySGF(const Mat& image, const string& sgf_data, const vector<Point2f>& intersections) {
    Mat verification_image = image.clone(); // Create a copy to draw on
    vector<pair<int, int>> setupBlack;
    vector<pair<int, int>> setupWhite;
    vector<Move> moves;

    parseSGFGame(sgf_data, setupBlack, setupWhite, moves);

    // Draw setup stones
    for (const auto& stone : setupBlack) {
        int row = stone.first;
        int col = stone.second;
        if (row >= 0 && row < 19 && col >= 0 && col < 19) {
            Point2f pt = intersections[row * 19 + col];
            circle(verification_image, pt, 10, Scalar(0, 0, 0), 2); // Black
            cout << "Black stone at setup (" << col << ", " << row << ") - Pixel: (" << pt.x << ", " << pt.y << ")" << endl;
        }
    }
    for (const auto& stone : setupWhite) {
        int row = stone.first;
        int col = stone.second;
        if (row >= 0 && row < 19 && col >= 0 && col < 19) {
            Point2f pt = intersections[row * 19 + col];
            circle(verification_image, pt, 10, Scalar(255, 255, 255), 2); // White
            cout << "White stone at setup (" << col << ", " << row << ") - Pixel: (" << pt.x << ", " << pt.y << ")" << endl;
        }
    }

    // Draw moves
    for (const auto& move : moves) {
        int row = move.row;
        int col = move.col;
        if (row >= 0 && row < 19 && col >= 0 && col < 19) {
            Point2f pt = intersections[row * 19 + col];
            if (move.player == 1) {
                circle(verification_image, pt, 10, Scalar(0, 0, 255), 2); // Black
                cout << "Black stone at move (" << col << ", " << row << ") - Pixel: (" << pt.x << ", " << pt.y << ")" << endl;
            } else if (move.player == 2) {
                circle(verification_image, pt, 10, Scalar(255, 0, 0), 2); // White
                cout << "White stone at move (" << col << ", " << row << ") - Pixel: (" << pt.x << ", " << pt.y << ")" << endl;
            }
            else if (move.player == 0){
                 putText(verification_image, "X", pt, FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2); // Mark with 'X'
                 cout << "Stone removed at (" << col << ", " << row << ") - Pixel: (" << pt.x << ", " << pt.y << ")" << endl;
            }
             if (!move.capturedStones.empty()) {
                cout << "  Captured: ";
                for (const auto& captured : move.capturedStones) {
                    cout << "(" << captured.second << ", " << captured.first << ") ";
                }
                cout << endl;
            }
        }
    }

    imshow("SGF Verification", verification_image);
    waitKey(0);
}

// Function to compare two SGF strings for semantic equivalence (order-insensitive)
//bool compareSGF(const string& sgf1, const string& sgf2) {
//    // 1. Tokenize both SGF strings.  Use a simple delimiter-based approach.
//    vector<string> tokens1;
//    vector<string> tokens2;
//    stringstream ss1(sgf1);
//    stringstream ss2(sgf2);
//    string token;
//
//    while (getline(ss1, token, ';')) {
//        if (!token.empty())
//            tokens1.push_back(token);
//    }
//    while (getline(ss2, token, ';')) {
//        if (!token.empty())
//            tokens2.push_back(token);
//    }
//
//    // 2. Remove the first token if it is header
//    if (tokens1.size() > 0 && tokens1[0].find("GM[") != std::string::npos){
//        tokens1.erase(tokens1.begin());
//    }
//     if (tokens2.size() > 0 && tokens2[0].find("GM[") != std::string::npos){
//        tokens2.erase(tokens2.begin());
//    }
//    // 3. Convert the token vectors to sets for efficient comparison.
//    set<string> set1(tokens1.begin(), tokens1.end());
//    set<string> set2(tokens2.begin(), tokens2.end());
//
//    // 4. Check if the sets are equal.
//    return set1 == set2;
//}



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
     pair<vector<double>, vector<double>> grid_lines = detectUniformGrid(image_bgr);
    vector<double> horizontal_lines = grid_lines.first;
    vector<double> vertical_lines = grid_lines.second;
    vector<Point2f> intersection_points = findIntersections(horizontal_lines, vertical_lines);
    Mat previous_board_state = board_state.clone();


    // 3. Generate the SGF for the current board state.
    string generated_sgf = generateSGF(board_state, intersection_points);
    cout << "Generated SGF:\n" << generated_sgf << endl;

    // 4. Load the "correct" SGF from the file.
    // ifstream correct_sgf_file("ScreenToSGF.sgf.txt");
    // if (!correct_sgf_file.is_open()) {
    //     cerr << "Error opening correct SGF file\n";
    //     return -1;
    // }
    // stringstream correct_sgf_stream;
    // correct_sgf_stream << correct_sgf_file.rdbuf();
    // string correct_sgf = correct_sgf_stream.str();
    // correct_sgf_file.close();

    // cout << "\nCorrect SGF:\n" << generated_sgf << endl;

    // 5. Compare the generated SGF with the correct SGF.
    //if (generated_sgf == correct_sgf) {
    //    cout << "\nSGF generation is correct!\n";
    //} else {
    //    cout << "\nSGF generation is incorrect.\n";
    //}
    //bool areEquivalent = compareSGF(generated_sgf, correct_sgf);
    //if (areEquivalent)
    //    cout << "The two SGF are equivalent" << endl;
    //else
    //    cout << "The two SGF are NOT equivalent" << endl;

    // 6. Verify the SGF data.
    verifySGF(image_bgr, generated_sgf, intersection_points);

    return 0;
}

