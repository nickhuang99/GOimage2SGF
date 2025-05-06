#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "common.h"

// --- Helper Function to Handle Keyboard Input for Calibration ---
// Takes the pressed key, modifies corner points (by reference).
// Returns the key code (27 for ESC, 's' for save, 0 or other keycode otherwise).
int handleCalibrationInput(int key, cv::Point2f& topLeft, cv::Point2f& topRight, int frame_width, int frame_height) {
    const int step = 5;
    int return_signal = 0;

    switch (key) {
        case 'u': topLeft.y -= step; topRight.y -= step; break;
        case 'd': topLeft.y += step; topRight.y += step; break;
        case 'n': topLeft.x += step; topRight.x -= step; break;
        case 'w': topLeft.x -= step; topRight.x += step; break;
        case 's': return_signal = 's'; break; // Signal save
        case 27: return_signal = 27; break; // Signal exit
        default:
             if (key != -1) return_signal = key; // Pass other keys through if needed
            break;
    }

    // Boundary checks (same as before)
    topLeft.x = std::max(0.0f, std::min((float)frame_width - 1, topLeft.x));
    topLeft.y = std::max(0.0f, std::min((float)frame_height - 1, topLeft.y));
    topRight.x = std::max(0.0f, std::min((float)frame_width - 1, topRight.x));
    topRight.y = std::max(0.0f, std::min((float)frame_height - 1, topRight.y));
    // Prevent crossing over (same as before)
    if (topRight.x < topLeft.x + 10.0f) {
         if (key == 'n') { topLeft.x -= step; topRight.x += step; }
         else if (key == 'w'){ topLeft.x += step; topRight.x -= step; }
         else { topRight.x = topLeft.x + 10.0f;}
    }

    return return_signal;
}

// --- Helper Function to Draw Calibration OSD (Unchanged) ---
void drawCalibrationOSD(cv::Mat& display_frame,
                       const cv::Point2f& tl, const cv::Point2f& tr,
                       const cv::Point2f& bl, const cv::Point2f& br)
{
    // ... (code from previous response: draws circles, lines, text in red etc) ...
    // --- Draw the Four Corner Circles ---
    int circle_radius = 5;
    cv::circle(display_frame, tl, circle_radius, cv::Scalar(0, 0, 255), -1);   // Red TL
    cv::circle(display_frame, tr, circle_radius, cv::Scalar(255, 0, 0), -1);   // Blue TR
    cv::circle(display_frame, br, circle_radius, cv::Scalar(0, 255, 255), -1); // Yellow BR
    cv::circle(display_frame, bl, circle_radius, cv::Scalar(255, 0, 255), -1); // Magenta BL

    // --- Draw Connecting Lines ---
    cv::Scalar line_color(0, 255, 0); // Green lines
    int line_thickness = 1;
    cv::line(display_frame, tl, tr, line_color, line_thickness);
    cv::line(display_frame, tr, br, line_color, line_thickness);
    cv::line(display_frame, br, bl, line_color, line_thickness);
    cv::line(display_frame, bl, tl, line_color, line_thickness);

    // --- OSD Text Settings ---
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    cv::Scalar help_text_color(0, 0, 255);     // Red for help text
    cv::Scalar coord_text_color(255, 200, 0); // Cyan for coordinates
    int text_thickness = 1;

    // --- Draw Help Text OSD ---
    std::string help_text = "Keys: u/d (up/down), w/n (wider/narrow), s (save), esc (exit)";
    double help_font_scale = 0.6;
    cv::Point help_text_origin(10, 20);
    cv::putText(display_frame, help_text, help_text_origin, font_face, help_font_scale, help_text_color, text_thickness, cv::LINE_AA);

    // --- Draw Coordinate Text OSD ---
    double coord_font_scale = 0.4;
    std::stringstream ss;
    ss << std::fixed << std::setprecision(0);

    ss.str(""); ss << "TL(" << tl.x << "," << tl.y << ")";
    cv::putText(display_frame, ss.str(), tl + cv::Point2f(10, -10), font_face, coord_font_scale, coord_text_color, text_thickness, cv::LINE_AA);
    ss.str(""); ss << "TR(" << tr.x << "," << tr.y << ")";
    cv::putText(display_frame, ss.str(), tr + cv::Point2f(-60, -10), font_face, coord_font_scale, coord_text_color, text_thickness, cv::LINE_AA);
    ss.str(""); ss << "BL(" << bl.x << "," << bl.y << ")";
    cv::putText(display_frame, ss.str(), bl + cv::Point2f(10, 20), font_face, coord_font_scale, coord_text_color, text_thickness, cv::LINE_AA);
    ss.str(""); ss << "BR(" << br.x << "," << br.y << ")";
    cv::putText(display_frame, ss.str(), br + cv::Point2f(-60, 20), font_face, coord_font_scale, coord_text_color, text_thickness, cv::LINE_AA);
}

// --- NEW Function to Save Corner Configuration (Unchanged) ---
bool saveCornerConfig(const std::string& filename,
                     const cv::Point2f& tl, const cv::Point2f& tr,
                     const cv::Point2f& bl, const cv::Point2f& br,
                     int frame_width, int frame_height)
{
    // ... (code from previous response to save key=value) ...
    std::ofstream outFile(filename);
    if (!outFile.is_open()) { /* ... error handling ... */ return false; }
    outFile << "# Go Board Corner Configuration..." << std::endl;
    outFile << "ImageWidth=" << frame_width << std::endl;
    outFile << "ImageHeight=" << frame_height << std::endl;
    outFile << std::fixed << std::setprecision(1);
    outFile << "\n# Pixel Coordinates" << std::endl;
    outFile << "TL_X_PX=" << tl.x << std::endl; outFile << "TL_Y_PX=" << tl.y << std::endl;
    outFile << "TR_X_PX=" << tr.x << std::endl; outFile << "TR_Y_PX=" << tr.y << std::endl;
    outFile << "BL_X_PX=" << bl.x << std::endl; outFile << "BL_Y_PX=" << bl.y << std::endl;
    outFile << "BR_X_PX=" << br.x << std::endl; outFile << "BR_Y_PX=" << br.y << std::endl;
    if (frame_width > 0 && frame_height > 0) {
        outFile << "\n# Percentage Coordinates (%)" << std::endl;
        outFile << "TL_X_PC=" << (tl.x / frame_width * 100.0f) << std::endl; outFile << "TL_Y_PC=" << (tl.y / frame_height * 100.0f) << std::endl;
        outFile << "TR_X_PC=" << (tr.x / frame_width * 100.0f) << std::endl; outFile << "TR_Y_PC=" << (tr.y / frame_height * 100.0f) << std::endl;
        outFile << "BL_X_PC=" << (bl.x / frame_width * 100.0f) << std::endl; outFile << "BL_Y_PC=" << (bl.y / frame_height * 100.0f) << std::endl;
        outFile << "BR_X_PC=" << (br.x / frame_width * 100.0f) << std::endl; outFile << "BR_Y_PC=" << (br.y / frame_height * 100.0f) << std::endl;
    }
    outFile.close();
    std::cout << "Corner configuration saved to " << filename << std::endl;
    return true;
}


// --- SIMPLIFIED Main Calibration Function ---
void runInteractiveCalibration(int camera_index) {
    cv::VideoCapture cap;
    cap.open(camera_index);
    if (!cap.isOpened()) { /* ... Error handling ... */ return; }

    std::cout << "Opened Camera Index: " << camera_index << std::endl;
    std::cout << "Starting Interactive Calibration..." << std::endl;

    std::string window_name = "Calibration - Adjust Top Corners (ESC to finish)";
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    cv::Mat frame;
    cv::Mat clean_frame_to_save;
    bool first_frame = true;
    cv::Point2f topLeft(0, 0), topRight(0, 0), bottomLeft(0, 0), bottomRight(0, 0);
    int frame_width = 0, frame_height = 0;

    while (true) { // Simple loop, break on ESC or error
        bool success = cap.read(frame);
        if (!success || frame.empty()) {
            std::cerr << "Warning: Could not read frame." << std::endl;
            if (cv::waitKey(50) == 27) break; // Allow ESC exit even on error
            continue;
        }

        // Store the clean frame
        clean_frame_to_save = frame.clone();

        if (first_frame) {
            frame_height = frame.rows;
            frame_width = frame.cols;
            // Initialize corners (same as before)
            float init_percent_x = 15.0f; float init_percent_y = 15.0f;
            topLeft = cv::Point2f(frame_width * init_percent_x / 100.0f, frame_height * init_percent_y / 100.0f);
            topRight = cv::Point2f(frame_width * (100.0f - init_percent_x) / 100.0f, frame_height * init_percent_y / 100.0f);
            bottomLeft = cv::Point2f(frame_width * init_percent_x / 100.0f, frame_height * (100.0f - init_percent_y) / 100.0f);
            bottomRight = cv::Point2f(frame_width * (100.0f - init_percent_x) / 100.0f, frame_height * (100.0f - init_percent_y) / 100.0f);
            first_frame = false;
            std::cout << "Initial Corners Set. Use Keys to Adjust Top Corners:" << std::endl;
            std::cout << "  U/D: Move Top Corners UP/DOWN" << std::endl;
            std::cout << "  N/W: Move Top Corners CLOSER(Narrow)/APART(Wider)" << std::endl;
            std::cout << "  S: Save current snapshot and config" << std::endl;
            std::cout << "  ESC: Exit and Print Final Coordinates" << std::endl;
        }

        // Create display copy and draw OSD
        cv::Mat display_frame = frame.clone();
        drawCalibrationOSD(display_frame, topLeft, topRight, bottomLeft, bottomRight);
        cv::imshow(window_name, display_frame);

        // Get key press and handle input/exit/save
        int key = cv::waitKey(30); // Wait 30ms and process events
        int key_result = handleCalibrationInput(key, topLeft, topRight, frame_width, frame_height);

        if (key_result == 27) { // ESC pressed
            break; // Exit loop
        } else if (key_result == 's') { // 's' pressed
            std::string image_filename = "./share/snapshot.jpg";
            std::string config_filename = "./share/config.txt";
            std::string debug_image_filename = "./share/snapshot_osd.jpg";
            std::cout << "Saving snapshot and config..." << std::endl;

            bool saved_image = false;
            if (bDebug) {
                 saved_image = cv::imwrite(debug_image_filename, display_frame);
            } else {
                 saved_image = cv::imwrite(image_filename, clean_frame_to_save);
            }
            if (!saved_image) std::cerr << "Error: Failed to save snapshot!" << std::endl;

            if (!saveCornerConfig(config_filename, topLeft, topRight, bottomLeft, bottomRight, frame_width, frame_height)) {
                 std::cerr << "Error: Failed to save config!" << std::endl;
            }
            std::cout << "Save complete." << std::endl;
        }
        // --- REMOVED the explicit getWindowProperty check ---
    }

    // --- Output Final Coordinates/Percentages (Same as before) ---
    // ...

    cap.release();
    cv::destroyAllWindows();
    std::cout << "Calibration window closed." << std::endl;
}