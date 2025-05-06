#include <algorithm> // For std::max, std::min
#include <cmath>     // For std::pow (if using circular radius)
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// --- Helper Function to Handle Keyboard Input for Calibration ---
// Takes the pressed key, modifies corner points (by reference), and returns
// true if ESC was pressed.
bool handleCalibrationInput(int key, cv::Point2f &topLeft,
                            cv::Point2f &topRight, int frame_width,
                            int frame_height) {
  const int step = 5; // Pixels to move per key press

  switch (key) {
  case 'u': // Move UP
    topLeft.y -= step;
    topRight.y -= step;
    break;
  case 'd': // Move DOWN
    topLeft.y += step;
    topRight.y += step;
    break;
  case 'n': // Move CLOSER (Narrower) horizontally
    topLeft.x += step;
    topRight.x -= step;
    break;
  case 'w': // Move APART (Wider) horizontally
    topLeft.x -= step;
    topRight.x += step;
    break;
  case 27: // ESC key
    std::cout << "ESC pressed. Finishing calibration." << std::endl;
    return true; // Signal to exit
  }

  // --- Boundary Checks (ensure corners stay within frame) ---
  // Prevent points from going off-screen
  topLeft.x = std::max(0.0f, std::min((float)frame_width - 1, topLeft.x));
  topLeft.y = std::max(0.0f, std::min((float)frame_height - 1, topLeft.y));
  topRight.x = std::max(0.0f, std::min((float)frame_width - 1, topRight.x));
  topRight.y = std::max(0.0f, std::min((float)frame_height - 1, topRight.y));

  // Prevent TL and TR from crossing over horizontally
  if (topRight.x < topLeft.x + (2 * 5)) { // Using radius 5 for check
    // Reset based on which key was likely pressed causing the overlap
    if (key == 'n') { // Narrowing caused overlap
      topRight.x = topLeft.x + (2 * 5);
      topLeft.x -= step;     // Revert the last 'n' step for TL
    } else if (key == 'w') { // Widening caused overlap (less likely but
                             // possible if near edge)
      topLeft.x = topRight.x - (2 * 5);
      topRight.x += step; // Revert the last 'w' step for TR
    } else { // If overlap somehow happens without 'n' or 'w', just set a
             // minimum distance
      topRight.x = topLeft.x + (2 * 5);
    }
  }
  // (Add similar boundary checks if bottom corners become adjustable)

  return false; // Signal to continue
}

// --- Helper Function to Draw Calibration OSD ---
void drawCalibrationOSD(cv::Mat &display_frame, const cv::Point2f &tl,
                        const cv::Point2f &tr, const cv::Point2f &bl,
                        const cv::Point2f &br) {
  // --- Draw the Four Corner Circles ---
  int circle_radius = 5;
  // Colors: TL=Red, TR=Blue, BR=Yellow, BL=Magenta
  cv::circle(display_frame, tl, circle_radius, cv::Scalar(0, 0, 255), -1);
  cv::circle(display_frame, tr, circle_radius, cv::Scalar(255, 0, 0), -1);
  cv::circle(display_frame, br, circle_radius, cv::Scalar(0, 255, 255), -1);
  cv::circle(display_frame, bl, circle_radius, cv::Scalar(255, 0, 255), -1);

  // --- Draw Connecting Lines ---
  cv::Scalar line_color(0, 255, 0); // Green lines
  int line_thickness = 1;
  cv::line(display_frame, tl, tr, line_color, line_thickness);
  cv::line(display_frame, tr, br, line_color, line_thickness);
  cv::line(display_frame, br, bl, line_color, line_thickness);
  cv::line(display_frame, bl, tl, line_color, line_thickness);

  // --- OSD Text Settings ---
  int font_face = cv::FONT_HERSHEY_SIMPLEX;
  cv::Scalar help_text_color(0, 0, 255);    // Red for help text
  cv::Scalar coord_text_color(255, 200, 0); // Cyan for coordinates
  int text_thickness = 1;

  // --- Draw Help Text OSD ---
  std::string help_text = "Keys: u/d (up/down), w/n (wider/narrow), esc (exit)";
  double help_font_scale = 0.6; // Slightly larger font for help text
  cv::Point help_text_origin(10, 20);
  cv::putText(display_frame, help_text, help_text_origin, font_face,
              help_font_scale, help_text_color, text_thickness, cv::LINE_AA);

  // --- Draw Coordinate Text OSD ---
  double coord_font_scale = 0.4; // Smaller font for coordinates
  std::stringstream ss;

  // Top-Left Coordinates
  ss.str(""); // Clear stringstream
  ss << "TL(" << std::fixed << std::setprecision(0) << tl.x << "," << tl.y
     << ")";
  cv::putText(display_frame, ss.str(), tl + cv::Point2f(10, -10), font_face,
              coord_font_scale, coord_text_color, text_thickness, cv::LINE_AA);

  // Top-Right Coordinates
  ss.str("");
  ss << "TR(" << std::fixed << std::setprecision(0) << tr.x << "," << tr.y
     << ")";
  cv::putText(display_frame, ss.str(), tr + cv::Point2f(-60, -10), font_face,
              coord_font_scale, coord_text_color, text_thickness,
              cv::LINE_AA); // Adjust offset

  // Bottom-Left Coordinates
  ss.str("");
  ss << "BL(" << std::fixed << std::setprecision(0) << bl.x << "," << bl.y
     << ")";
  cv::putText(display_frame, ss.str(), bl + cv::Point2f(10, 20), font_face,
              coord_font_scale, coord_text_color, text_thickness,
              cv::LINE_AA); // Adjust offset

  // Bottom-Right Coordinates
  ss.str("");
  ss << "BR(" << std::fixed << std::setprecision(0) << br.x << "," << br.y
     << ")";
  cv::putText(display_frame, ss.str(), br + cv::Point2f(-60, 20), font_face,
              coord_font_scale, coord_text_color, text_thickness,
              cv::LINE_AA); // Adjust offset
}
// --- Updated Main Calibration Function ---
void runInteractiveCalibration(int camera_index) {
  cv::VideoCapture cap;
  cap.open(camera_index);

  if (!cap.isOpened()) {
    std::cerr << "Error: Could not open camera with index " << camera_index
              << std::endl;
    return;
  }

  std::cout << "Opened Camera Index: " << camera_index << std::endl;
  std::cout << "Starting Interactive Calibration..." << std::endl;

  std::string window_name = "Calibration - Adjust Top Corners (ESC to finish)";
  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

  cv::Mat frame;
  bool first_frame = true;
  cv::Point2f topLeft(0, 0), topRight(0, 0), bottomLeft(0, 0),
      bottomRight(0, 0);
  int frame_width = 0, frame_height = 0;

  while (true) {
    bool success = cap.read(frame);

    if (!success || frame.empty()) {
      std::cerr << "Warning: Could not read frame." << std::endl;
      if (cv::waitKey(50) == 27)
        break;
      continue;
    }

    if (first_frame) {
      frame_height = frame.rows;
      frame_width = frame.cols;
      float init_percent_x = 15.0f;
      float init_percent_y = 15.0f;
      topLeft = cv::Point2f(frame_width * init_percent_x / 100.0f,
                            frame_height * init_percent_y / 100.0f);
      topRight = cv::Point2f(frame_width * (100.0f - init_percent_x) / 100.0f,
                             frame_height * init_percent_y / 100.0f);
      bottomLeft =
          cv::Point2f(frame_width * init_percent_x / 100.0f,
                      frame_height * (100.0f - init_percent_y) / 100.0f);
      bottomRight =
          cv::Point2f(frame_width * (100.0f - init_percent_x) / 100.0f,
                      frame_height * (100.0f - init_percent_y) / 100.0f);
      first_frame = false;
      std::cout << "Initial Corners Set. Use Keys to Adjust Top Corners:"
                << std::endl;
      std::cout << "  U/D: Move Top Corners UP/DOWN" << std::endl;
      std::cout << "  N/W: Move Top Corners CLOSER(Narrow)/APART(Wider)"
                << std::endl;
      std::cout << "  ESC: Exit and Print Final Coordinates" << std::endl;
    }

    // Create a copy to draw on
    cv::Mat display_frame = frame.clone();

    // Draw the OSD elements using the helper function
    drawCalibrationOSD(display_frame, topLeft, topRight, bottomLeft,
                       bottomRight);

    // Display the frame
    cv::imshow(window_name, display_frame);

    // Handle keyboard input using the helper function
    int key = cv::waitKey(30);
    if (handleCalibrationInput(key, topLeft, topRight, frame_width,
                               frame_height)) {
      break; // Exit if ESC was pressed (handleCalibrationInput returned true)
    }
  }

  // --- Output Final Coordinates/Percentages (Same as before) ---
  std::cout << "\n--- Final Corner Pixel Coordinates ---" << std::endl;
  std::cout << "Top Left:     (" << topLeft.x << ", " << topLeft.y << ")"
            << std::endl;
  std::cout << "Top Right:    (" << topRight.x << ", " << topRight.y << ")"
            << std::endl;
  std::cout << "Bottom Right: (" << bottomRight.x << ", " << bottomRight.y
            << ")" << std::endl;
  std::cout << "Bottom Left:  (" << bottomLeft.x << ", " << bottomLeft.y << ")"
            << std::endl;

  if (frame_width > 0 && frame_height > 0) {
    std::cout << "\n--- Final Corner Percentages ---" << std::endl;
    std::cout << std::fixed
              << std::setprecision(1); // Set precision for percentages
    std::cout << "TL X%: " << (topLeft.x / frame_width * 100.0f) << std::endl;
    std::cout << "TL Y%: " << (topLeft.y / frame_height * 100.0f) << std::endl;
    std::cout << "TR X%: " << (topRight.x / frame_width * 100.0f) << std::endl;
    std::cout << "TR Y%: " << (topRight.y / frame_height * 100.0f) << std::endl;
    std::cout << "BR X%: " << (bottomRight.x / frame_width * 100.0f)
              << std::endl;
    std::cout << "BR Y%: " << (bottomRight.y / frame_height * 100.0f)
              << std::endl;
    std::cout << "BL X%: " << (bottomLeft.x / frame_width * 100.0f)
              << std::endl;
    std::cout << "BL Y%: " << (bottomLeft.y / frame_height * 100.0f)
              << std::endl;
  }

  // Release resources
  cap.release();
  cv::destroyAllWindows();
  std::cout << "Calibration window closed." << std::endl;
}