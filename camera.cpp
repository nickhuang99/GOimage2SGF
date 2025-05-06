#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "common.h"

// --- Helper Function to Handle Keyboard Input for Calibration ---
// Takes the pressed key, modifies corner points (by reference).
// Update function signature:
int handleCalibrationInput(int key, cv::Point2f &topLeft, cv::Point2f &topRight,
                           cv::Point2f &bottomLeft,
                           cv::Point2f &bottomRight, // Add these
                           int frame_width, int frame_height) {
  const int step = 5;
  int return_signal = 0;

  switch (key) {
  // Top edge controls
  case 'u':
    topLeft.y -= step;
    topRight.y -= step;
    break;
  case 'd':
    topLeft.y += step;
    topRight.y += step;
    break;
  case 'w':
    topLeft.x -= step;
    topRight.x += step;
    break;
  case 'n':
    topLeft.x += step;
    topRight.x -= step;
    break;

  // Bottom edge controls
  case 'k':
    bottomLeft.y -= step;
    bottomRight.y -= step;
    break; // bottom edge up
  case 'j':
    bottomLeft.y += step;
    bottomRight.y += step;
    break; // bottom edge down
  case 'l':
    bottomLeft.x -= step;
    bottomRight.x += step;
    break; // bottom edge wider (NEW: 'l' instead of ',')
  case 'm':
    bottomLeft.x += step;
    bottomRight.x -= step;
    break; // bottom edge narrower

  case 's':
    return_signal = 's';
    break;
  case 27:
    return_signal = 27;
    break; // ESC
  default:
    if (key != -1)
      return_signal = key;
    break;
  }

  // Boundary checks for all four corners
  topLeft.x = std::max(0.0f, std::min((float)frame_width - 1, topLeft.x));
  topLeft.y = std::max(0.0f, std::min((float)frame_height - 1, topLeft.y));
  topRight.x = std::max(0.0f, std::min((float)frame_width - 1, topRight.x));
  topRight.y = std::max(0.0f, std::min((float)frame_height - 1, topRight.y));

  bottomLeft.x = std::max(0.0f, std::min((float)frame_width - 1, bottomLeft.x));
  bottomLeft.y =
      std::max(0.0f, std::min((float)frame_height - 1, bottomLeft.y));
  bottomRight.x =
      std::max(0.0f, std::min((float)frame_width - 1, bottomRight.x));
  bottomRight.y =
      std::max(0.0f, std::min((float)frame_height - 1, bottomRight.y));

  // Prevent crossing over for top edge (existing logic)
  if (topRight.x < topLeft.x + 10.0f) { // Minimum 10px separation
    if (key == 'n') {
      topLeft.x -= step;
      topRight.x += step;
    } // Revert if crossed by 'n'
    else if (key == 'w') {
      topLeft.x += step;
      topRight.x -= step;
    } // Revert if crossed by 'w'
    else {
      topRight.x = topLeft.x + 10.0f;
    } // Generic fix if already crossed
  }

  // Prevent crossing over for bottom edge (NEW logic)
  if (bottomRight.x < bottomLeft.x + 10.0f) { // Minimum 10px separation
    if (key == 'm') {
      bottomLeft.x -= step;
      bottomRight.x += step;
    } // Revert if crossed by 'm'
    else if (key == ',') {
      bottomLeft.x += step;
      bottomRight.x -= step;
    } // Revert if crossed by ',' (comma)
    else {
      bottomRight.x = bottomLeft.x + 10.0f;
    } // Generic fix
  }

  // Optional: Prevent vertical crossing (e.g., TL.y > BL.y)
  // For simplicity, this is not added here, but could be considered.
  // Users can visually manage this. The perspective transform can handle
  // non-convex quads but it's generally better if the quadrilateral is
  // well-behaved.

  return return_signal;
}

// --- Helper Function to Draw Calibration OSD (Unchanged) ---
void drawCalibrationOSD(cv::Mat &display_frame, const cv::Point2f &tl,
                        const cv::Point2f &tr, const cv::Point2f &bl,
                        const cv::Point2f &br) {
  // ... (code from previous response: draws circles, lines, text in red etc)
  // ...
  // --- Draw the Four Corner Circles ---
  int circle_radius = 5;
  cv::circle(display_frame, tl, circle_radius, cv::Scalar(0, 0, 255),
             -1); // Red TL
  cv::circle(display_frame, tr, circle_radius, cv::Scalar(255, 0, 0),
             -1); // Blue TR
  cv::circle(display_frame, br, circle_radius, cv::Scalar(0, 255, 255),
             -1); // Yellow BR
  cv::circle(display_frame, bl, circle_radius, cv::Scalar(255, 0, 255),
             -1); // Magenta BL

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
  double help_font_scale = 0.5;
  cv::Point help_text_origin(10, 20);
  cv::Point help_text_origin_line2(10, 35);

  // --- Draw Help Text OSD ---
  // UPDATED HELP TEXT with 'l' and all lowercase
  std::string help_text_line1 =
      "top: u/d (y), w/n (x) | bot: k/j (y), l/m (x)"; // Changed ',' to 'l'
  std::string help_text_line2 = "s: save, esc: exit";  // All lowercase

  cv::putText(display_frame, help_text_line1, help_text_origin, font_face,
              help_font_scale, help_text_color, text_thickness, cv::LINE_AA);
  cv::putText(display_frame, help_text_line2, help_text_origin_line2, font_face,
              help_font_scale, help_text_color, text_thickness, cv::LINE_AA);

  // --- Draw Coordinate Text OSD ---
  double coord_font_scale = 0.4;
  std::stringstream ss;
  ss << std::fixed << std::setprecision(0);

  ss.str("");
  ss << "TL(" << tl.x << "," << tl.y << ")";
  cv::putText(display_frame, ss.str(), tl + cv::Point2f(10, -10), font_face,
              coord_font_scale, coord_text_color, text_thickness, cv::LINE_AA);
  ss.str("");
  ss << "TR(" << tr.x << "," << tr.y << ")";
  cv::putText(display_frame, ss.str(), tr + cv::Point2f(-60, -10), font_face,
              coord_font_scale, coord_text_color, text_thickness, cv::LINE_AA);
  ss.str("");
  ss << "BL(" << bl.x << "," << bl.y << ")";
  cv::putText(display_frame, ss.str(), bl + cv::Point2f(10, 20), font_face,
              coord_font_scale, coord_text_color, text_thickness, cv::LINE_AA);
  ss.str("");
  ss << "BR(" << br.x << "," << br.y << ")";
  cv::putText(display_frame, ss.str(), br + cv::Point2f(-60, 20), font_face,
              coord_font_scale, coord_text_color, text_thickness, cv::LINE_AA);
}

// --- NEW Function to Save Corner Configuration (Unchanged) ---
bool saveCornerConfig(const std::string &filename, const cv::Point2f &tl,
                      const cv::Point2f &tr, const cv::Point2f &bl,
                      const cv::Point2f &br, int frame_width,
                      int frame_height) {
  // ... (code from previous response to save key=value) ...
  std::ofstream outFile(filename);
  if (!outFile.is_open()) { /* ... error handling ... */
    return false;
  }
  outFile << "# Go Board Corner Configuration..." << std::endl;
  outFile << "ImageWidth=" << frame_width << std::endl;
  outFile << "ImageHeight=" << frame_height << std::endl;
  outFile << std::fixed << std::setprecision(1);
  outFile << "\n# Pixel Coordinates" << std::endl;
  outFile << "TL_X_PX=" << tl.x << std::endl;
  outFile << "TL_Y_PX=" << tl.y << std::endl;
  outFile << "TR_X_PX=" << tr.x << std::endl;
  outFile << "TR_Y_PX=" << tr.y << std::endl;
  outFile << "BL_X_PX=" << bl.x << std::endl;
  outFile << "BL_Y_PX=" << bl.y << std::endl;
  outFile << "BR_X_PX=" << br.x << std::endl;
  outFile << "BR_Y_PX=" << br.y << std::endl;
  if (frame_width > 0 && frame_height > 0) {
    outFile << "\n# Percentage Coordinates (%)" << std::endl;
    outFile << "TL_X_PC=" << (tl.x / frame_width * 100.0f) << std::endl;
    outFile << "TL_Y_PC=" << (tl.y / frame_height * 100.0f) << std::endl;
    outFile << "TR_X_PC=" << (tr.x / frame_width * 100.0f) << std::endl;
    outFile << "TR_Y_PC=" << (tr.y / frame_height * 100.0f) << std::endl;
    outFile << "BL_X_PC=" << (bl.x / frame_width * 100.0f) << std::endl;
    outFile << "BL_Y_PC=" << (bl.y / frame_height * 100.0f) << std::endl;
    outFile << "BR_X_PC=" << (br.x / frame_width * 100.0f) << std::endl;
    outFile << "BR_Y_PC=" << (br.y / frame_height * 100.0f) << std::endl;
  }
  outFile.close();
  std::cout << "Corner configuration saved to " << filename << std::endl;
  return true;
}

// Definition of the new utility function
bool trySetCameraResolution(cv::VideoCapture &cap, int desired_width,
                            int desired_height, bool attempt_fallback_format) {
  if (!cap.isOpened()) {
    if (bDebug)
      std::cerr << "Debug: trySetCameraResolution - VideoCapture not open."
                << std::endl;
    return false;
  }

  bool success = false;
  int initial_fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
  std::string initial_format_name = "MJPEG";

  if (bDebug) {
    std::cout << "Debug: trySetCameraResolution - Attempting "
              << initial_format_name << " at " << desired_width << "x"
              << desired_height << std::endl;
  }

  // Attempt 1: MJPEG (or a primary preferred format)
  cap.set(cv::CAP_PROP_FOURCC, initial_fourcc);
  cap.set(cv::CAP_PROP_FRAME_WIDTH, static_cast<double>(desired_width));
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<double>(desired_height));

  // Allow some time for settings to apply on some cameras/backends
  // cap.grab(); // You might need a short delay or a grab here for settings to
  // stick before get() cv::waitKey(50); // Small delay

  int actual_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int actual_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  int actual_fourcc_int = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
  char fourcc_str[5] = {0};
  memcpy(fourcc_str, &actual_fourcc_int, 4);

  if (bDebug) {
    std::cout << "Debug: trySetCameraResolution - After 1st attempt (MJPEG): "
              << "Actual Size: " << actual_width << "x" << actual_height
              << ", Actual FOURCC: " << fourcc_str
              << " (Requested: " << desired_width << "x" << desired_height
              << ")" << std::endl;
  }

  if (actual_width == desired_width && actual_height == desired_height) {
    success = true;
  }

  // Attempt 2: Fallback format (e.g., YUYV) if first failed and fallback is
  // enabled
  if (!success && attempt_fallback_format) {
    if (bDebug)
      std::cout << "Debug: trySetCameraResolution - Initial attempt failed. "
                   "Trying fallback YUYV."
                << std::endl;

    int fallback_fourcc = cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V');
    std::string fallback_format_name = "YUYV";

    cap.set(cv::CAP_PROP_FOURCC, fallback_fourcc);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, static_cast<double>(desired_width));
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<double>(desired_height));

    // cap.grab();
    // cv::waitKey(50);

    actual_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    actual_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    actual_fourcc_int = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
    memcpy(fourcc_str, &actual_fourcc_int, 4); // Update fourcc_str

    if (bDebug) {
      std::cout << "Debug: trySetCameraResolution - After 2nd attempt (YUYV): "
                << "Actual Size: " << actual_width << "x" << actual_height
                << ", Actual FOURCC: " << fourcc_str
                << " (Requested: " << desired_width << "x" << desired_height
                << ")" << std::endl;
    }

    if (actual_width == desired_width && actual_height == desired_height) {
      success = true;
    }
  }

  if (success && bDebug) {
    std::cout << "Debug: trySetCameraResolution - Successfully set to "
              << actual_width << "x" << actual_height
              << " with FOURCC: " << fourcc_str << std::endl;
  } else if (!success && bDebug) {
    std::cout
        << "Debug: trySetCameraResolution - Failed to set desired resolution "
        << desired_width << "x" << desired_height
        << ". Final actual size: " << actual_width << "x" << actual_height
        << ", FOURCC: " << fourcc_str << std::endl;
  }
  return success;
}

// --- SIMPLIFIED Main Calibration Function ---
void runInteractiveCalibration(int camera_index) {
  cv::VideoCapture cap;
  // Instead of cap.open(camera_index);
  cap.open(camera_index, cv::CAP_V4L2);
  if (!cap.isOpened()) {
    std::string error_message =
        "OpenCV (CAP_V4L2) failed to open video capture device with index " +
        Num2Str(camera_index).str() + ". " + "Device path hint: /dev/videoX " +
        "Possible reasons: \n"
        "1. Camera not connected or powered.\n"
        "2. Invalid camera index or device path.\n"
        "3. Camera is already in use by another application.\n"
        "4. Insufficient permissions (check /dev/videoX ownership/group, add "
        "user to 'video' group, or try sudo if appropriate).\n"
        "5. V4L2 backend issues or camera driver problems.\n"
        "6. OpenCV was not built with V4L2 support for this camera.";
    THROWGEMERROR(error_message);
  }
  if (bDebug)
    std::cout << "Debug: Calibration - Requesting frame size "
              << g_capture_width << "x" << g_capture_height << "." << std::endl;
  // Use the new utility function
  if (!trySetCameraResolution(cap, g_capture_width, g_capture_height, true)) {
    // Get current actuals for the error message
    int final_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int final_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::stringstream ss;
    ss << "Calibration: Failed to set desired resolution " << g_capture_width
       << "x" << g_capture_height
       << " even after fallback. Actual resolution is " << final_width << "x"
       << final_height << ".";
    THROWGEMERROR(ss.str());
  }
  // Proceed with these dimensions, as trySetCameraResolution confirmed them or
  // failed trying.
  int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  if (bDebug) {
    std::cout << "Debug: Calibration - Actual frame dimensions from camera: "
              << frame_width << "x" << frame_height << std::endl;
  }
  std::cout << "Opened Camera Index: " << camera_index << std::endl;
  std::cout << "Starting Interactive Calibration..." << std::endl;

  std::string window_name = "Calibration - Adjust Top Corners (ESC to finish)";
  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

  cv::Mat frame;
  cv::Mat clean_frame_to_save;
  float init_percent_x = 15.0f;
  float init_percent_y = 15.0f;
  cv::Point2f topLeft(frame_width * init_percent_x / 100.0f,
                      frame_height * init_percent_y / 100.0f);
  cv::Point2f topRight(frame_width * (100.0f - init_percent_x) / 100.0f,
                       frame_height * init_percent_y / 100.0f);
  cv::Point2f bottomLeft(frame_width * init_percent_x / 100.0f,
                         frame_height * (100.0f - init_percent_y) / 100.0f);
  cv::Point2f bottomRight(frame_width * (100.0f - init_percent_x) / 100.0f,
                          frame_height * (100.0f - init_percent_y) / 100.0f);

  // UPDATED CONSOLE HELP TEXT with 'l' and all lowercase
  std::cout << "Initial Corners Set. Use Keys to Adjust:" << std::endl;
  std::cout << "  top edge controls:" << std::endl;
  std::cout << "    u: move top edge up" << std::endl;
  std::cout << "    d: move top edge down" << std::endl;
  std::cout << "    w: make top edge wider" << std::endl;
  std::cout << "    n: make top edge narrower" << std::endl;
  std::cout << "  bottom edge controls:" << std::endl;
  std::cout << "    k: move bottom edge up" << std::endl;
  std::cout << "    j: move bottom edge down" << std::endl;
  std::cout << "    l: make bottom edge wider"
            << std::endl; // Changed from COMMA
  std::cout << "    m: make bottom edge narrower" << std::endl;
  std::cout << "------------------------------------" << std::endl;
  std::cout << "  s: save current snapshot and config to ./share/" << std::endl;
  std::cout << "  esc: exit" << std::endl;

  while (true) { // Simple loop, break on ESC or error
    bool success = cap.read(frame);
    if (!success || frame.empty()) {
      std::cerr << "Warning: Could not read frame." << std::endl;
      if (cv::waitKey(50) == 27)
        break; // Allow ESC exit even on error
      continue;
    }

    // Store the clean frame
    clean_frame_to_save = frame.clone();

    // Create display copy and draw OSD
    cv::Mat display_frame = frame.clone();
    drawCalibrationOSD(display_frame, topLeft, topRight, bottomLeft,
                       bottomRight);
    cv::imshow(window_name, display_frame);

    // Get key press and handle input/exit/save
    int key = cv::waitKey(30); // Wait 30ms and process events
    int key_result =
        handleCalibrationInput(key, topLeft, topRight, bottomLeft, bottomRight,
                               frame_width, frame_height);

    if (key_result == 27) {         // ESC pressed
      break;                        // Exit loop
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
      if (!saved_image)
        std::cerr << "Error: Failed to save snapshot!" << std::endl;

      if (!saveCornerConfig(config_filename, topLeft, topRight, bottomLeft,
                            bottomRight, frame_width, frame_height)) {
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