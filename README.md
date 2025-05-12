# Go Environment Manager (GEM)

## Overview

The Go Environment Manager (GEM) is a command-line tool designed to analyze Go board positions from images or live webcams, manage SGF (Smart Game Format) files, and assist in Go game recording. It leverages OpenCV for image processing and supports both direct V4L2 and OpenCV's VideoCapture backend for webcam interactions. Recent enhancements focus on a robust calibration system that significantly improves the accuracy of board detection and stone classification.

## Features

* **Advanced Image Processing & Board Analysis:**
    * **Perspective Correction:** Corrects perspective distortion in Go board images using either default percentages or, more reliably, corner data from a `share/config.txt` file generated during calibration.
    * **Calibrated Grid Generation:** Utilizes calibration data (specifically, the ideal board representation from `getBoardCornersCorrected`) to generate a perfectly uniform 19x19 grid on the perspective-corrected image. This replaces previous complex line detection algorithms for improved accuracy and consistency.
    * **Calibrated Color-Based Stone Identification:**
        * Samples colors in the Lab color space at each grid intersection using an adaptive sampling radius (calculated by `calculateAdaptiveSampleRadius` based on board size in the corrected view).
        * Classifies stones (black, white, empty) by comparing sampled Lab colors against calibrated Lab values for black stones, white stones, and the average empty board color. These reference colors are stored in `share/config.txt` during calibration. This method (`classifyIntersectionsByCalibration`, `classifySingleIntersectionByDistance`) is more robust than relying solely on dynamic clustering.
        * Uses median Lab values from sampled regions (`getAverageLab`) to reduce noise.
    * **Post-processing:** Applies filters to reduce noise and improve stone detection accuracy (e.g., removing isolated white stones that might be glare).

* **SGF Management:**
    * Generates SGF files from an analyzed board state (AB/AW setup using `generateSGF`).
    * Parses SGF files to extract header information, setup stones, and game moves (`parseSGFHeader`, `parseSGFGame`).
    * Visually verifies a board image against an SGF file by overlaying SGF data on the image (`verifySGF`).
    * Compares two SGF files for semantic equivalence (`compareSGF`).
    * Determines the SGF move made between two board states (`determineSGFMove`).

* **Video Capture & Enhanced Calibration:**
    * **Device Probing:** Lists connected video devices and their capabilities (driver, card name, supported formats via V4L2 using `probeVideoDevices`).
    * **Snapshotting:** Captures images from a specified webcam using either V4L2 or OpenCV backend (`captureSnapshot`, `captureFrame`).
    * **Interactive Calibration (`runInteractiveCalibration`):**
        * Provides a sophisticated two-window interface:
            * **Raw Camera Feed (`WINDOW_RAW_FEED`):** Users adjust four corner markers (Top-Left, Top-Right, Bottom-Left, Bottom-Right) using keyboard inputs (`1`-`4` to select a corner, `i`/`j`/`k`/`l` to move the active corner).
            * **Corrected Preview (`WINDOW_CORRECTED_PREVIEW`):** Shows a live perspective-corrected view of the board based on the user-adjusted corners. This window displays a grid and target markers for stone placement, aiding in precise alignment.
        * **Comprehensive Configuration Saving:** Upon saving (`s` key):
            * Saves the raw pixel coordinates of the four adjusted corners to `share/config.txt`.
            * Samples Lab colors from the corrected view for:
                * The stone at the Top-Left physical location (expected black).
                * The stone at the Top-Right physical location (expected white).
                * The stone at the Bottom-Left physical location (expected black).
                * The stone at the Bottom-Right physical location (expected white).
                * The average color of empty board spaces (sampled from mid-points).
            * These sampled Lab values (L, a, b) are saved to `share/config.txt`.
            * The image dimensions (width, height) at the time of calibration are also saved.
        * **Snapshot Generation:**
            * `share/snapshot_raw_calibration.jpg`: The raw frame from the camera when calibration was saved.
            * `share/snapshot.jpg`: The perspective-corrected version of the board used for color sampling.
            * `share/snapshot_osd.jpg` (if debug mode is active): The raw frame with OSD markers.
    * **Capture Backend Selection:** Supports V4L2 (default) and OpenCV (`-M` or `--mode` option) for webcam operations.
    * **Resolution Control:** Allows specifying desired capture resolution (e.g., `640x480`, `1280x720`) via the `--size` option, facilitated by `trySetCameraResolution`.

* **Command-Line Interface:**
    * Flexible options to perform various tasks (see `gem.cpp` `displayHelpMessage`).
    * Debug mode (`-d` or `--debug`) for verbose output and visualizations.

* **Error Handling:**
    * Custom error class `GEMError` (using `THROWGEMERROR` macro) for detailed diagnostics, including file, line, and function information.
    * Specific `SGFError` for SGF parsing issues.

* **Configuration & Output:**
    * Primary calibration data is stored in `share/config.txt`.
    * Snapshots and SGF files are typically saved in the `./share/` directory or as specified by user.
    * Constants for these paths (`CALIB_CONFIG_PATH`, `CALIB_SNAPSHOT_PATH`, etc.) are defined in `camera.cpp` and declared in `common.h`.

## Building

To build GEM, you need:

* A C++ compiler supporting C++17 (e.g., g++)
* OpenCV (version 4 or later) - Development libraries (`libopencv-dev`)
* `libv4l-dev` (for V4L2 support)

Use the `build.sh` script provided, or compile manually:

```bash
./build.sh
# or
# g++ -g image.cpp sgf.cpp snapshot.cpp gem.cpp camera.cpp -o gem.exe `pkg-config --cflags --libs opencv4` -lv4l2

  
This command compiles all the C++ source files and links them with the necessary OpenCV and V4L2 libraries. The \-g flag includes debugging symbols.

## **Important: Permissions and the share Directory**

### **Webcam Access (V4L2 & OpenCV)**

Operations involving webcam access (probing, snapshots, calibration) require permission to access video device files (e.g., /dev/video0).

* **Running as root (sudo):** sudo ./gem.exe .... Generally not recommended for regular use.  
* **Using the video group (Recommended):** Add your user to the video group:  
  Bash  
  sudo usermod \-aG video $USER  
  You will need to **log out and log back in** for this change to take effect.

### **The share Directory**

GEM saves calibration data, snapshots, and generated SGF files into a share subdirectory within its working directory. Create this directory and ensure it has open permissions, especially if running with modified group IDs for device access:

Bash

mkdir \-p share  
chmod 777 share

This allows the program to write necessary files (e.g., config.txt, snapshot.jpg) without permission errors.

## **Usage**

Bash

./gem.exe \[options\]

### **Options:**

* \-d, \--debug: Enable debug output. Place this option first if used.  
* \-D, \--device \<device\_path\>: Specify the video device path (default: /dev/video0).  
* \-M, \--mode \<backend\>: Specify capture backend ('v4l2' or 'opencv', default: v4l2).  
* \--size \<WxH\>: Specify capture resolution (e.g., 640x480, 1280x720). Default: 640x480.  
* \-b, \--calibration: Run interactive board corner and color calibration using the webcam. Saves results to share/config.txt and various snapshots to the share/ directory.  
* \--test-calibration-config: Loads share/config.txt and share/snapshot.jpg (or share/snapshot\_osd.jpg if debug) to draw the calibrated corners on the snapshot for verification.  
* \-p, \--process-image \<image\_path\>: Process the specified Go board image. Requires valid share/config.txt.  
* \-g, \--generate-sgf \<input\_image\> \<output\_sgf\>: Process an image and generate an SGF file. Requires valid share/config.txt.  
* \-v, \--verify \<image\_path\> \<sgf\_path\>: Overlay SGF data on the image to visually verify stone positions. Requires valid share/config.txt.  
* \-c, \--compare \<sgf\_path1\> \<sgf\_path2\>: Compare two SGF files for semantic equivalence.  
* \--parse \<sgf\_path\>: Parse an SGF file and print its header, setup stones, and moves.  
* \--probe-devices: List available video capture devices and their supported formats (uses V4L2).  
* \-s, \--snapshot \<output\_file\_path\>: Capture a snapshot from the webcam and save it (e.g., share/my\_snapshot.jpg).  
* \-r, \--record-sgf \<output\_sgf\_path\>: Capture a snapshot, process it, and generate an SGF file (e.g., share/recorded\_game.sgf). Requires valid share/config.txt.  
* \--test-perspective \<image\_path\>: (Development/Test) Test the perspective correction on an image using current share/config.txt or defaults.  
* \-h, \--help: Display the help message.

### **Examples:**

1. **Enable debug mode, specify camera /dev/video1, and probe devices:**  
   Bash  
   ./gem.exe \-d \-D /dev/video1 \--probe-devices

2. **Run interactive calibration for /dev/video0 using OpenCV backend and 1280x720 resolution:**  
   Bash  
   ./gem.exe \-M opencv \--size 1280x720 \-b

3. **Take a snapshot and save it to share/board\_setup.jpg:**  
   Bash  
   ./gem.exe \-s share/board\_setup.jpg

4. **Process a local image (ensure share/config.txt exists and matches image dimensions if applicable):**  
   Bash  
   ./gem.exe \-p ./my\_go\_game.png

5. **Generate an SGF from an image:**  
   Bash  
   ./gem.exe \-g ./go\_board\_image.jpg ./output\_game.sgf

6. **Test your current calibration visually:**  
   Bash  
   ./gem.exe \--test-calibration-config

## **Functionality Details**

* **Board Processing (processGoBoard in image.cpp):**  
  1. **Requires valid calibration data** (share/config.txt) matching the input image's original dimensions for raw corner perspective correction, and for stone/board color references. Throws an error if calibration data is missing or incomplete.  
  2. Applies perspective correction using raw corner data from config.txt (getBoardCorners \-\> correctPerspective).  
  3. Converts the corrected image to Lab color space.  
  4. Generates a uniform 19x19 grid directly from the idealized board view defined by getBoardCornersCorrected (which uses a fixed percentage of the corrected image dimensions).  
  5. Finds intersection points of this ideal grid.  
  6. Samples Lab colors at each intersection (getAverageLab using an adaptive radius).  
  7. Classifies each intersection as black, white, or empty by comparing its sampled Lab color to the calibrated Lab reference values for black stones, white stones, and the empty board (from config.txt) using a weighted Euclidean distance metric and L-value heuristics (classifyIntersectionsByCalibration).  
  8. Applies a post-processing filter to remove isolated "white" stones (often glare).  
  9. Outputs the 19x19 board state matrix and an image with detected stones overlaid.  
* **Calibration (runInteractiveCalibration in camera.cpp):**  
  * Displays a raw camera feed where the user adjusts four corner markers (topLeft\_raw, topRight\_raw, bottomLeft\_raw, bottomRight\_raw) using 1-4 to select and ijkl to move.  
  * Simultaneously displays a corrected preview window showing the board based on the current raw corners and a target set of corrected points (getBoardCornersCorrected). This preview includes a grid and markers for stone placement guidance.  
  * When 's' is pressed:  
    * Saves the current raw corner coordinates and camera frame dimensions.  
    * Performs perspective correction on the current raw frame.  
    * Samples Lab colors from this corrected frame at the four corner locations (TL, TR, BL, BR assuming specific stone placements) and at board mid-points to calculate average black, white, and empty board Lab values.  
    * Saves all these values (raw corners, image dimensions, Lab values for TL, TR, BL, BR stones, and average board color) into share/config.txt.  
    * Saves share/snapshot\_raw\_calibration.jpg (the raw camera frame) and share/snapshot.jpg (the corrected frame used for sampling).  
* **SGF Generation (generateSGF in sgf.cpp):** Creates an SGF file with AB (add black) and AW (add white) properties based on the board state matrix from processGoBoard.  
* **Device Probing (probeVideoDevices in snapshot.cpp):** Iterates through /dev/videoX devices, queries their capabilities using V4L2 ioctl calls, and lists supported formats and frame sizes.

## **Contributing**

Contributions to GEM are welcome\! Please follow these general steps:

1. Fork the repository.  
2. Create a new branch for your feature or bug fix.  
3. Implement your changes.  
4. Test thoroughly, including building and running with various options.  
5. Update README.md if your changes affect usage, building, or add features.  
6. Submit a pull request with a clear description of your changes.

## **License**

MIT License  
Copyright (c) 2025 Qingzhe Huang/nickhuang99  
Permission is hereby granted, free of charge, to any person obtaining a copy  
of this software and associated documentation files (the "Software"), to deal  
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell  
copies of the Software, and to permit persons to whom the Software is  
furnished to do so, subject to the following conditions:  
The above copyright notice and this permission notice shall be included in all  
copies or substantial portions of the Software.  
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,  
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER  
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  
SOFTWARE.