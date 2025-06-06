# ---

**Go Environment Manager (GEM)**

## **Overview**

The Go Environment Manager (GEM) is a command-line tool designed to analyze Go board positions from images or live webcams, manage SGF (Smart Game Format) files, and assist in Go game recording. It leverages OpenCV for image processing and supports both direct V4L2 and OpenCV's VideoCapture backend for webcam interactions. Recent enhancements focus on a robust, multi-pass corner detection system that significantly improves the accuracy and reliability of board analysis under various conditions.

## **Features**

* **Advanced Image Processing & Board Analysis:**  
  * **Robust, Two-Pass Corner Detection:** The system now uses a sophisticated, multi-stage process (adaptive\_detect\_stone\_robust) to reliably find the four corner stones of the board111. This method is highly resilient to variations in lighting, board position, and perspective distortion.  
    * **Iterative Guessing (Pass 1):** Instead of relying on a single, fixed guess, the algorithm iteratively tests a grid of initial position estimates within each quadrant of the image to find a likely stone candidate22. This ensures the stone is found even if the board is not perfectly centered.  
    * **Shape-First Blob Detection:** For each guess, it performs a perspective warp and then searches for the best "stone-like" shape using find\_best\_round\_shape\_iterative33. This advanced helper function prioritizes geometric properties (circularity, area) over fixed color values, allowing it to find stones in inconsistent lighting.  
    * **Refinement (Pass 2):** Once a candidate blob is found in Pass 1, its center is mapped back to the original image. This highly accurate coordinate is then used to perform a second, refined perspective transform, ensuring the final board analysis is precise.  
  * **Calibrated Grid Generation:** After perspective correction, the tool generates a mathematically perfect 19x19 grid based on the detected corner locations44. This avoids the noise and unreliability of traditional line-detection algorithms.  
  * **Calibrated Color-Based Stone Identification:**  
    * Samples colors in the perceptually uniform **Lab color space** at each grid intersection.  
    * Classifies stones (black, white, empty) by comparing sampled colors against calibrated Lab values stored in share/config.txt.  
    * Uses a weighted distance formula that is more robust to lighting variations than simple thresholding55.  
* **Camera & Calibration Workflows:**  
  * **Interactive Calibration (-B):** A user-friendly graphical interface to manually adjust the four board corners, sample stone and board colors, and save all parameters to share/config.txt6.  
  * **Automated Calibration from Snapshot (-A):** If a share/snapshot.jpg exists (ideally from a previous calibration), this mode runs the full robust corner detection and color sampling non-interactively to generate a new configuration file7.  
  * **Multiple Capture Backends:** Supports both OpenCV's default VideoCapture (-M cv) and direct V4L2 (-M v4l2) for webcam interaction on Linux systems8.  
* **SGF & Game Management:**  
  * **SGF Generation (-p):** Analyzes a board image and generates an SGF file representing the current position99.  
  * **SGF Comparison (-g):** Compares two board states (e.g., before.sgf and after.sgf) to generate a new SGF file containing only the moves made between the two states10.  
  * **Live Game Recording (-v):** A comprehensive workflow to record a live game by taking snapshots, analyzing board changes, and appending moves to a game record SGF file.  
* **Debugging and Visualization:**  
  * **Debug Mode (-d):** Enables detailed console logging and displays intermediate OpenCV image processing steps, including the new iterative corner search attempts11111111.  
  * **Log Levels (-O):** Granular control over log verbosity, from NONE to DEBUG12.  
  * **SGF Rendering (-s):** Renders a board position from an SGF file and saves it as an image, useful for creating test cases or visualizing game states13131313.

## **Building**

The project uses a simple build script and relies on pkg-config to find the necessary OpenCV 4 libraries.

Bash

\# Ensure you have g++, OpenCV 4, and libv4l-dev installed  
\# sudo apt-get install build-essential libopencv-dev pkg-config libv4l-dev

\# Run the build script  
./build.sh

This compiles all .cpp files and creates the executable gem.exe in the root directory14.

## **Basic Usage**

All workflows rely on a properly calibrated share/config.txt file for best results. It is highly recommended to run the interactive calibration at least once.

1. **First-Time Calibration:**  
   Bash  
   \# Run interactive calibration with your webcam  
   ./gem.exe \-B \-d 

   * Follow the on-screen instructions to align the corners and sample colors. A share/config.txt and share/snapshot.jpg will be saved.  
2. **Analyze a Board Image:**  
   Bash  
   \# Process the snapshot taken during calibration to generate an SGF  
   ./gem.exe \-p ./share/snapshot.jpg

3. **Record a Live Game:**  
   Bash  
   \# Start the live game recording workflow  
   ./gem.exe \-v

### **Command-Line Options**

* \-h, \--help: Display the help message.  
* \-d, \--debug: Enable debug mode (shows intermediate images).  
* \-B, \--run-interactive-calibration: Start the interactive board calibration workflow.  
* \-A, \--run-auto-calibration: Run automatic calibration using share/snapshot.jpg.  
* \-p \<image\_path\>, \--process-image \<image\_path\>: Process a single image to detect stones.  
* \-g \<before.sgf\> \<after.sgf\>, \--generate-sgf-diff \<before.sgf\> \<after.sgf\>: Generate the difference between two SGF files.  
* \-s \<input.sgf\> \<output\_image.jpg\>, \--sgf-to-image \<input.sgf\> \<output\_image.jpg\>: Render an SGF file to an image.  
* \-v, \--video-capture-workflow: Start the live game recording workflow.  
* \--robust-corners: Use the advanced robust corner detection algorithm.  
* \-O \<level\>, \--log-level \<level\>: Set the logging verbosity (0=NONE, 1=ERROR, 2=WARN, 3=INFO, 4=DEBUG).  
* \-D \<path\>, \--device \<path\>: Specify the camera device path (e.g., /dev/video0).  
* \-M \<mode\>, \--capture-mode \<mode\>: Set capture mode (v4l2 or cv).  
* \-S \<WxH\>, \--size \<WxH\>: Set camera capture resolution (e.g., 1280x720).

## **Contributing**

Contributions are welcome\! Please follow these general steps:

1. Fork the repository.  
2. Create a new branch for your feature or bug fix.  
3. Implement your changes.  
4. Test thoroughly,15 including building and running with various options.  
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
**Sources**  
1\. [https://github.com/nickhuang99/GOimage2SGF](https://github.com/nickhuang99/GOimage2SGF)