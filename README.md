# Go Environment Manager (GEM)

## Overview

The Go Environment Manager (GEM) is a command-line tool designed to analyze Go board positions from images or live webcams, manage SGF (Smart Game Format) files, and assist in Go game recording. It leverages OpenCV for image processing and V4L2 (or OpenCV's backend) for webcam interactions.

## Features

* **Image Processing:**  
    * Corrects perspective distortion in Go board images.  
    * Detects the 19x19 grid lines on the Go board.  
    * Identifies black and white stones at intersections using color clustering (Lab color space).  
    * Applies post-processing filters to improve stone detection accuracy.  
* **SGF Management:**  
    * Generates SGF files from an analyzed board state (AB/AW setup).  
    * Parses SGF files to extract header information, setup stones, and game moves.  
    * Verifies a board image against an SGF file by overlaying SGF data on the image.  
    * Compares two SGF files for semantic equivalence.  
    * Determines the SGF move made between two board states.  
* **Video Capture & Calibration:**  
    * Probes connected video devices to list available webcams and their capabilities.  
    * Captures snapshots (images) from a specified webcam.  
    * Provides an interactive calibration mode to define the board area in the webcam view, saving these coordinates to `share/config.txt`. The `getBoardCorners` function in `image.cpp` will then use this configuration.  
    * Supports two capture backends: direct V4L2 calls and OpenCV's VideoCapture (which can also use V4L2).  
* **Command-Line Interface:**  
    * Flexible options to perform various tasks.  
    * Debug mode (`-d` or `--debug`) for verbose output.  
* **Error Handling:**  
    * Custom error classes (`GEMError`, `SGFError`) for more informative diagnostics.

## Building

To build GEM, you need:

* A C++ compiler supporting C++17 (e.g., g++)  
* OpenCV (version 4 or later) \- Development libraries (`libopencv-dev`)  
* libv4l-dev (for V4L2 support)

Use the following command to compile the source code:

```bash  
g++ \-g image.cpp sgf.cpp snapshot.cpp gem.cpp camera.cpp \-o gem.exe `pkg-config \--cflags \--libs opencv4` \-lv4l2

This command compiles all the C++ source files and links them with the necessary OpenCV and V4L2 libraries. The \-g flag includes debugging symbols.

## **Important: Permissions and the share Directory**

### **Webcam Access (V4L2)**

Operations involving direct webcam access via V4L2 (like probing devices, taking snapshots with the V4L2 backend, or calibration) require permission to access video device files (e.g., /dev/video0).

* **Running as root (sudo):** The simplest way is to run the application with sudo ./gem.exe .... However, this is generally not recommended for regular use due to security risks.  
* **Using the video group (Recommended):** A better approach is to add your user to the video group. This group typically has the necessary permissions to access video devices without needing root privileges for the application itself.  
  Bash  
  sudo usermod \-aG video $USER  
  You will need to **log out and log back in** for this change to take effect.

### **The share Directory**

Even when running as a non-root user who is a member of the video group, there's a nuance: the program, when accessing V4L2 devices, might operate under the effective group ID of video. This video group might not have write permissions in your user's home directory or the project's current working directory if it was created solely by your user.  
To ensure that GEM can save files like snapshot.jpg (from snapshot/calibration) and config.txt (from calibration) without permission errors, a share subdirectory is used. You need to create this directory and give it open permissions:

Bash

mkdir share  
chmod 777 share

This allows the program, even when its effective group for device access might be video, to write necessary files into the share folder. The calibration process specifically saves snapshot.jpg (or snapshot\_osd.jpg in debug mode) and config.txt into this share directory.

## **Usage**

./gem.exe \[options\]

### **Options:**

* \-d, \--debug: Enable debug output. Place this option first if used.  
* \-D, \--device \<device\_path\>: Specify the video device path (default: /dev/video0). Place this option early if used with device-dependent operations.  
* \-M, \--mode \<backend\>: Specify capture backend ('v4l2' or 'opencv', default: v4l2).  
* \-b, \--calibration: Run interactive board corner calibration using the webcam. Saves results to share/config.txt and a snapshot to share/snapshot.jpg.  
* \-p, \--process-image \<image\_path\>: Process the specified Go board image and display the result (if debug mode is active or explicitly coded).  
* \-g, \--generate-sgf \<input\_image\> \<output\_sgf\>: Process an image and generate an SGF file representing the board state.  
* \-v, \--verify \<image\_path\> \<sgf\_path\>: Overlay SGF data on the image to visually verify stone positions.  
* \-c, \--compare \<sgf\_path1\> \<sgf\_path2\>: Compare two SGF files for semantic equivalence.  
* \--parse \<sgf\_path\>: Parse an SGF file and print its header, setup stones, and moves.  
* \--probe-devices: List available video capture devices and their supported formats.  
* \-s, \--snapshot \<output\_file\_path\>: Capture a snapshot from the webcam and save it to the specified path (e.g., share/my\_snapshot.jpg).  
* \-r, \--record-sgf \<output\_sgf\_path\>: Capture a snapshot, process it, and generate an SGF file (e.g., share/recorded\_game.sgf).  
* \-t, \--test-perspective \<image\_path\>: (Development/Test) Test the perspective correction on an image.  
* \-h, \--help: Display the help message.

### **Examples:**

1. **Enable debug mode and specify a different camera for probing:**  
   Bash  
   ./gem.exe \-d \-D /dev/video1 \--probe-devices

2. **Run interactive calibration for /dev/video0 using OpenCV backend:**  
   Bash  
   ./gem.exe \-M opencv \-b

   *(Ensure the share directory exists and is writable as explained above.)*  
3. **Take a snapshot from /dev/video0 and save it to share/board\_setup.jpg:**  
   Bash  
   ./gem.exe \-s share/board\_setup.jpg

   *(Ensure you have permissions or are in the video group, and share is writable.)*  
4. **Process a local image:**  
   Bash  
   ./gem.exe \-p ./my\_go\_game.png

5. **Generate an SGF from an image:**  
   Bash  
   ./gem.exe \-g ./go\_board\_image.jpg ./output\_game.sgf

6. **Record a current board position from webcam to an SGF file:**  
   Bash  
   ./gem.exe \-r share/live\_game.sgf

## **Functionality Details**

* **Board Processing (processGoBoard in image.cpp):**  
  1. Applies perspective correction using corners defined in share/config.txt (if available and dimensions match) or default percentages.  
  2. Detects horizontal and vertical grid lines.  
  3. Finds intersection points.  
  4. Samples colors (in Lab space) around intersections.  
  5. Uses K-Means clustering (k=3) on sampled colors to identify black stones, white stones, and empty board points.  
  6. Classifies clusters based on their Lab characteristics.  
  7. Applies a post-processing filter to remove isolated white stones (often glare or reflections).  
* **Calibration (runInteractiveCalibration in camera.cpp):**  
  * Displays a live feed from the selected camera.  
  * Allows interactive adjustment of the four corner points of the Go board area using keyboard inputs (u/d for up/down, w/n for wider/narrower).  
  * Saves the adjusted corner coordinates (both pixel and percentage values, along with image dimensions) to share/config.txt.  
  * Saves a snapshot of the current frame to share/snapshot.jpg (or share/snapshot\_osd.jpg if bDebug is true).  
* **SGF Generation (generateSGF in sgf.cpp):** Creates an SGF file with AB (add black) and AW (add white) properties based on the detected board state.  
* **Device Probing (probeVideoDevices in snapshot.cpp):** Iterates through /dev/videoX devices, queries their capabilities (driver, card name, supported formats) using V4L2 ioctl calls.

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

