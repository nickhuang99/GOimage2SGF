# **Go Environment Manager (GEM) v1.0.0**

A command-line tool to analyze Go board positions from images or live webcams, manage SGF files, and assist in game recording, featuring a robust, adaptive board detection algorithm.

## **🚀 Quick Start: Hello World**

This guide will get you from a checkout to a successfully detected board in two commands.

### **Prerequisites**

* **Hardware**: A webcam.  
* **Software**: g++, pkg-config, libopencv4-dev, libv4l-dev.  
  sudo apt-get install build-essential libopencv-dev pkg-config libv4l-dev

### **Step 1: Build the Application**

Clone the repository and run the build script.  
./build.sh

This creates the gem.exe executable.

### **Step 2: Calibrate the Board**

This is the "Hello World" step. Place your Go board in front of the camera with **four stones** on the corner intersections:

* **Black** stones at the Top-Left (TL) and Bottom-Left (BL) corners.  
* **White** stones at the Top-Right (TR) and Bottom-Right (BR) corners.

Then, run the **auto-calibration** command:  
./gem.exe \-A

The application will automatically capture an image, detect the board and corner stones, sample the colors, and save a complete share/config.txt file. It will also save a raw snapshot to share/snapshot\_raw\_calibration.jpg.

### **Step 3: Verify the Result**

To see the board detection in action on the image you just took, run:  
./gem.exe \--image ./share/snapshot\_raw\_calibration.jpg \--detect-board

An OpenCV window will appear showing the captured image with the detected board outline drawn in green, confirming that the core feature of the project is working perfectly.

## **🌟 Features**

* **Robust Board Detection**:  
  * **Pass 1 (Rough Detection)**: Uses find\_board\_quadrilateral\_rough to find all plausible 4-sided shapes in the image, creating a list of board candidates.  
  * **Pass 2 (Adaptive Refinement)**: The core of the application's intelligence. For each candidate, it performs the following:  
    * **Primary** Method **(High-Quality Images)**: It first attempts a fast and high-precision detection using goodFeaturesToTrack to find grid intersections. A strict quality check ensures this method only succeeds on clear, high-contrast images.  
    * **Fallback Method (Challenging Images)**: If the primary method fails, it automatically switches to a robust fallback strategy.  
      * **Enhanced Pre-processing**: Uses **CLAHE** (Contrast Limited Adaptive Histogram Equalization) and **Bilateral Filtering** to dramatically improve local contrast and reduce noise in poorly lit images.  
      * **Masked Line Search**: Uses the rough shape from Pass 1 to create a **mask**, ensuring the Hough Line Transform only searches for grid lines *inside* the board, ignoring background noise.  
      * **K-Means Clustering**: The cleaned-up horizontal and vertical lines are clustered using cv::kmeans to find the 19 center points for each axis, providing a highly accurate and noise-resistant grid definition.  
* **Full Calibration Suite**:  
  * **Automatic Calibration (-A)**: The recommended one-shot method to get started.  
  * **Interactive Calibration (-B)**: A user-friendly GUI to manually adjust corners and save a configuration, ideal for tricky setups.  
* **SGF & Game Management**:  
  * **Tournament Mode (-t)**: A guided workflow to record a live game step-by-step, capturing images and appending moves to a master tournament.sgf file.  
  * **Study** Mode **(-u)**: A replay tool to step through a recorded game, showing the board state and the corresponding snapshot for each move.

## **🛠️ Building**

The project uses a simple build script and relies on pkg-config to find the necessary OpenCV 4 libraries.  
\# Ensure you have the required dependencies installed  
\# sudo apt-get install build-essential libopencv-dev pkg-config libv4l-dev

\# Run the build script from the project root  
./build.sh

## **⚙️ Command-Line Options**

### **Main Workflows**

| Flag | Description | Example |
| :---- | :---- | :---- |
| \-A, \--auto-calibration | **(Recommended First Step)** Captures from webcam, detects corners, and saves share/config.txt. | ./gem.exe \-A |
| \-B, \--interactive-calibration | Starts a GUI to manually adjust corners and save a configuration. | ./gem.exe \-B |
| \--detect-board | Detects the board in an image and displays the result. Requires \--image. | ./gem.exe \--image ./share/img.jpg \--detect-board |
| \-t, \--tournament | Starts the turn-by-turn game recording workflow. | ./gem.exe \-t \--game-name "MyGame" |
| \-u, \--study | Starts the turn-by-turn game replay workflow. | ./gem.exe \-u \--game-name "MyGame" |
| \-p \<path\>, \--process-image \<path\> | Full processing of an image (detects stones) based on existing config. | ./gem.exe \-p ./share/board\_state.jpg |

### **Utility Options**

| Flag | Description | Example |
| :---- | :---- | :---- |
| \-h, \--help | Displays the help message. | ./gem.exe \-h |
| \-d, \--debug | Enables detailed logging and intermediate debug image windows. | ./gem.exe \-A \-d |
| \-O \<level\>, \--log-level \<level\> | Sets log verbosity (0=NONE, 1=ERROR, 3=INFO, 4=DEBUG). | ./gem.exe \-A \-O 4 |
| \-s \<path\>, \--snapshot \<path\> | Captures a single image from the webcam to the specified path. | ./gem.exe \-s ./share/my\_snapshot.jpg |
| \--image \<path\> | Specifies the input image for workflows like \--detect-board. | ./gem.exe \--image file.jpg \--detect-board |
| \-D \<path\>, \--device \<path\> | Specifies the camera device path (e.g., /dev/video1). | ./gem.exe \-B \-D /dev/video1 |
| \--size \<WxH\> | Specifies camera resolution (e.g., 1280x720). | ./gem.exe \-B \--size 1280x720 |

## **🤝 Contributing**

Contributions are welcome\! Please follow these general steps:

1. Fork the repository.  
2. Create a new branch for your feature or bug fix.  
3. Implement your changes.  
4. Test thoroughly, including building and running with various options.  
5. Update README.md if your changes affect usage, building, or add features.  
6. Submit a pull request with a clear description of your changes.

## **📜 License**

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
