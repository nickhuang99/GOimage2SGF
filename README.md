\# Go Environment Manager (GEM)

\#\# Overview

The Go Environment Manager (GEM) is a command-line tool designed to process Go board images, manage SGF (Smart Game Format) files, and capture snapshots from webcams. It provides functionalities to:

\* Process Go board images to detect the board grid and stone positions.  
\* Generate SGF files from Go board images.  
\* Verify Go board states against SGF files.  
\* Compare SGF files.  
\* Parse SGF files to extract game information.  
\* Probe available video capture devices.  
\* Capture snapshots from webcams.

\#\# Features

\* \*\*Image Processing:\*\* Detects the Go board grid and stones from images. Handles various image qualities and perspectives.  
\* \*\*SGF Management:\*\* Generates, verifies, compares, and parses SGF files, enabling interaction with Go game records.  
\* \*\*Video Capture:\*\* Probes video devices and captures snapshots, allowing for automated acquisition of Go board images.  
\* \*\*Command-Line Interface:\*\* Provides a flexible and powerful command-line interface for easy integration into scripts and workflows.  
\* \*\*Error Handling:\*\* Robust error handling with custom exceptions to provide informative error messages.  
\* \*\*Debug Mode:\*\* Optional debug mode for detailed output during development and troubleshooting.

\#\# Functionality

GEM supports the following operations:

\* \`process-image\`: Processes a Go board image to extract board state.  
\* \`generate-sgf\`: Generates an SGF file from a Go board image.  
\* \`verify\`: Verifies a Go board image against an SGF file.  
\* \`compare\`: Compares two SGF files.  
\* \`parse\`: Parses an SGF file to extract game data.  
\* \`probe-devices\`: Lists available video devices. Requires root privileges.  
\* \`snapshot\`: Captures a snapshot from a specified webcam.

\#\# Building

To build GEM, you need to have:

\* A C++ compiler (e.g., g++)  
\* OpenCV (version 4 or later)  
\* libv4l2

Use the following command to compile the source code:

\`\`\`bash  
g++ \-g image.cpp sgf.cpp snapshot.cpp gem.cpp \-o gem.exe \`pkg-config \--cflags \--libs opencv4\` \-lv4l2

This command compiles all the source files (image.cpp, sgf.cpp, snapshot.cpp, gem.cpp) and links them with the necessary OpenCV and libv4l2 libraries.

## **Usage**

gem \[options\]

### **Options**

* \-d, \--debug: Enable debug output. Must be at the beginning of the options.  
* \-D, \--device \<device\_path\>: Specify the video device path (e.g., /dev/video0). Must be at the beginning for probe-devices to use it.  
* \-p, \--process-image \<image\_path\>: Process the Go board image.  
* \-g, \--generate-sgf \<input\_image\> \<output\_sgf\>: Generate SGF from image.  
* \-v, \--verify \<image\_path\> \<sgf\_path\>: Verify board state against SGF.  
* \-c, \--compare \<sgf\_path1\> \<sgf\_path2\>: Compare two SGF files.  
* \--parse \<sgf\_path\>: Parse an SGF file.  
* \--probe-devices: List available video devices. Requires root privileges.  
* \-s, \--snapshot \<output\_file\>: Capture a snapshot from the webcam. Requires root privileges.  
* \-h, \--help: Display this help message.

**Important Notes:**

* The \-d/--debug and \-D/--device options should be placed at the beginning of the command-line arguments.  
* Snapshot operations (--probe-devices, \-s/--snapshot) often require root privileges. Use sudo to run the program.

### **Examples**

* Process a Go board image:  
  ./gem.exe \-p board.jpg

* Generate an SGF file:  
  ./gem.exe \-g board.jpg game.sgf

* Capture a snapshot from the default webcam:  
  sudo ./gem.exe \-s snapshot.jpg

* Capture a snapshot from a specific webcam:  
  sudo ./gem.exe \-D /dev/video1 \-s snapshot.jpg

* List available video devices:  
  sudo ./gem.exe \--probe-devices

## **Testing**

After building, you can test the functionality by running the executable with the appropriate options. Ensure you have sample Go board images and SGF files available for testing the image processing and SGF management features. For video capture testing, ensure you have a working webcam connected.

## **Contributing**

Contributions to GEM are welcome\! Please follow these steps:

1. Fork the repository.  
2. Create a new branch for your feature or bug fix.  
3. Implement your changes.  
4. Test thoroughly.  
5. Submit a pull request.

## **License**

MIT License  
Copyright (c) \[Year\] \[Your Name\]  
Permission is hereby granted, free of charge, to any person obtaining a copy  
of this software and associated documentation files (the "Software"), to deal  
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell  
copies of the Software, and to permit persons to whom the Software is  
furnished to do so, subject to the following conditions:  
The above copyright notice and this permission notice shall be included in all  
copies or substantial portions of the Software.  
THE SOFTWARE IS PROVIDED "AS