# Go Environment Manager (GEM)

## Overview

The Go Environment Manager (GEM) is a command-line tool designed to process Go board images, manage SGF (Smart Game Format) files, and capture snapshots from webcams. It provides functionalities to:

* Process Go board images to detect the board grid and stone positions.
* Generate SGF files from Go board images.
* Verify Go board states against SGF files.
* Compare SGF files.
* Parse SGF files to extract game information.
* Probe available video capture devices.
* Capture snapshots from webcams.

## Features

* **Image Processing:** Detects the Go board grid and stones from images. Handles various image qualities and perspectives.
* **SGF Management:** Generates, verifies, compares, and parses SGF files, enabling interaction with Go game records.
* **Video Capture:** Probes video devices and captures snapshots, allowing for automated acquisition of Go board images.
* **Command-Line Interface:** Provides a flexible and powerful command-line interface for easy integration into scripts and workflows.
* **Error Handling:** Robust error handling with custom exceptions to provide informative error messages.
* **Debug Mode:** Optional debug mode for detailed output during development and troubleshooting.

## Functionality

GEM supports the following operations:

* `process-image`: Processes a Go board image to extract board state.
* `generate-sgf`: Generates an SGF file from a Go board image.
* `verify`: Verifies a Go board image against an SGF file.
* `compare`: Compares two SGF files.
* `parse`: Parses an SGF file to extract game data.
* `probe-devices`: Lists available video capture devices and their capabilities.
* `snapshot`: Captures a snapshot from a specified webcam.

## Building

To build GEM, you need to have:

* A C++ compiler (e.g., g++)
* OpenCV (version 4 or later)
* libv4l2

Use the following command to compile the source code:

```bash
g++ -g image.cpp sgf.cpp snapshot.cpp gem.cpp -o gem.exe `pkg-config --cflags --libs opencv4` -lv4l2