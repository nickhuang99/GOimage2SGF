#!/bin/bash
#set -x
set -v
g++ -std=c++17 -g logger.cpp image.cpp sgf.cpp snapshot.cpp gem.cpp camera.cpp -o gem.exe `pkg-config --cflags --libs opencv4` -lv4l2
