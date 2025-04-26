#ifndef UTILITY_H
#define UTILITY_H

#include <opencv2/opencv.hpp>
#include <linux/videodev2.h>
#include <set>
#include <string>
#include <vector>
#include <map>

struct SGFHeader {
  int gm;         // Game
  int ff;         // File Format
  std::string ca; // Character Set
  std::string ap; // Application
  int sz;         // Size of the board
};


// Structure to hold video device information
struct VideoDeviceInfo {
  std::string device_path;
  std::string driver_name;
  std::string card_name;
  uint32_t capabilities;
  std::vector<uint32_t> supported_formats;
};

// Structure to represent a single move, including captured stones
struct Move {
  int player; // 1 for Black, 2 for White, 0 for remove
  int row;
  int col;
  std::set<std::pair<int, int>>
      capturedStones; // Coordinates of captured stones

  // Define the equality operator for Move objects.
  bool operator==(const Move &other) const {
    return (player == other.player && row == other.row && col == other.col &&
            capturedStones == other.capturedStones);
  }
};
std::string getFormatDescription(uint32_t format);
std::string getCapabilityDescription(uint32_t cap);
std::pair<std::vector<double>, std::vector<double>>
detectUniformGrid(const cv::Mat &image);
std::vector<cv::Point2f>
findIntersections(const std::vector<double> &horizontal_lines,
                  const std::vector<double> &vertical_lines);
void processGoBoard(const cv::Mat &image_bgr, cv::Mat &board_state,
                    cv::Mat &board_with_stones,
                    std::vector<cv::Point2f> &intersection_points);
std::string generateSGF(const cv::Mat &board_state,
                        const std::vector<cv::Point2f> &intersections);
std::string determineSGFMove(const cv::Mat &before_board_state,
                             const cv::Mat &next_board_state);
void verifySGF(const cv::Mat &image, const std::string &sgf_data,
               const std::vector<cv::Point2f> &intersections);
bool compareSGF(const std::string &sgf1, const std::string &sgf2);
void parseSGFGame(const std::string &sgfContent,
                  std::set<std::pair<int, int>> &setupBlack,
                  std::set<std::pair<int, int>> &setupWhite,
                  std::vector<Move> &moves);
SGFHeader parseSGFHeader(const std::string &sgf_content);
std::vector<VideoDeviceInfo> probeVideoDevices(int max_devices = 256);
bool captureSnapshot(const std::string& device_path, const std::string& output_path);
#endif // UTILITY_H