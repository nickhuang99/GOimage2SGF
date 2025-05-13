#ifndef UTILITY_H
#define UTILITY_H

#include <linux/videodev2.h>
#include <map>
#include <opencv2/core/types.hpp> // For cv::Point2f
#include <opencv2/opencv.hpp>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <cstring>
#include <charconv>


extern bool bDebug;
extern int g_capture_width;
extern int g_capture_height;
extern std::string g_device_path;

extern const std::string CALIB_CONFIG_PATH;
extern const std::string CALIB_SNAPSHOT_PATH;
extern const std::string CALIB_SNAPSHOT_DEBUG_PATH;

#define WHITE 2
#define BLACK 1
#define EMPTY 0

// Custom exception class for GEM errors
class GEMError : public std::runtime_error {
public:
  GEMError(const std::string &message, const char *file, int line,
           const char *function)
      : std::runtime_error(constructMessage(message, file, line, function)) {}

private:
  static std::string constructMessage(const std::string &message,
                                      const char *file, int line,
                                      const char *function) {
    std::stringstream ss;
    ss << message << " in " << function << " at " << file << ":" << line;
    return ss.str();
  }
};
class Num2Str {
  std::string m_str;

public:
  template <typename T> Num2Str(T n) {
    std::stringstream ss;
    ss << n;
    m_str = ss.str();
  }
  operator std::string() { return m_str; }
  std::string str() { return m_str; }
};

class Str2Num {
  int value;
  bool valid;

public:
  explicit Str2Num(const char *data) {
    auto [ptr, ec] = std::from_chars(data, data + strlen(data), value);
    valid = ec == std::errc();
  }
  operator bool() { return valid; }
  int val() { return value; }
};

// Macro for throwing GEMError exceptions with source code information
#define THROWGEMERROR(message)                                                 \
  throw GEMError(message, __FILE__, __LINE__, __PRETTY_FUNCTION__);

class SGFError : public std::runtime_error {
public:
  SGFError(const std::string &message) : std::runtime_error(message) {}
};

// --- Capture Mode Selection ---
enum CaptureMode {
  MODE_V4L2,  // Use direct V4L2 calls
  MODE_OPENCV // Use OpenCV VideoCapture
};
extern CaptureMode gCaptureMode; // Defined and defaulted in gem.cpp

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
  // std::vector<uint32_t> supported_formats; // OLD
  std::vector<std::string>
      supported_format_details; // NEW: Will store "FORMAT (WxH, WxH...)"
};

// --- Struct to hold all calibration data ---
struct CalibrationData {
  std::vector<cv::Point2f> corners; // TL, TR, BR, BL order
  cv::Vec3f lab_tl = {-1.0f, -1.0f, -1.0f};
  cv::Vec3f lab_tr = {-1.0f, -1.0f, -1.0f};
  cv::Vec3f lab_bl = {-1.0f, -1.0f, -1.0f};
  cv::Vec3f lab_br = {-1.0f, -1.0f, -1.0f};
  cv::Vec3f lab_board_avg = {-1.0f, -1.0f, -1.0f}; 

  // --- MODIFIED/CLARIFIED ---
  std::string device_path;       // Device path string (e.g., "/dev/video0")
  int image_width = 0;           // Frame width at time of calibration
  int image_height = 0;          // Frame height at time of calibration

  bool corners_loaded = false;
  bool colors_loaded = false; 
  bool board_color_loaded = false; 
  bool dimensions_loaded = false;           // Covers image_width_at_calibration, image_height_at_calibration
  bool device_path_loaded = false;          // NEW flag for device_path_at_calibration
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

bool captureFrame(const std::string &device_path, cv::Mat &frame);

// // captureSnapshot now uses the selected mode via gCaptureMode
bool captureSnapshot(const std::string &device_path,
                     const std::string &output_path);
// Keep displayWebcamFeed declaration if it's in snapshot.cpp now

// void displayWebcamFeed(int camera_index); // Or runInteractiveCalibration
cv::Mat correctPerspective(const cv::Mat &image);

// Declare the function to display webcam feed (defined in snapshot.cpp)
void runInteractiveCalibration(int camera_index);

bool trySetCameraResolution(
    cv::VideoCapture &cap, int desired_width, int desired_height,
    bool attempt_fallback_format = true); // Default to attempt fallback

// Function to load only the corner coordinates from the config file
std::vector<cv::Point2f>
loadCornersFromConfigFile(const std::string &config_path);

// Declare utility function (defined in image.cpp)
cv::Vec3f getAverageLab(const cv::Mat &image_lab, cv::Point2f center,
                        int radius);
CalibrationData loadCalibrationData(const std::string &config_path);

int calculateAdaptiveSampleRadius(float board_pixel_width,
                                  float board_pixel_height);

std::vector<cv::Point2f> getBoardCornersCorrected(int width, int height);

#endif // UTILITY_H