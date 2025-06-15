#ifndef UTILITY_H
#define UTILITY_H

#include "logger.h" // <<< NEW: Include logger header
#include <charconv>
#include <cstring>
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

extern bool bDebug;
extern int g_capture_width;
extern int g_capture_height;
extern std::string g_device_path;
extern std::string g_exp_quadrant_str;

extern const std::string CALIB_CONFIG_PATH;
extern const std::string CALIB_SNAPSHOT_PATH;
extern const std::string CALIB_SNAPSHOT_DEBUG_PATH;
extern const std::string CALIB_SNAPSHOT_RAW_PATH;

extern std::string g_default_input_image_path;

// NEW: Thresholds for detectSpecificColoredRoundShape
extern const int MORPH_OPEN_KERNEL_SIZE_STONE;
extern const int MORPH_OPEN_ITERATIONS_STONE;
extern const int MORPH_CLOSE_KERNEL_SIZE_STONE;
extern const int MORPH_CLOSE_ITERATIONS_STONE;
extern const double MIN_STONE_AREA_RATIO;
extern const double MAX_STONE_AREA_RATIO;
extern const double MIN_STONE_CIRCULARITY_WHITE;
extern const double MIN_STONE_CIRCULARITY_BLACK;
extern const int MIN_CONTOUR_POINTS_STONE;

extern float G_RAW_SEARCH_L_SEPARATOR;
extern double G_ROUGH_RAW_AREA_MIN_FACTOR;
extern double G_ROUGH_RAW_AREA_MAX_FACTOR;
extern double G_MIN_ROUGH_RAW_CIRCULARITY;

#define WHITE 2
#define BLACK 1
#define EMPTY 0

extern const float CALIB_L_TOLERANCE_STONE;
extern const float CALIB_AB_TOLERANCE_STONE;

// NEW Enum for specifying corner quadrant
enum class CornerQuadrant {
  TOP_LEFT = 0,
  TOP_RIGHT,
  BOTTOM_RIGHT,
  BOTTOM_LEFT,
};

constexpr const char *toString(CornerQuadrant quadrant) {
  switch (quadrant) {
  case CornerQuadrant::TOP_LEFT:
    return "TOP_LEFT";
  case CornerQuadrant::TOP_RIGHT:
    return "TOP_RIGHT";
  case CornerQuadrant::BOTTOM_LEFT:
    return "BOTTOM_LEFT";
  case CornerQuadrant::BOTTOM_RIGHT:
    return "BOTTOM_RIGHT";
  default:
    return "UNKNOWN"; // 处理未定义值（可选）
  }
}

std::ostream &operator<<(std::ostream &os, CornerQuadrant quadrant);

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
  std::string device_path; // Device path string (e.g., "/dev/video0")
  int image_width = 0;     // Frame width at time of calibration
  int image_height = 0;    // Frame height at time of calibration

  bool corners_loaded = false;
  bool colors_loaded = false;
  bool board_color_loaded = false;
  bool dimensions_loaded =
      false; // Covers image_width_at_calibration, image_height_at_calibration
  bool device_path_loaded = false; // NEW flag for device_path_at_calibration
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

struct CandidateBlob {
  cv::Point2f center_in_roi_coords; // Center within the ROI it was found
  double area;
  double circularity;
  float l_base_used;
  float l_tolerance_used;
  double score; // Can be used to reflect confidence or just mark as found
  std::vector<cv::Point> contour_points_in_roi; // Relative to ROI
  cv::Vec3f sampled_lab_color_from_contour;     // Lab color sampled from this
                                                // specific blob
  int classified_color_after_shape_found;       // BLACK, WHITE, EMPTY/OTHER
  cv::Rect roi_used_in_search;                  // <<-- ADD THIS LINE

  CandidateBlob()
      : area(0), circularity(0), l_base_used(0), l_tolerance_used(0),
        score(-1.0), sampled_lab_color_from_contour(-1, -1, -1),
        classified_color_after_shape_found(EMPTY) {}

  bool isValid() const {
    return area > 0 && score >= 0;
  } // Valid if area is positive and score indicates found
};

cv::Vec3f get_rough_board_lab_color(const cv::Mat &raw_image_bgr);

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

bool validateSGgfMove(const cv::Mat &before_board_state,
                      const cv::Mat &next_board_state, int preColor);

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
void runCaptureCalibration();

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

void drawSimulatedGoBoard(
    const std::string
        &full_tournament_sgf_content, // Entire game history from tournament.sgf
    int display_up_to_move_idx, // Display state AFTER this many B/W moves (0
                                // for setup, 1 after 1st B/W move, etc.)
    cv::Mat &output_image,
    int highlight_this_move_idx = -1, // Move index (0 for setup, 1-based for
                                      // B/W moves) to highlight, -1 for none
    int canvas_size_px = 760          // Default canvas size
);
bool processAndSaveCalibration(
    const cv::Mat &final_raw_bgr_for_snapshot,
    const std::vector<cv::Point2f> &current_source_points,
    bool enhanced_detection_was_successful,
    const std::vector<cv::Vec3f>
        *enhanced_lab_colors, // Pointer to vector of 4 cv::Vec3f
    float enhanced_avg_radius_px);

bool detectFourCornersGoBoard(
    const cv::Mat &input_image,
    std::vector<cv::Point2f> &detected_corners_tl_tr_br_bl);

bool detectColoredRoundShape(
    const cv::Mat &inputImage, // BGR image
    const cv::Rect &regionOfInterest,
    int expectedColor, // Use BLACK or WHITE macros
    cv::Point2f &detectedCenter, float &detectedRadius,
    // Optional: Pass calibration data if color ranges should come from there
    const CalibrationData *calibData = nullptr);

bool saveCornerConfig(
    const std::string &filename, const std::string &device_path_for_config,
    int frame_width, int frame_height, const cv::Point2f &tl_raw,
    const cv::Point2f &tr_raw, const cv::Point2f &bl_raw,
    const cv::Point2f &br_raw,
    const cv::Vec3f
        &lab_tl_sampled_corrected, // Original sampling from corrected
    const cv::Vec3f &lab_tr_sampled_corrected,
    const cv::Vec3f &lab_bl_sampled_corrected,
    const cv::Vec3f &lab_br_sampled_corrected,
    const cv::Vec3f &avg_lab_board_sampled,
    // --- New parameters for enhanced detection data ---
    bool enhanced_data_available,
    const std::vector<cv::Vec3f>
        *lab_corners_sampled_raw_enhanced, // Pointer to allow nullptr if not
                                           // available
    float detected_avg_stone_radius_raw    // Or individual radii
);

// --- NEW Function Declarations ---
/**
 * @brief Calculates the Region of Interest (ROI) for a given grid intersection
 * on a perspective-corrected board.
 *
 * @param target_col The target column index (0-18).
 * @param target_row The target row index (0-18).
 * @param corrected_image_width_px The full width of the perspective-corrected
 * image.
 * @param corrected_image_height_px The full height of the perspective-corrected
 * image.
 * @param grid_lines The number of lines on the board (e.g., 19 for a standard
 * Go board).
 * @return cv::Rect The calculated ROI.
 */
cv::Rect calculateGridIntersectionROI(int target_col, int target_row,
                                      int corrected_image_width_px,
                                      int corrected_image_height_px,
                                      int grid_lines = 19);

/**
 * @brief Detects the presence and color of a stone at a specific grid position
 * on a perspective-corrected board.
 *
 * @param corrected_bgr_image The perspective-corrected BGR image of the Go
 * board.
 * @param target_col The target column index (0-18) to check.
 * @param target_row The target row index (0-18) to check.
 * @param calib_data Loaded calibration data containing color references.
 * @return int The detected stone color (BLACK, WHITE, or EMPTY).
 */
int detectStoneAtPosition(const cv::Mat &corrected_bgr_image, int target_col,
                          int target_row, const CalibrationData &calib_data);

bool detectSpecificColoredRoundShape(const cv::Mat &inputBgrImage,
                                     const cv::Rect &regionOfInterest,
                                     const cv::Vec3f &expectedAvgLabColor,
                                     float l_tolerance, float ab_tolerance,
                                     float expectedPixelRadius,
                                     cv::Point2f &detectedCenter,
                                     float &detectedRadius);

// Signature for experimental function (V5: fixed guess, two-pass detection with
// iterative refinement of perspective)
bool experimental_scan_for_quadrant_stone( // Name kept, but logic will be V5
    const cv::Mat &rawBgrImage, CornerQuadrant targetScanQuadrant,
    const CalibrationData &calibData,
    cv::Point2f &out_final_raw_corner_guess, // The raw corner guess that led to
                                             // the successful PASS 2 warp
    cv::Mat &out_final_corrected_image,      // The image from the PASS 2 warp
    cv::Point2f &out_detected_stone_center_in_final_corrected,
    float &out_detected_stone_radius_in_final_corrected,
    cv::Rect &out_focused_roi_in_final_corrected);

std::vector<cv::Point2f> getBoardCorners(const cv::Mat &inputImage);

bool find_largest_color_blob_in_roi(
    const cv::Mat &image_to_search_bgr, const cv::Rect &roi_in_image,
    const cv::Vec3f &target_lab_color, float l_tol, float ab_tol,
    // Outputs:
    cv::Point2f
        &out_blob_center_in_image_coords, // Relative to image_to_search_bgr
    double &out_blob_area);

void runAutoCalibrationWorkflow();
bool verifyCalibrationBeforeSave(const CalibrationData &calibData,
                                 const cv::Mat &image_to_verify);
bool sampleCalibrationColors(const cv::Mat &raw_image,
                             CalibrationData &calibData);
void saveCalibrationData(const CalibrationData &data,
                         const std::string &config_path);
bool verifyCalibrationAfterSave(const cv::Mat &raw_image_for_verification);

bool detectCalibratedBoardState(const cv::Mat &rawBgrImage,
                                CalibrationData &out_calib_data);

bool find_blob_candidates_in_raw_quadrant(
    const cv::Mat &raw_image_bgr, CornerQuadrant quadrant,
    int expected_stone_color, const CalibrationData &calibData,
    std::vector<CandidateBlob> &out_candidate_blobs);
#endif // UTILITY_H