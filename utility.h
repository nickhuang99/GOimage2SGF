#ifndef UTILITY_H
#define UTILITY_H

#include <vector>
#include <opencv2/opencv.hpp>

std::pair<std::vector<double>, std::vector<double>> detectUniformGrid(const cv::Mat& image);
std::vector<cv::Point2f> findIntersections(const std::vector<double>& horizontal_lines, const std::vector<double>& vertical_lines);
void processGoBoard(const cv::Mat& image_bgr, cv::Mat& board_state, cv::Mat& board_with_stones);
#endif // UTILITY_H