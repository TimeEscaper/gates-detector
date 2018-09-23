#ifndef GATES_LOCATOR_IMGPROC_LOGIC_H
#define GATES_LOCATOR_IMGPROC_LOGIC_H

#include <vector>
#include <opencv2/opencv.hpp>


void linearChannelRescale(const cv::Mat &src, cv::Mat& dst, int maxIntensity);

void closing(const cv::Mat &image, cv::Mat& dst);

void gpuMeanShift(const cv::Mat& src, cv::Mat& dst);

void cpuMeanShift(const cv::Mat& src, cv::Mat& dst);

void extractValueChannel(const cv::Mat& src, cv::Mat& dst);

void extractHueChannel(const cv::Mat& src, cv::Mat& dst);

void extractLinesSolid(const cv::Mat& src, cv::Mat& dst);

void extractVerticalLines(const cv::Mat& src, cv::Mat& dst);

void detectLines(const cv::Mat& src, std::vector<cv::Vec4f>& lines);

float getLineSlope(const cv::Vec4f& line);

#endif //GATES_LOCATOR_IMGPROC_LOGIC_H