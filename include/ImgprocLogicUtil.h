#ifndef GATES_LOCATOR_IMGPROCUTIL_LOGIC_H
#define GATES_LOCATOR_IMGPROCUTIL_LOGIC_H

#include <vector>
#include <opencv2/opencv.hpp>


void show(const char* name, const cv::Mat &image);

void linearChannelRescale(const cv::Mat &src, cv::Mat& dst, int maxIntensity);

void closing(const cv::Mat &image, cv::Mat& dst);

void gpuMeanShift(const cv::Mat& src, cv::Mat& dst);

void cpuMeanShift(const cv::Mat& src, cv::Mat& dst);

void blur(const cv::Mat& src, cv::Mat& dst);

void bilateral(const cv::Mat&, cv::Mat& dst);

void canny(const cv::Mat& src, cv::Mat& dst);

void extractValueChannel(const cv::Mat& src, cv::Mat& dst);

void morphology(const cv::Mat& src, cv::Mat& dst);


#endif //GATES_LOCATOR_IMGPROCUTIL_LOGIC_H
