#include "../include/ImgprocLogicUtil.h"

#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>


void show(const char* name, const cv::Mat &image) {
    cv::namedWindow(name);
    cv::imshow(name, image);
    cv::waitKey();
    cv::destroyWindow(name);
}

void linearChannelRescale(const cv::Mat &src, cv::Mat& dst, int maxIntensity) {

    assert(src.type() == CV_8UC1);

    double max, min;
    cv::minMaxLoc(src, &min, &max);
    double diff = max - min;

    dst = cv::Mat(src.rows, src.cols, src.type());

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            dst.at<uchar>(i, j) = ((src.at<uchar>(i, j) - min) / diff) * maxIntensity;
        }
    }
}

void closing(const cv::Mat &src, cv::Mat& dst) {
    cv::Mat structElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5), cv::Point(2, 2));
    cv::morphologyEx(src, dst, cv::MORPH_CLOSE, structElement, cv::Point(-1,-1), 1);
}

void gpuMeanShift(const cv::Mat& src, cv::Mat& dst) {
    cv::cuda::GpuMat gpuSrc, gpuDest;
    gpuSrc.upload(src);
    cv::cuda::cvtColor(gpuSrc, gpuSrc, CV_BGR2BGRA);
    cv::cuda::meanShiftFiltering(gpuSrc, gpuDest, 30, 5);
    gpuDest.download(dst);
}

void cpuMeanShift(const cv::Mat& src, cv::Mat& dst) {
    cv::pyrMeanShiftFiltering(src, dst, 30, 5);
}

void blur(const cv::Mat& src, cv::Mat& dst) {
    cv::blur(src, dst, cv::Size(7, 7));
}

void bilateral(const cv::Mat& src, cv::Mat& dst) {
    cv::bilateralFilter(src, dst, 9, 75, 75);
    cv::Mat buff = src;
    /**
    for (int i = 0; i < 3; i++) {
        cv::bilateralFilter(buff, dst, 9, 75, 75);
        cv::bilateralFilter(dst, buff, 9, 75, 75);
    }
     */
}

void canny(const cv::Mat& src, cv::Mat& dst) {
    cv::Canny(src, dst, 155, 230);
}

void extractValueChannel(const cv::Mat& src, cv::Mat& dst) {
    cv::Mat hsv;
    cv::cvtColor(src, hsv, CV_BGR2HSV);
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsv, hsvChannels);
    dst = hsvChannels[2];
}

void morphology(const cv::Mat& src, cv::Mat& dst) {
    int size = 1;
    cv::Mat element = cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(2*size+1, 2*size+1));
    cv::morphologyEx(src, dst, cv::MORPH_CLOSE, element);
}