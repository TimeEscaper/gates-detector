#include "../include/ImgprocLogic.h"

#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>


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

void extractValueChannel(const cv::Mat& src, cv::Mat& dst) {
    cv::Mat hsv;
    cv::cvtColor(src, hsv, CV_BGR2HSV);
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsv, hsvChannels);
    dst = hsvChannels[2];
}

void extractHueChannel(const cv::Mat& src, cv::Mat& dst) {
    cv::Mat hsv;
    cv::cvtColor(src, hsv, CV_BGR2HSV);
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsv, hsvChannels);
    dst = hsvChannels[0];
}

void extractLinesSolid(const cv::Mat& src, cv::Mat& dst) {

    cv::Ptr<cv::LineSegmentDetector> detector = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
    std::vector<cv::Vec4f> lines;
    detector->detect(src, lines);

    dst = cv::Mat(src.rows, src.cols, CV_8UC1);
    dst.setTo(cv::Scalar(255));

    for (int i = 0; i < lines.size(); i++) {
        cv::line(dst, cv::Point2f(lines[i][0], lines[i][1]), cv::Point2f(lines[i][2], lines[i][3]),
                 cv::Scalar(0, 0, 255), 11);
    }
}

void extractVerticalLines(const cv::Mat& src, cv::Mat& dst) {

    cv::Ptr<cv::LineSegmentDetector> detector = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
    std::vector<cv::Vec4f> lines;
    detector->detect(src, lines);

    dst = cv::Mat(src.rows, src.cols, CV_8UC1);
    dst.setTo(cv::Scalar(255));

    std::vector<cv::Vec4f> filteredLines;
    for (int i = 0; i < lines.size(); i++) {
        if (getLineSlope(lines[i]) < 10.0f)
            filteredLines.push_back(lines[i]);
    }

    detector->drawSegments(dst, filteredLines);
}

void detectLines(const cv::Mat& src, std::vector<cv::Vec4f>& lines) {
    cv::Ptr<cv::LineSegmentDetector> detector = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
    detector->detect(src, lines);
}

float getLineSlope(const cv::Vec4f& line) {
    if (line[1] == line[3])
        return INFINITY;
    float tan = std::abs(line[0] - line[2]) / std::abs(line[1] - line[3]);
    float arctan = std::atan(tan);
    float res = arctan * 180.0f / CV_PI;
    return res;
}