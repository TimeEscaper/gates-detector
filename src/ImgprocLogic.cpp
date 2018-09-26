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

void blur(const cv::Mat& src, cv::Mat& dst) {
    cv::Mat mat = src;
    //cv::blur(src, dst, cv::Size(7, 7));
    double discard_ratio = 0.05;
    int hists[3][256];
    memset(hists, 0, 3*256*sizeof(int));

    for (int y = 0; y < mat.rows; ++y) {
        uchar* ptr = mat.ptr<uchar>(y);
        for (int x = 0; x < mat.cols; ++x) {
            for (int j = 0; j < 3; ++j) {
                hists[j][ptr[x * 3 + j]] += 1;
            }
        }
    }

    // cumulative hist
    int total = mat.cols*mat.rows;
    int vmin[3], vmax[3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 255; ++j) {
            hists[i][j + 1] += hists[i][j];
        }
        vmin[i] = 0;
        vmax[i] = 255;
        while (hists[i][vmin[i]] < discard_ratio * total)
            vmin[i] += 1;
        while (hists[i][vmax[i]] > (1 - discard_ratio) * total)
            vmax[i] -= 1;
        if (vmax[i] < 255 - 1)
            vmax[i] += 1;
    }


    for (int y = 0; y < mat.rows; ++y) {
        uchar* ptr = mat.ptr<uchar>(y);
        for (int x = 0; x < mat.cols; ++x) {
            for (int j = 0; j < 3; ++j) {
                int val = ptr[x * 3 + j];
                if (val < vmin[j])
                    val = vmin[j];
                if (val > vmax[j])
                    val = vmax[j];
                ptr[x * 3 + j] = static_cast<uchar>((val - vmin[j]) * 255.0 / (vmax[j] - vmin[j]));
            }
        }
    }

    dst = mat;
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

    cv::Mat buff = cv::Mat(src.rows, src.cols, CV_8UC1);
    buff.setTo(cv::Scalar(255));

    double maxLength = 0;
    std::vector<cv::Vec4f> filteredLines;
    for (int i = 0; i < lines.size(); i++) {
        if (getLineSlope(lines[i]) < 10.0f) {
            filteredLines.push_back(lines[i]);
            double length = getLength(lines[i]);
            if (length > maxLength)
                maxLength = length;
        }
    }

    std::vector<cv::Vec4f> lengthFilteredLines;
    for (int i = 0; i < filteredLines.size(); i++) {
        double length = getLength(filteredLines[i]);
        if (length >= 0.1*maxLength)
            lengthFilteredLines.push_back(filteredLines[i]);
    }


    detector->drawSegments(buff, lengthFilteredLines);

    std::vector<cv::Mat> channels;
    cv::split(buff, channels);

    dst = channels[0];
}

void detectLines(const cv::Mat& src, std::vector<cv::Vec4f>& lines) {
    cv::Ptr<cv::LineSegmentDetector> detector = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
    detector->detect(src, lines);
}

void drawContours(const cv::Mat& src, cv::Mat& dst) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours( src, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
    dst = cv::Mat::zeros(src.size(), CV_8UC3);
    for( size_t i = 0; i< contours.size(); i++ )
    {
        cv::drawContours( dst, contours, (int)i, cv::Scalar(255,0,0), 2, cv::LINE_8, hierarchy, 0 );
    }
}

void morphology(const cv::Mat& src, cv::Mat& dst) {
    int size = 1;
    cv::Mat element = cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(2*size+1, 2*size+1));
    cv::morphologyEx(src, dst, cv::MORPH_CLOSE, element);
}

float getLineSlope(const cv::Vec4f& line) {
    if (line[1] == line[3])
        return INFINITY;
    float tan = std::abs(line[0] - line[2]) / std::abs(line[1] - line[3]);
    float arctan = std::atan(tan);
    float res = arctan * 180.0f / CV_PI;
    return res;
}

float getLength(const cv::Vec4f& line) {
    return std::sqrt((line[2]-line[0])*(line[2]-line[0]) + (line[3]-line[1])*(line[3]-line[1]));
}