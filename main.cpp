#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include "include/ImgprocLogic.h"
#include "include/ImgprocPipeline.h"
#include "include/Utils.h"


#define EXAMPLE_IMAGE "/home/sibirsky/gates_locator_images/gates.jpg"

int main() {

    cv::Mat src = cv::imread(EXAMPLE_IMAGE);
    show("Source", src);

#ifndef OLD

    cv::Mat image = createPipeline(src)
            .apply(gpuMeanShift, "MeanShift")
            .apply(extractValueChannel, "Value channel")
                    //.apply(extractLinesSolid, "Draw solid lines")
            .apply(extractVerticalLines, "Draw vertical lines")
            //.apply(morphology, "Closing")
            .getImage();

    /**
    cv::Mat image = createPipeline
     (src)
            .apply(blur)
            //.apply(bilateral, "Filter")
            //.apply(gpuMeanShift, "MeanShift")
            .apply(extractHueChannel, "Hue channel")
            //.apply(bilateral, "Bilateral")
            .apply(drawContours, "Contours")
            //.apply(blur, "Blur")
            //.apply(blur, "Blur 2")
            //.apply([](const cv::Mat& src, cv::Mat& dst) -> void {
                //linearChannelRescale(src, dst, 360);
            //})
                    //.apply(canny)
                    //.apply(extractLinesSolid, "Draw solid lines")
            //.apply(extractVerticalLines, "Draw vertical lines")
            .getImage();
            */

    /**
    cv::Mat image = createPipeline(src)
            //.apply(blur)
            //.apply(bilateral, "Filter")
            //.apply(gpuMeanShift, "MeanShift")
            .apply(extractHueChannel, "Value channel")
            .apply([](const cv::Mat& src, cv::Mat& dst) -> void {
                linearChannelRescale(src, dst, 360);
            })
            //.apply(canny)
            //.apply(extractLinesSolid, "Draw solid lines")
            .apply(extractVerticalLines, "Draw vertical lines")
            .getImage();
            */

    show("Lines", image);

#else

    cv::Mat image = createPipeline(src)
            .apply(extractHueChannel)
            .apply([](const cv::Mat& src, cv::Mat& dst) -> void {
                linearChannelRescale(src, dst, 360);
            })
            .apply([](const cv::Mat& src, cv::Mat& dst) -> void {
                cv::fastNlMeansDenoising(src, dst);
            })
            .apply([](const cv::Mat& src, cv::Mat& dst) -> void {
                cv::adaptiveThreshold(src, dst, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 51, 1);
            })
            .getImage();


    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(image, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
    cv::drawContours(src, contours, -1, cv::Scalar(0, 0, 255), 1, 8, hierarchy);

    std::vector<std::vector<cv::Point>> hulls(contours.size());
    double maxArea = 0.0;
    int maxAreaIndex = 0;
    for (int i = 0; i < contours.size(); i++) {
        cv::convexHull(contours[i], hulls[i]);
        double area = cv::contourArea(hulls[i]);
        if (area > maxArea) {
            maxArea = area;
            maxAreaIndex = i;
        }
    }
    cv::drawContours(src, hulls, maxAreaIndex, cv::Scalar(255, 0, 0), 1, 8, hierarchy);
    show("Hulls", src);

    cv::Moments moments = cv::moments(hulls[maxAreaIndex], false);
    cv::Point2f massCenter = cv::Point2f(moments.m10/moments.m00, moments.m01/moments.m00);

    cv::circle(src, massCenter, 10, cv::Scalar(0, 255, 0), 3);
    show("Center", src);

#endif

    return 0;
}