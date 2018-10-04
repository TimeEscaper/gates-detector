#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include "include/ImgprocLogic.h"
#include "include/ImgprocPipeline.h"
#include "include/Utils.h"


#define EXAMPLE_IMAGE "/home/sibirsky/gates_locator_images/frame1-1002.jpg"

#define TEMPLATE "/home/sibirsky/gates_locator_images/template6_left.png"

int main() {

    cv::Mat src = cv::imread(EXAMPLE_IMAGE);
    show("Source", src);

#ifndef OLD


    /**
    cv::Mat templ = cv::imread(TEMPLATE);
    cv::resize(templ, templ, cv::Size(0,0), 1.5, 1.5);
    cv::Mat canvas = src;

    //int resultCols = src.cols - templ.cols + 1;
    //int resultRows = src.rows - templ.rows + 1;

    //cv::Mat result(resultRows, resultCols, CV_32FC1);
    cv::Mat result;

    cv::matchTemplate(src, templ, result, CV_TM_CCORR_NORMED, cv::noArray());
    cv::normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

    double minValue, maxValue;
    cv::Point minLoc, maxLoc, matchLoc;

    cv::minMaxLoc(result, &minValue, &maxValue, &minLoc, &maxLoc, cv::Mat());
    matchLoc = maxLoc;

    cv::rectangle(canvas, matchLoc, cv::Point(matchLoc.x + templ.cols , matchLoc.y + templ.rows),
            cv::Scalar::all(0), 2, 8, 0);
    cv::rectangle(result, matchLoc, cv::Point(matchLoc.x + templ.cols , matchLoc.y + templ.rows),
            cv::Scalar::all(0), 2, 8, 0);

    show("Result", canvas);
    show("Result", result);
     */



    cv::Mat image = createPipeline(src)
            //.apply(blur, "Blur")
            .apply(gpuMeanShift, "MeanShift")
            .apply(extractValueChannel, "Value channel")
                    //.apply(extractLinesSolid, "Draw solid lines")
            //.apply(extractVerticalLines, "Draw vertical lines")
            .apply(morphology, "Closing")
            .getImage();

    std::vector<cv::Vec4f> verticalLines;
    detectVerticalLines(image, verticalLines);
    //detectHorizontalLines(image, verticalLines);

    std::vector<cv::Point2f> allPoints;
    for (int i = 0; i < verticalLines.size(); i++) {
        allPoints.push_back(cv::Point2f(verticalLines[i][0], verticalLines[i][1]));
        allPoints.push_back(cv::Point2f(verticalLines[i][2], verticalLines[i][3]));
        cv::circle(src, cv::Point2f(verticalLines[i][0], verticalLines[i][1]), 5, cv::Scalar(255,0,0), 2);
        cv::circle(src, cv::Point2f(verticalLines[i][2], verticalLines[i][3]), 5, cv::Scalar(255,0,0), 2);
    }
    show("Points", src);

    std::sort(allPoints.begin(), allPoints.end(),
            [](const cv::Point2f& a, const cv::Point2f& b) -> bool { return a.x > b.x; });

    cv::Point2f currentPoint = allPoints[0];
    std::vector<std::vector<cv::Point2f>> pointLines;
    std::vector<cv::Point2f> currentPointLine;
    currentPointLine.push_back(currentPoint);

    for (int i = 1; i < allPoints.size(); i++) {

        if (std::abs(allPoints[i].x - currentPoint.x) > 14.0f) {
            pointLines.push_back(currentPointLine);
            currentPointLine.clear();
        }
        currentPointLine.push_back(allPoints[i]);

        currentPoint = allPoints[i];
    }
    pointLines.push_back(currentPointLine);

    std::vector<cv::Vec4f> mergedLines;
    for (int i = 0; i < pointLines.size(); i++) {
        // TODO: Use line fitting
        auto edges  = std::minmax_element(pointLines[i].begin(), pointLines[i].end(),
                [](const cv::Point2f& a, const cv::Point2f& b) {
                    return a.y > b.y;
        });

        mergedLines.push_back({(*(edges.first)).x, (*(edges.first)).y, (*(edges.second)).x, (*(edges.second)).y});
    }

    /**
    for (int i = 0; i < mergedLines.size(); i++) {
        cv::line(src, cv::Point2f(mergedLines[i][0], mergedLines[i][1]), cv::Point2f(mergedLines[i][2], mergedLines[i][3]),
                 cv::Scalar(0, 0, 255), 11);
    }
    show("Merged lines", src);
     */

    std::sort(mergedLines.begin(), mergedLines.end(), [](const cv::Vec4f& a, const cv::Vec4f& b) {
        return getLength(a) > getLength(b);
    });

    cv::Vec4f line1 = mergedLines[0]; // Longest line
    cv::Vec4f line2 = mergedLines[1]; // Second longest line

    float verticalRelation = getLength(line2) / getLength(line1);
    float sidesRelation = getDistance(line1[0], line1[1], line2[0], line2[1]) / getLength(line2);
    if (verticalRelation < 0.4f || sidesRelation < 0.4f || sidesRelation > 1.4f)
        return 0;

    // TODO: find point projections
    cv::Point2f topLeft(std::min(line1[2], line2[2]), std::min(line1[3], line2[3]));
    cv::Point2f bottomRight(std::max(line1[0], line2[0]), std::min(line1[1], line2[1]));

    cv::Rect gatesRect(topLeft, bottomRight);

    cv::rectangle(src, topLeft, bottomRight, cv::Scalar(0, 0, 255));
    show("Gates", src);


    /**
    std::vector<cv::Vec4f> filteredLinses;
    filterLinesByDistance(verticalLines, filteredLinses);

    float minX = INFINITY;
    float maxX = 0;
    float minY = INFINITY;
    float maxY = 0;
    for (int i = 0; i < filteredLinses.size(); i++) {
        float lineMinX = std::min(filteredLinses[i][0], filteredLinses[i][2]);
        float lineMaxX = std::max(filteredLinses[i][0], filteredLinses[i][2]);
        float lineMinY = std::min(filteredLinses[i][1], filteredLinses[i][3]);
        float lineMaxY = std::max(filteredLinses[i][1], filteredLinses[i][3]);

        minX = std::min(minX, lineMinX);
        maxX = std::max(maxX, lineMaxX);
        minY = std::min(minY, lineMinY);
        maxY = std::max(maxY, lineMaxY);
    }

    cv::Mat canvas = src;
    drawLines(canvas, filteredLinses);
    show("Filtered lines", canvas);

    cv::Rect roiRect(minX, minY, maxX - minX, maxY - minY);
    cv::Mat roi = src(roiRect);
    show("ROI", roi);

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

    //show("Lines", image);

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