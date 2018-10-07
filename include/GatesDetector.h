#ifndef GATES_LOCATOR_GATESDETECTOR_H
#define GATES_LOCATOR_GATESDETECTOR_H


#include "GatesDescriptor.h"

class GatesDetector {

private:

    void defaultPreprocess(const cv::Mat& src, cv::Mat& dst);

    void detectVerticalLines(const cv::Mat& image, std::vector<cv::Vec4f>& lines);

    float getLineSlope(const cv::Vec4f& line);

    float getLength(const cv::Vec4f& line);

    float getDistance(float x1, float y1, float x2, float y2);

    cv::Point2f getProjection(const cv::Vec4f& line, const cv::Point2f& point);

public:

    GatesDetector() = default;
    ~GatesDetector() = default;

    GatesDetector& operator=(const GatesDetector& other) = default;

    GatesDescriptor detect(const cv::Mat& src, bool withPreprocess);

};


#endif //GATES_LOCATOR_GATESDETECTOR_H
