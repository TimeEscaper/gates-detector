#ifndef GATES_LOCATOR_GATESDESCRIPTOR_H
#define GATES_LOCATOR_GATESDESCRIPTOR_H


#include <opencv2/opencv.hpp>

class GatesDescriptor {

private:

    bool gates;
    std::vector<cv::Point2f> corners;
    cv::Point2f center;
    cv::Rect boundingRect;

    GatesDescriptor(bool gates, const std::vector<cv::Point2f>& corners);

public:

    static GatesDescriptor noGates();
    static GatesDescriptor create(const std::vector<cv::Point2f>& corners);

    GatesDescriptor(const GatesDescriptor& other);
    ~GatesDescriptor() = default;

    GatesDescriptor& operator=(const GatesDescriptor& other);

    bool hasGates();
    std::vector<cv::Point2f> getCorners();
    cv::Point2f getCenter();
    cv::Rect getBoundingRect();

};


#endif //GATES_LOCATOR_GATESDESCRIPTOR_H
