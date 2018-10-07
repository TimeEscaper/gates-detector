#include "../include/GatesDescriptor.h"


GatesDescriptor GatesDescriptor::noGates() { return GatesDescriptor(false, std::vector<cv::Point2f>(4)); }

GatesDescriptor GatesDescriptor::create(const std::vector<cv::Point2f>& corners) { return GatesDescriptor(true, corners); }

GatesDescriptor::GatesDescriptor(bool gates, const std::vector<cv::Point2f>& corners) {
    this->gates = gates;
    if (gates) {
        this->corners = corners;

        cv::Moments moments = cv::moments(corners, false);
        this->center = cv::Point2f(moments.m10 / moments.m00, moments.m01 / moments.m00);

        this->boundingRect = cv::boundingRect(corners);
    }
}

GatesDescriptor::GatesDescriptor(const GatesDescriptor &other) {
    this->gates = other.gates;
    this->corners = other.corners;
    this->center = other.center;
    this->boundingRect = other.boundingRect;
}

GatesDescriptor& GatesDescriptor::operator=(const GatesDescriptor &other) {
    if (this != &other) {
        this->gates = other.gates;
        this->corners = other.corners;
        this->center = other.center;
        this->boundingRect = other.boundingRect;
    }
    return *this;
}

bool GatesDescriptor::hasGates() { return gates; }

std::vector<cv::Point2f> GatesDescriptor::getCorners() { return corners; }

cv::Point2f GatesDescriptor::getCenter() { return center; }

cv::Rect GatesDescriptor::getBoundingRect() { return boundingRect; }
