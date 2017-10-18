#ifndef FRAME_H
#define FRAME_H

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/line_descriptor.hpp>
// #include "Log.h"
// #include "Statistic.h"
// #include "GeometryR.h"
// #include "CameraIntrinsic.h"

#define PI 3.1415926535897932

class Frame {
public:
    Frame() = default;
    ~Frame() = default;
    Frame(const cv::Mat& image, double timeStamp = 0);
    Frame(const cv::Mat& image, const cv::Mat& R, const cv::Mat& t, double timeStamp = 0);
    Frame(const Frame& frame);
    void operator=(const Frame& frame);

public:
    cv::Mat mImg;
    std::vector<cv::KeyPoint> mKps;
    double mTimeStamp;
    cv::Mat mR;
    cv::Mat mt;

public:
    void ExtractFeaturePoint();
    void ExtractFastPointOnLevel(cv::Mat image, std::vector<cv::KeyPoint>& keypoints);
    void ExtractEdgePointOnLevel(cv::Mat image, std::vector<cv::KeyPoint>& keypoints);

    std::vector<float> GetGradientMagnitude(cv::Mat image, std::vector<cv::Point2f> pts);
    void ComputeResponses(cv::Mat image, std::vector<cv::KeyPoint>& keypoints, int FASTThres);
    std::vector<cv::KeyPoint> NMSMask(cv::Mat image, std::vector<cv::KeyPoint>& points, std::vector<bool>& ifGetNMS);

    cv::Mat GetDoubleSE3();
    cv::Mat GetTcwMat();

public: // for debug and display
    void ShowImage();

};

#endif