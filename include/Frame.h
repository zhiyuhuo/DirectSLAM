#ifndef DIRECTSLAM_FRAME_H
#define DIRECTSLAM_FRAME_H

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "Log.h"
#include "Statistic.h"

class Frame {
public:
    Frame() = default;
    ~Frame() = default;
    Frame(const cv::Mat& image, double timeStamp = 0);
    Frame(const Frame& frame);
    void operator=(const Frame& frame);

public:
    std::vector<cv::Mat> mImgPyr;
    std::vector<std::vector<cv::KeyPoint> > mKpsPyr;
    std::vector<std::vector<float> > mDepthPyr;
    std::vector<std::vector<Statistic> > mStatisticPyr;
    float mScaleFactor;
    double mTimeStamp;
    cv::Mat mR;
    cv::Mat mt;

public:
    void ExtractFeaturePoint();
    void ExtractFastPointOnLevel(cv::Mat image, std::vector<cv::KeyPoint>& keypoints, int level);
    void ExtractEdgePointOnLevel(cv::Mat image, std::vector<cv::KeyPoint>& keypoints, int level);

    std::vector<float> GetGradientMagnitude(cv::Mat image, std::vector<cv::Point2f> pts);
    void ComputeResponses(cv::Mat image, std::vector<cv::KeyPoint>& keypoints, int FASTThres);
    std::vector<cv::KeyPoint> NMSMask(cv::Mat image, std::vector<cv::KeyPoint>& points, std::vector<bool>& ifGetNMS);

    cv::Mat GetDoubleSE3();
    cv::Mat GetTcwMat();

public: // for debug and display
    void ShowPyr(int levelShow);

};

class Measure2D {
    cv::Point2f mKeypoint;
    float       mD;
    
};

#endif