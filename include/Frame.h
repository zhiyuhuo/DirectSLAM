#ifndef DIRECTSLAM_FRAME_H
#define DIRECTSLAM_FRAME_H

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "Log.h"

class Frame {
public:
    Frame() = default;
    ~Frame() = default;
    Frame(const cv::Mat& image, double timeStamp = 0);
    Frame(Frame& frame);
    void operator=(Frame& frame);

public:
    std::vector<cv::Mat> mImgPyr;
    std::vector<std::vector<cv::KeyPoint> > mKpsPyr;
    std::vector<std::vector<float> > mDepthPyr;

public:
    float mScaleFactor;
    double mTimeStamp;
    cv::Mat mR;
    cv::Mat mt;

public:
    void ExtractFASTPyr();
    void InitDepthPyr(float initDepth);

public: // for debug and display
    void ShowPyr();

};

class Measure2D {
    cv::Point2f mKeypoint;
    float       mD;
    
};

#endif