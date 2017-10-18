#ifndef DIRECTSLAM_FRAME_H
#define DIRECTSLAM_FRAME_H

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/line_descriptor.hpp>
// #include "opencv2/core/utility.hpp"
// #include <opencv2/imgproc.hpp>
// #include <opencv2/features2d.hpp>
// #include <opencv2/highgui.hpp>
#include "Log.h"
#include "Statistic.h"
#include "GeometryR.h"
#include "CameraIntrinsic.h"

#define PI 3.1415926535897932

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

    // landmark lines related members
    std::vector<cv::line_descriptor::KeyLine> mKeyLines;
    std::vector<int>              mLandmarkLinesIndexs;
    std::vector<cv::Mat>          mLandmarkPlanes;
    std::map<int, cv::Point2f>    mLandmarkIntersectPts;

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
    void ShowLines();

public: // landmark lines related functions. The LSD and LST will only work at 0 scale 
        // since it is not very sensitive to time but sensitive to accuracy
    void    ExtractLines();
    void    FindLandmarkLines();
    int     FindLandmarkLine0();
    int     FindValidLines();
    void    ExtractLandmarkLinePlane(CameraIntrinsic* K);
    cv::Mat ExtractLinePlane(cv::line_descriptor::KeyLine line, CameraIntrinsic* K, cv::Mat R, cv::Mat t);
    void    TrackLandmarkLineRefFrame(Frame& refframe);
    void    TrackLandmarkLines(std::vector<cv::line_descriptor::KeyLine> reflines, 
                               std::vector<cv::line_descriptor::KeyLine> lines,
                               std::vector<int>&                         indexs,
                               int                                       trackNum,
                               float                                     diagLength);
    int     MatchLineFromScene(cv::line_descriptor::KeyLine refline, std::vector<cv::line_descriptor::KeyLine> lines, float diagLength);
    float   MatchBetweenTwoLines(cv::line_descriptor::KeyLine line1, cv::line_descriptor::KeyLine line2);
    int    GetAllIntersectionPointsFromLandmarks();
    cv::Point2f IntersectionOfTwoLines(cv::line_descriptor::KeyLine line1, cv::line_descriptor::KeyLine line2);

};

class Measure2D {
    cv::Point2f mKeypoint;
    float       mD;
    
};

#endif