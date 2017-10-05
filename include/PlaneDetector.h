#ifndef DIRECTSLAM_PD_H
#define DIRECTSLAM_PD_H

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "Log.h"
#include "Timer.h"
#include "CameraIntrinsic.h"
#include "Frame.h"
#include "TextureSegment.h"

enum PlaneDetectionState {
    VOID,
    INITIALIZING,
    TRACKING,
    FILTERING,
    END
};

class PlaneDetector {
public: // constructors and destructors
    PlaneDetector() = default;
    ~PlaneDetector() = default;
    PlaneDetector(CameraIntrinsic* K);

public: // members
    int mLevel;
    int mFrameNum;
    CameraIntrinsic* mK;
    PlaneDetectionState mState;

    Frame mRefFrame;
    std::vector<Frame> mFrameVecBuffer;

    // plane calculation data
    std::vector<cv::Point2f> mPixelsMatchHMatrixOnRefFrame;

    // function members
    TextureSegment mTextureSeg;

public: // functions
    bool   SetRefFrame(Frame& f);
    bool   AddObvFrame(Frame& f);
    PlaneDetectionState Detect(cv::Mat image, std::vector<float> R_, std::vector<float> t_);

    void   DetectMatchByOpticalFlow(Frame& ref, Frame& f);
    cv::Mat   ComputeHomographyFromMatchedPoints(std::vector<cv::Point2f> pts0, std::vector<cv::Point2f> pts1, std::vector<int>& indexsPlane);
    std::vector<float> CheckHomographyReprojError(cv::Mat H, std::vector<cv::Point2f> pts0, std::vector<cv::Point2f> pts1);
    cv::Mat RecoverPlaneFromHomographyAndRT(cv::Mat H01, cv::Mat R0, cv::Mat t0, cv::Mat R1, cv::Mat t1);
    cv::Mat RecoverPlaneFromPointPairsAndRT(std::vector<cv::Point2f> pts0, std::vector<cv::Point2f> pts1, 
                                                       cv::Mat R0, cv::Mat t0, cv::Mat R1, cv::Mat t1);
    
    // for drawing the grids on the horizontal surface.
    // F(S|G)
    cv::Mat CalculateConditionalDistribution_SurfaceGrid(std::vector<cv::Point2f> ptsMatchHMat);  
    // F(S|T)
    cv::Mat CalculateConditionalDistribution_SurfaceTexture(cv::Mat F_S_G, cv::Mat F_T_G);
    // F(T|G)
    cv::Mat CalculateConditionalDistribution_TextureGrid();
    // F(T)
    cv::Mat CalculateMarginalDistribution_Texture(cv::Mat F_T_G);
    // F(S)
    cv::Mat CalculateMarginalDistribution_Surface(cv::Mat F_S_G);

    // get the prob of a key point belonging to the horizontal surface.
    void GetProb_SurfacePoint();
    // get the prob of a grid belonging to the horizontal surface.
    void GetProb_SurfaceGrid();

    float  GetPatchIntense(float u, float v, int width, unsigned char* image);
    
};

#endif