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

#ifndef  __ANDROID__
#include "Viewer.h"
#endif

enum PlaneDetectionState {
    VOID,
    INITIALIZING,
    TRACKING,
    FILTERING,
    FAILED,
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
    int mTrackFrameIndex;

    // plane calculation data
    std::vector<cv::Point2f> mPixelsMatchHMatrixSurfaceOnRefFrame;
    std::vector<cv::Point3f> mPoints3DMatchHMatrixSurface;

    // function members
    TextureSegment mTextureSeg;

    // result members
    std::vector<float> mMainPlane;
    std::vector<float> mAnchorPoint;
    int mWinnerTextureID;

public: // functions
    bool   SetRefFrame(Frame& f);
    bool   AddObvFrame(Frame& f);
    PlaneDetectionState Detect(cv::Mat image, std::vector<float> R_, std::vector<float> t_);

    bool   DetectMatchByOpticalFlow(Frame& ref, Frame& f);
    bool   DetectMatchByBatchOpticalFlow(Frame& ref, std::vector<Frame>& fSet);
    cv::Mat   ComputeHomographyFromMatchedPoints(std::vector<cv::Point2f> pts0, std::vector<cv::Point2f> pts1, std::vector<int>& indexsPlane);
    std::vector<float> CheckHomographyReprojError(cv::Mat H, std::vector<cv::Point2f> pts0, std::vector<cv::Point2f> pts1);
    bool   RecoverPlaneFromHomographyAndRT(cv::Mat H01, cv::Mat R0, cv::Mat t0, cv::Mat R1, cv::Mat t1);
    bool   RecoverPlaneFromPointPairsAndRT(std::vector<cv::Point2f> pts0, std::vector<cv::Point2f> pts1, 
                                                       cv::Mat R0, cv::Mat t0, cv::Mat R1, cv::Mat t1,
                                                       std::vector<cv::Point2f>& pixels3d,
                                                       std::vector<cv::Point3f>& points3d,
                                                       std::vector<float>& mainPlane, std::vector<float>& anchorPoint);
    bool   UpdatePlaneByTextureRelatedPoints(std::vector<cv::Point2f> pts, cv::Mat F_T_G, cv::Mat F_S_T, std::vector<int>& indexPts);
    std::vector<cv::Point3f> GetPlaneRegionUsingAnchorPointAndTexture();
    
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

    float  RecoverPlaneFrom3DPoints(std::vector<cv::Point3f> p3ds, std::vector<float>& mainPlane, std::vector<float>& anchorPoint);
    float  GetGridProb(cv::Point2f gridCenter, cv::Point2f pt, float gridR);
    float  GetDistPoint2Plane(cv::Point3f pt, std::vector<float> plane);
    float  GetPatchIntense(float u, float v, int width, unsigned char* image);
    
};

#endif