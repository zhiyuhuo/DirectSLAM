#ifndef PLANEDETECTOR_H
#define PLANEDETECTOR_H

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

#define FRAMENUM 30

enum PlaneDetectionState {
    VOID,
    TRACKING,
    DETECTING,
    REFINING,
    FAILED,
    END
};

class PlaneDetector {
public: // constructors and destructors
    PlaneDetector() = default;
    ~PlaneDetector() = default;
    PlaneDetector(CameraIntrinsic* K);

public: // members
    int mFrameNum;
    CameraIntrinsic* mK;
    PlaneDetectionState mState;

    Frame mRefFrame;
    std::vector<Frame> mFramesBuffer;
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
    PlaneDetectionState Detect(const cv::Mat& image, cv::Mat R_, cv::Mat t_);

    // import frames
    bool   SetRefFrame(Frame& f);
    bool   AddFrameToBuffer(Frame& f);

    // OF tracking
    bool   DetectMatchByOpticalFlow(Frame& ref, Frame& f);
    bool   DetectMatchByBatchOpticalFlow(Frame& ref, std::vector<Frame>& fSet);

    // calculate histogram from point pairs by RANSAC
    cv::Mat   ComputeHomographyFromMatchedPoints(std::vector<cv::Point2f> pts0, std::vector<cv::Point2f> pts1, 
                                                 std::vector<int>& indexsPlane);
    
    // check dual-direction reprojection errors for several point pairs
    std::vector<float> CheckHomographyReprojErrors(cv::Mat H, std::vector<cv::Point2f> pts0, std::vector<cv::Point2f> pts1);

    // check dual-direction reprojection error  for a point pair
    float CheckHomographyReprojError(cv::Mat H, cv::Point2f pt0_, cv::Point2f pt1_);
    
    // recover plane from triangluated points
    bool   RecoverPlaneFromPointPairsAndRT(std::vector<cv::Point2f> pts0, std::vector<cv::Point2f> pts1, 
                                                       cv::Mat R0, cv::Mat t0, cv::Mat R1, cv::Mat t1,
                                                       std::vector<cv::Point2f>& pixels3d,
                                                       std::vector<cv::Point3f>& points3d,
                                                       std::vector<float>& mainPlane, std::vector<float>& anchorPoint);

    // distance of a 3D point to a 3D plane
    float  GetDistPoint2Plane(cv::Point3f pt, std::vector<float> plane);

    // calculate plane from 3D points using egien
    float  RecoverPlaneFrom3DPoints(std::vector<cv::Point3f> p3ds, std::vector<float>& mainPlane, std::vector<float>& anchorPoint);

    // get the [0,1] prob value that a point to a grid
    float  GetGridProb(cv::Point2f gridCenter, cv::Point2f pt, float gridR);
    
    // calculate the probablistic model of surface(plane)---grid---texture
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

    // re-calculate plane by points associated to the main texture.
    bool   UpdatePlaneByTextureRelatedPoints(std::vector<cv::Point2f> pts, cv::Mat F_T_G, cv::Mat F_S_T, std::vector<int>& indexPts);

    // get plane region. for drawing
    std::vector<cv::Point3f> GetPlaneRegionUsingAnchorPointAndTexture();
};

#endif