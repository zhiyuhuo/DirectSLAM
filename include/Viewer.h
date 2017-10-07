#ifndef __ANDROID__

#ifndef DIRECTSLAM_VIEWER_H
#define DIRECTSLAM_VIEWER_H

#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#include <pangolin/pangolin.h>

#include <mutex>

#include "CameraIntrinsic.h"
#include "PlaneDetector.h"
#include "Frame.h"

class Frame;
class CameraIntrinsic;
class PlaneDetector;

class Viewer
{
    public:
        Viewer() = default;
        Viewer(PlaneDetector* _pd): pd(_pd), isFinished(false) {};

        // 3D Drawing
        void run();
        void Stop();

        void drawRefFrame();
        void drawFrames();
        void drawPixelsDepth();
        void drawMapAxis();

        static void DrawPlane(CameraIntrinsic* K, Frame& f, std::vector<float> mainPlane, std::vector<cv::Point3f> points);
        static void DrawSquare(CameraIntrinsic* K, Frame& f, std::vector<float> mainPlane, float sqaureCentroidX, float sqaureCentroidY);
        static void DrawAR(CameraIntrinsic* K, Frame& f, std::vector<float> mainPlane);
        static void DrawAR(CameraIntrinsic* K, Frame& f, std::vector<float> mainPlane, std::vector<float> anchorPoint);
        static void GetAxis(CameraIntrinsic* K, Frame& f, const cv::Point3f& p3d1, const cv::Point3f &p3d2,
                        cv::Point2f& p2d1, cv::Point2f& p2d2);

    private:
        PlaneDetector* pd;

        std::mutex mMutexFinish;
        bool isFinished;

        pangolin::OpenGlRenderState* s_cam;
        pangolin::View* d_cam;
        pangolin::OpenGlMatrix M;

};

#endif   // VIEWER_H

#endif  // __ANDROID__
