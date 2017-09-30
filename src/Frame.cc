#include "Frame.h"

Frame::Frame(const cv::Mat& image, double timeStamp)
{
    mTimeStamp = timeStamp;
    mScaleFactor = 2.0;
    mR = cv::Mat::eye(3, 3, CV_32FC1);
    mt = cv::Mat::zeros(3, 1, CV_32FC1);

    int LEVEL_NUM = 4;
    mImgPyr.resize(LEVEL_NUM, cv::Mat());

    for (int i = 0; i < LEVEL_NUM; i++) {
        if (i == 0) {
            image.copyTo(mImgPyr[i]);
        } else {
            cv::Size downSize( mImgPyr[i-1].cols/int(mScaleFactor), mImgPyr[i-1].rows/int(mScaleFactor) );
            cv::pyrDown( mImgPyr[i-1], mImgPyr[i], downSize );
        }
    }
}

Frame::Frame(const Frame& frame)
{
    if (!mImgPyr.size()) {
        mImgPyr.resize(4);
    }
    for (int i = 0; i < frame.mImgPyr.size(); i++) {
        mImgPyr[i] = frame.mImgPyr[i].clone();
    }
    mKpsPyr       = frame.mKpsPyr;
    mDepthPyr     = frame.mDepthPyr;
    mStatisticPyr = frame.mStatisticPyr;
    mScaleFactor  = frame.mScaleFactor;
    mTimeStamp    = frame.mTimeStamp;
    mR            = frame.mR.clone();
    mt            = frame.mt.clone(); 
}

void ExtractAllPixels()
{

}

void Frame::ExtractFastPyr()
{
    mKpsPyr.resize(mImgPyr.size());
    for (int i = 0; i < mImgPyr.size(); i++) {
        std::vector<cv::KeyPoint> keypoints;
        cv::FAST(mImgPyr[i](cv::Rect(mImgPyr[i].cols/4, mImgPyr[i].rows/4, mImgPyr[i].cols/2, mImgPyr[i].rows/2)), 
                            keypoints, 7, false, cv::FastFeatureDetector::TYPE_9_16);
        for (int j = 0; j < keypoints.size(); j++) {
            keypoints[j].pt = keypoints[j].pt + cv::Point2f(mImgPyr[i].cols/4, mImgPyr[i].rows/4);
            keypoints[j].octave = i;
        }
        mKpsPyr[i] = keypoints;
    }
}

void Frame::ExtractGradientPyr(int threshold)
{
    mKpsPyr.resize(mImgPyr.size());
    cv::Mat Gu = (cv::Mat_<float>(3,3) << -1,  0,  1, -2, 0, 2, -1, 0, 1);
    cv::Mat Gv = (cv::Mat_<float>(3,3) << -1, -2, -1,  0, 0, 0,  1, 2, 1);
    float gu, gv, mag;
    for (int i = 0; i < mImgPyr.size(); i++) {
        std::vector<cv::KeyPoint> keypoints;
        // cv::FAST(mImgPyr[i](cv::Rect(mImgPyr[i].cols/4, mImgPyr[i].rows/4, mImgPyr[i].cols/2, mImgPyr[i].rows/2)), 
        //                     keypoints, 20, true, cv::FastFeatureDetector::TYPE_9_16);

        cv::Mat imagef;
        mImgPyr[i].convertTo(imagef, CV_32FC1);
        for (int v = 0; v < mImgPyr[i].rows; v++) {
            for (int u = 0; u < mImgPyr[i].cols; u++) {
                if (u < mImgPyr[i].cols/8 || u > mImgPyr[i].cols*7/8 || v < mImgPyr[i].rows/8 || v > mImgPyr[i].rows*7/8)
                    continue;
                    
                gu = float(cv::sum(imagef(cv::Rect(u-1, v-1, 3, 3)).mul(Gu))[0]);
                gv = float(cv::sum(imagef(cv::Rect(u-1, v-1, 3, 3)).mul(Gv))[0]);
                mag = sqrt(gu*gu + gv*gv);
                if (mag > threshold) {
                    keypoints.push_back(cv::KeyPoint(float(u), float(v), 1, atan2(gv, gu), mag));
                }
            }
        }

        for (int j = 0; j < keypoints.size(); j++) {
            keypoints[j].octave = i;
        }
        mKpsPyr[i] = keypoints;
    }   
}

void Frame::ExtractFeaturePoint()
{
    mKpsPyr.resize(mImgPyr.size());
    for (int i = 0; i < mImgPyr.size(); i++) {
        std::vector<cv::KeyPoint> keypoints;
        ExtractFeaturePointOnLevel(mImgPyr[i], keypoints, i);
        mKpsPyr[i] = keypoints;
    }
}

void Frame::ExtractFeaturePointOnLevel(cv::Mat image, std::vector<cv::KeyPoint>& keypoints, int level)
{
    int threshold_low = 10;
    int threshold_high = 30;
    cv::Mat edge = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    cv::Canny(image, edge, threshold_low, threshold_high);
    // cv::imshow("canny edge", edge);
    std::vector<cv::Point2f> pts;
    for (int v = 0; v < image.rows; v++) {
        for (int u = 0; u < image.cols; u++) {
            if (  v > image.rows / 8 && v < image.rows * 7/8 
               && u > image.cols / 8 && u < image.cols * 7/8) {
                if (edge.data[u + v*image.cols] > 0) {
                    pts.push_back(cv::Point2f(u, v));
                }
            }
        }
    }

    std::vector<float> grad_megs = GetGradientMagnitude(image, pts);
    std::vector<cv::Point2f> pts_nms = NMSMaskOnCannyEdge(image, pts, grad_megs);
    std::vector<float> grad_megs_nms = GetGradientMagnitude(image, pts_nms);
    keypoints.resize(pts_nms.size());
    for (int i = 0; i < keypoints.size(); i++) {
        keypoints[i].pt = pts_nms[i];
        keypoints[i].response = grad_megs_nms[i];
        keypoints[i].octave = level;
    }

}

void Frame::InitDepthPyr(float initDepth)
{
    mDepthPyr.resize(mKpsPyr.size());
    mStatisticPyr.resize(mKpsPyr.size());
    for (int i = 0; i < mImgPyr.size(); i++) {
        std::vector<float> depth(mKpsPyr[i].size(), initDepth);
        std::vector<Statistic> statistic(mKpsPyr[i].size());
        mDepthPyr[i] = depth;
        mStatisticPyr[i] = statistic;
    }
}

cv::Mat Frame::GetDoubleSE3()
{
    cv::Mat so3 = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Rodrigues(mR, so3);
    cv::Mat res = ( cv::Mat_<double>(6,1) 
        << double(so3.at<float>(0,0)), double(so3.at<float>(1,0)), double(so3.at<float>(2,0)), 
           double(mt.at<float>(0,0)), double(mt.at<float>(1,0)), double(mt.at<float>(2,0)) );

    return res;
}

cv::Mat Frame::GetTcwMat()
{
    if (mR.empty() || mt.empty() )
        return cv::Mat::eye(4, 4, CV_32FC1);

    cv::Mat res = cv::Mat::eye(4, 4, CV_32FC1);
    cv::Mat Rt = mR.t();
    cv::Mat _t = -Rt*mt;
    Rt.copyTo(res.rowRange(0,3).colRange(0,3));
    _t.copyTo(res.rowRange(0,3).col(3));
    return res;
}

void Frame::ShowPyr(int levelShow)
{
    #ifndef __ANDROID__
    int LEVELSHOW = levelShow;
    cv::Mat img4Show = mImgPyr[LEVELSHOW].clone();
    cv::cvtColor(img4Show, img4Show, CV_GRAY2BGR);

    if (mKpsPyr.size() < LEVELSHOW) {
        return ;
    }

    for (int i = LEVELSHOW; i < LEVELSHOW+1; i++) {
        for (int j = 0; j < mKpsPyr[i].size(); j++) {
            cv::Scalar color;
            switch (mKpsPyr[i][j].octave) {
                case 3: color = cv::Scalar(231, 255, 0); break;
                case 2: color = cv::Scalar(255, 219, 0); break;
                case 1: color = cv::Scalar(255, 113, 0); break;
                case 0: color = cv::Scalar(255, 0, 0); break;
                default: color = cv::Scalar(255, 0, 255); break;
            }
            // cv::circle(img4Show, 
            //             mKpsPyr[i][j].pt * pow(mScaleFactor, mKpsPyr[i][j].octave),
            //             3 * pow(mScaleFactor, mKpsPyr[i][j].octave),
            //             color, 1
            //         );
            cv::circle(img4Show, 
                        mKpsPyr[i][j].pt,
                        1,
                        color, 1
                    );
        }
        //cv::drawKeypoints(img4Show, mKpsPyr[i], img4Show);
    }
    cv::imshow("keypoints in the frame", img4Show);
    #endif
}

void Frame::operator=(const Frame& frame)
{
    if (!mImgPyr.size()) {
        mImgPyr.resize(4);
    }
    for (int i = 0; i < frame.mImgPyr.size(); i++) {
        mImgPyr[i] = frame.mImgPyr[i].clone();
    }
    mKpsPyr       = frame.mKpsPyr;
    mDepthPyr     = frame.mDepthPyr;
    mStatisticPyr = frame.mStatisticPyr;
    mScaleFactor  = frame.mScaleFactor;
    mTimeStamp    = frame.mTimeStamp;
    mR            = frame.mR.clone();
    mt            = frame.mt.clone();
}

std::vector<float> Frame::GetGradientMagnitude(cv::Mat image, std::vector<cv::Point2f> pts)
{
    std::vector<float> res(pts.size(), 0);
    cv::Mat imagef;
    image.convertTo(imagef, CV_32FC1);
    cv::Mat Gu = (cv::Mat_<float>(3,3) << -1,  0,  1, -2, 0, 2, -1, 0, 1);
    cv::Mat Gv = (cv::Mat_<float>(3,3) << -1, -2, -1,  0, 0, 0,  1, 2, 1);

    int    u,  v;
    float gu, gv;
    for (int i = 0; i < pts.size(); i++) {
        u = int(pts[i].x + 0.5);
        v = int(pts[i].y + 0.5);
        gu = float(cv::sum(imagef(cv::Rect(u, v, 3, 3)).mul(Gu))[0]);
        gv = float(cv::sum(imagef(cv::Rect(u, v, 3, 3)).mul(Gv))[0]);
        res[i] = sqrt(gu*gu + gv*gv);

        // std::cout << i << " " << res[i] << std::endl;
    }

    return res;
}

std::vector<cv::Point2f> Frame::NMSMaskOnCannyEdge(cv::Mat image, const std::vector<cv::Point2f>& edge_points, const std::vector<float>& gradient_mags)
{
    std::vector<int>  mask(image.cols*image.rows, -1);
    int* pmask = (int*)mask.data();
    std::vector<bool> ifSuppressed(edge_points.size(), false); 
    std::vector<cv::Point2f> edge_points_afternms;

    for (int i = 0; i < edge_points.size(); i++) {
        pmask[int(edge_points[i].y+0.5) * image.cols + int(edge_points[i].x+0.5)] = i;
    }

    int R = image.cols / 100;
    int idx = 0;
    int u, v;
    for (int i = 0; i < edge_points.size(); i++) {
        if (!ifSuppressed[i]) {
            for (int dv = -R; dv <= R; dv++) {
                for (int du = -R; du <= R; du++) {
                    if (du*du + dv*dv < R*R ) {
                        u = int(edge_points[i].x+0.5);
                        v = int(edge_points[i].y+0.5);
                        idx = pmask[(dv + v)*image.cols + du + u];
                        if  ( idx >= 0) {
                            // std::cout << i << " " << u << " " << v << " " << du << " " << dv << " " 
                            //           << gradient_mags[i] << " " << idx << " " << gradient_mags[idx] << std::endl;
                            if (gradient_mags[i] > gradient_mags[idx]) {
                                ifSuppressed[idx] = true;
                            } 
                            else if (gradient_mags[i] < gradient_mags[idx]) {
                                ifSuppressed[i] = true;
                            }
                        }
                    }
                }
            }

        }

    }

    int count = 0;
    for (int i = 0; i < edge_points.size(); i++) {
        if (!ifSuppressed[i]) {
            edge_points_afternms.push_back(edge_points[i]);
            continue;
        }
    }

    std::cout << "num of points before and after nms: " << edge_points.size() << " " << edge_points_afternms.size() << std::endl;

    return edge_points_afternms;
}

