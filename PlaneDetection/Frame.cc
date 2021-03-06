#include "Frame.h"

inline float AngleDiff(float a1, float a2) 
{
    float diff = a1 - a2;
    while (diff > PI) {
        diff -= PI;
    }
    while (diff < -PI) {
        diff += PI;
    }
    if (diff < 0) {
        diff = -diff;
    }
    diff = (diff < PI-diff)? diff : PI-diff;
    return diff;
}

Frame::Frame(const cv::Mat& image, double timeStamp)
{
    mTimeStamp = timeStamp;
    mR = cv::Mat::eye(3, 3, CV_32FC1);
    mt = cv::Mat::zeros(3, 1, CV_32FC1);
    mImg = image.clone();
}

Frame::Frame(const cv::Mat& image, const cv::Mat& R, const cv::Mat& t, double timeStamp)
{
    mTimeStamp = timeStamp;
    mR = R.clone();
    mt = t.clone();
    mImg = image.clone();
}

Frame::Frame(const Frame& frame)
{
    mImg                    = frame.mImg.clone();
    mKps                    = frame.mKps;
    mTimeStamp              = frame.mTimeStamp;
    mR                      = frame.mR.clone();
    mt                      = frame.mt.clone(); 
}

void Frame::ExtractFeaturePoint()
{
    std::vector<cv::KeyPoint> keypoints, keypoints_fast, keypoints_edge;
    std::vector<bool> ifGetNMS_fast, ifGetNMS_edge;

    ExtractFastPointOnLevel(mImg, keypoints_fast);
    keypoints_fast = NMSMask(mImg, keypoints_fast, ifGetNMS_fast);
    keypoints.insert(keypoints.end(), keypoints_fast.begin(), keypoints_fast.end());

    // ExtractEdgePointOnLevel(mImg, keypoints_edge);
    // keypoints_edge = NMSMask(mImg, keypoints_edge, ifGetNMS_edge);
    // keypoints.insert(keypoints.end(), keypoints_edge.begin(), keypoints_edge.end());

    // for (int i = 0; i < keypoints.size(); i++) {
    //     std::cout << keypoints[i].class_id << " " << keypoints[i].response << " " << std::endl;
    // }

    mKps = keypoints;
    
}

void Frame::ExtractFastPointOnLevel(cv::Mat image, std::vector<cv::KeyPoint>& keypoints)
{
    int fastThreshold = 5;

    std::vector<cv::KeyPoint> kps;
    cv::FAST(image(cv::Rect(image.cols/8, image.rows/8, image.cols*6/8,image.rows*6/8)), 
                        kps, fastThreshold, true, cv::FastFeatureDetector::TYPE_9_16);
    for (int i = 0; i < kps.size(); i++) {
        kps[i].pt = kps[i].pt + cv::Point2f(image.cols/8, image.rows/8);
    }
    ComputeResponses(image, kps, fastThreshold);
    for (int i = 0; i < kps.size(); i++) {
        keypoints.push_back(kps[i]);
        kps[i].class_id = 0;
    }
}

void Frame::ExtractEdgePointOnLevel(cv::Mat image, std::vector<cv::KeyPoint>& keypoints)
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
    std::vector<cv::KeyPoint> kps(pts.size());
    for (int i = 0; i < kps.size(); i++) {
        kps[i].pt = pts[i];
        kps[i].response = grad_megs[i];
        kps[i].class_id = 1;
    }
    for (int i = 0; i < kps.size(); i++) {
        keypoints.push_back(kps[i]);
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

void Frame::ShowImage()
{
    #ifndef __ANDROID__
    cv::Mat img4Show = mImg.clone();
    cv::cvtColor(img4Show, img4Show, CV_GRAY2BGR);

    for (int i = 0; i < mKps.size(); i++) {
        cv::Scalar color;
        switch (mKps[i].class_id) {
            case 3: color = cv::Scalar(231, 255, 0); break;
            case 2: color = cv::Scalar(255, 219, 0); break;
            case 1: color = cv::Scalar(255, 113, 0); break;
            case 0: color = cv::Scalar(255, 0, 0); break;
            default: color = cv::Scalar(255, 0, 255); break;
        }
        cv::circle(img4Show, 
                    mKps[i].pt,
                    1,
                    color, 1
                );
    }
    cv::imshow("keypoints in the frame", img4Show);
    #endif
}

void Frame::operator=(const Frame& frame)
{
    mImg          = frame.mImg.clone();
    mKps          = frame.mKps;
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
        gu = float(cv::sum(imagef(cv::Rect(u, v, 3, 3)).mul(Gu))[0]) * 0.5;
        gv = float(cv::sum(imagef(cv::Rect(u, v, 3, 3)).mul(Gv))[0]) * 0.5;
        res[i] = sqrt(gu*gu + gv*gv);

        // std::cout << i << " " << res[i] << std::endl;
    }

    return res;
}

void Frame::ComputeResponses(cv::Mat image, std::vector<cv::KeyPoint>& keypoints, int FASTThres) 
{
    if (keypoints.size() == 0) 
        return;

    std::vector<float> responses(keypoints.size());

    int offset[16];
    int fast_ring16_x[16] = {0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1};
    int fast_ring16_y[16] = {3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3};

    for (int i = 0; i < 16; ++i) {
        offset[i] = fast_ring16_y[i]*image.cols + fast_ring16_x[i];
    }

    for (int i = 0, _end = keypoints.size(); i < _end; ++i) {
        unsigned char *_data = (unsigned char *)image.data;
        _data = _data + (int)(keypoints[i].pt.x + keypoints[i].pt.y*image.cols);
        int cb  = *_data + FASTThres;
        int c_b = *_data - FASTThres; 
        int sp = 0, sn = 0;

        for (int j = 0; j < 16; ++j) {
            int p = _data[offset[j]];
                
            if (p > cb)
                sp += (p-cb);
            else if (p < c_b)
                sn += (c_b-p);
        }

        if (sp > sn)
            responses[i] = sp / 16;
        else
            responses[i] = sn / 16;

    }

    for (int i = 0; i < responses.size(); i++) {
        keypoints[i].response = responses[i];
    }
}

std::vector<cv::KeyPoint> Frame::NMSMask(cv::Mat image, std::vector<cv::KeyPoint>& points, std::vector<bool>& ifGetNMS)
{
    std::vector<int>  mask(image.cols*image.rows, -1);
    int* pmask = (int*)mask.data();
    std::vector<bool> ifSuppressed(points.size(), false); 

    for (int i = 0; i < points.size(); i++) {
        pmask[int(points[i].pt.y+0.5) * image.cols + int(points[i].pt.x+0.5)] = i;
    }

    int R = image.cols / 100;
    if (R < 1) {
        R = 1;
    }
    int idx = 0;
    int u, v;
    for (int i = 0; i < points.size(); i++) {
        if (!ifSuppressed[i]) {
            for (int dv = -R; dv <= R; dv++) {
                for (int du = -R; du <= R; du++) {
                    if (du*du + dv*dv < R*R ) {
                        u = int(points[i].pt.x+0.5);
                        v = int(points[i].pt.y+0.5);
                        idx = pmask[(dv + v)*image.cols + du + u];
                        if  ( idx >= 0) {
                            // std::cout << i << " " << u << " " << v << " " << du << " " << dv << " " 
                            //           << gradient_mags[i] << " " << idx << " " << gradient_mags[idx] << std::endl;
                            if (points[i].response > points[idx].response) {
                                ifSuppressed[idx] = true;
                            } 
                            else if (points[i].response < points[idx].response) {
                                ifSuppressed[i] = true;
                            }
                        }
                    }
                }
            }

        }

    }

    std::vector<cv::KeyPoint> points_afternms;
    int count = 0;
    for (int i = 0; i < points.size(); i++) {
        // std::cout << points[i].class_id << " " << points[i].response << " " << ifSuppressed[i] << std::endl;
        if (!ifSuppressed[i]) {
            points_afternms.push_back(points[i]);
            continue;
        }
    }
    ifGetNMS = ifSuppressed;

    // std::cout << "num of points before and after nms: " << ifSuppressed.size() << " " << points_afternms.size() << std::endl;

    return points_afternms;
}