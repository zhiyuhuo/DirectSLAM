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
    mKpsPyr      = frame.mKpsPyr;
    mDepthPyr    = frame.mDepthPyr;
    mScaleFactor = frame.mScaleFactor;
    mTimeStamp   = frame.mTimeStamp;
    mR           = frame.mR.clone();
    mt           = frame.mt.clone(); 
}

void Frame::ExtractFastPyr()
{
    mKpsPyr.resize(mImgPyr.size());
    for (int i = 0; i < mImgPyr.size(); i++) {
        std::vector<cv::KeyPoint> keypoints;
        cv::FAST(mImgPyr[i](cv::Rect(mImgPyr[i].cols/4, mImgPyr[i].rows/4, mImgPyr[i].cols/2, mImgPyr[i].rows/2)), 
                            keypoints, 20, true, cv::FastFeatureDetector::TYPE_9_16);
        for (int j = 0; j < keypoints.size(); j++) {
            keypoints[j].pt = keypoints[j].pt + cv::Point2f(mImgPyr[i].cols/4, mImgPyr[i].rows/4);
            keypoints[j].octave = i;
        }
        mKpsPyr[i] = keypoints;
    }
}

void Frame::ExtractSlopePyr(int threshold)
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

void Frame::InitDepthPyr(float initDepth)
{
    mDepthPyr.resize(mKpsPyr.size());
    for (int i = 0; i < mImgPyr.size(); i++) {
        std::vector<float> depth(mKpsPyr[i].size(), initDepth);
        mDepthPyr[i] = depth;
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


void Frame::ShowPyr(int levelShow)
{
    #ifndef __ANDROID__
    int LEVELSHOW = levelShow;
    cv::Mat img4Show = mImgPyr[LEVELSHOW].clone();
    cv::cvtColor(img4Show, img4Show, CV_GRAY2BGR);
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
    mKpsPyr      = frame.mKpsPyr;
    mDepthPyr    = frame.mDepthPyr;
    mScaleFactor = frame.mScaleFactor;
    mTimeStamp   = frame.mTimeStamp;
    mR           = frame.mR.clone();
    mt           = frame.mt.clone();
}

