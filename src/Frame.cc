#include "Frame.h"

Frame::Frame(const cv::Mat& image, double timeStamp)
{
    mTimeStamp = timeStamp;
    mScaleFactor = 2.0;

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

void Frame::ExtractFASTPyr()
{
    mKpsPyr.resize(mImgPyr.size());
    for (int i = 0; i < mImgPyr.size(); i++) {
        std::vector<cv::KeyPoint> keypoints;
        cv::FAST(mImgPyr[i], keypoints, 10, true, cv::FastFeatureDetector::TYPE_9_16);
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


void Frame::ShowPyr()
{
    #ifndef __ANDROID__
    int LEVELSHOW = 3;
    cv::Mat img4Show = mImgPyr[LEVELSHOW].clone();
    cv::cvtColor(img4Show, img4Show, CV_GRAY2BGR);
    for (int i = LEVELSHOW; i < mImgPyr.size(); i++) {

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
                        3,
                        color, 1
                    );
        }
        //cv::drawKeypoints(img4Show, mKpsPyr[i], img4Show);
    }
    cv::imshow("keypoints in the frame", img4Show);
    #endif
}

void Frame::operator=(Frame& frame)
{
    mScaleFactor = frame.mScaleFactor;
    mTimeStamp   = frame.mTimeStamp;
    mR           = frame.mR;
    mt           = frame.mt;
    mKpsPyr      = frame.mKpsPyr;
    mDepthPyr    = frame.mDepthPyr;

    if (!mImgPyr.size()) {
        mImgPyr.resize(4);
    }
    for (int i = 0; i < frame.mImgPyr.size(); i++) {
        mImgPyr[i] = frame.mImgPyr[i].clone();
    }
}

