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
    mKpsPyr                 = frame.mKpsPyr;
    mDepthPyr               = frame.mDepthPyr;
    mStatisticPyr           = frame.mStatisticPyr;
    mScaleFactor            = frame.mScaleFactor;
    mTimeStamp              = frame.mTimeStamp;
    mR                      = frame.mR.clone();
    mt                      = frame.mt.clone(); 
    mKeyLines               = frame.mKeyLines;
    mLandmarkLinesIndexs    = frame.mLandmarkLinesIndexs;
    for (int i = 0; i < frame.mLandmarkPlanes.size(); i++) {
        mLandmarkPlanes[i]  = frame.mLandmarkPlanes[i].clone();
    }
    mLandmarkIntersectPts   = frame.mLandmarkIntersectPts;
}

void Frame::ExtractFeaturePoint()
{
    mKpsPyr.resize(mImgPyr.size());
    for (int i = 0; i < mImgPyr.size(); i++) {
        std::vector<cv::KeyPoint> keypoints, keypoints_fast, keypoints_edge;
        std::vector<bool> ifGetNMS_fast, ifGetNMS_edge;

        ExtractFastPointOnLevel(mImgPyr[i], keypoints_fast, i);
        keypoints_fast = NMSMask(mImgPyr[i], keypoints_fast, ifGetNMS_fast);

        ExtractEdgePointOnLevel(mImgPyr[i], keypoints_edge, i);
        keypoints_edge = NMSMask(mImgPyr[i], keypoints_edge, ifGetNMS_edge);

        keypoints.insert(keypoints.end(), keypoints_fast.begin(), keypoints_fast.end());
        keypoints.insert(keypoints.end(), keypoints_edge.begin(), keypoints_edge.end());

        // for (int i = 0; i < keypoints.size(); i++) {
        //     std::cout << keypoints[i].class_id << " " << keypoints[i].response << " " << std::endl;
        // }

        mKpsPyr[i] = keypoints;
    }
}

void Frame::ExtractFastPointOnLevel(cv::Mat image, std::vector<cv::KeyPoint>& keypoints, int level)
{
    int fastThreshold = 10 - 3*level;
    if (fastThreshold < 5) {
        fastThreshold = 5;
    }

    std::vector<cv::KeyPoint> kps;
    cv::FAST(image(cv::Rect(image.cols/8, image.rows/8, image.cols*6/8,image.rows*6/8)), 
                        kps, fastThreshold, false, cv::FastFeatureDetector::TYPE_9_16);
    for (int i = 0; i < kps.size(); i++) {
        kps[i].pt = kps[i].pt + cv::Point2f(image.cols/8, image.rows/8);
        kps[i].octave = level;
        kps[i].class_id = 0;
    }
    ComputeResponses(image, kps, fastThreshold);
    for (int i = 0; i < kps.size(); i++) {
        keypoints.push_back(kps[i]);
    }
}

void Frame::ExtractEdgePointOnLevel(cv::Mat image, std::vector<cv::KeyPoint>& keypoints, int level)
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
        kps[i].octave = level;
        kps[i].class_id = 1;
    }
    for (int i = 0; i < kps.size(); i++) {
        keypoints.push_back(kps[i]);
    }
}

void Frame::ExtractLines()
{
    /* create a binary mask */
    cv::Mat mask = cv::Mat::ones( mImgPyr[0].size(), CV_8UC1 );
    /* create a pointer to a BinaryDescriptor object with default parameters */
    cv::Ptr<cv::line_descriptor::BinaryDescriptor> bd = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();
    /* compute lines */
    bd->detect( mImgPyr[0], mKeyLines, mask );

    Log_info("mKeyLines number: {}", mKeyLines.size());   
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
            switch (mKpsPyr[i][j].class_id) {
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

void Frame::ShowLines()
{
    #ifndef __ANDROID__
    std::vector<cv::line_descriptor::KeyLine> lines; 
    for (int i = 0; i < mLandmarkLinesIndexs.size(); i++) {
        lines.push_back(mKeyLines[mLandmarkLinesIndexs[i]]);
    }

    cv::Mat output = mImgPyr[0].clone();
    /* draw lines extracted from octave 0 */
    if( output.channels() == 1 ) {
        cv::cvtColor( output, output, cv::COLOR_GRAY2BGR );
    }
    int R, G, B;
    for ( size_t i = 0; i < lines.size(); i++ )
    {
        cv::line_descriptor::KeyLine kl = lines[i];
        if( kl.octave == 0)
        {
            /* get a random color */
            R = ( rand() % (int) ( 255 + 1 ) );
            G = ( rand() % (int) ( 255 + 1 ) );
            B = ( rand() % (int) ( 255 + 1 ) );
//             R = 255;
//             G = 128;
//             B = 64;

            /* get extremes of line */
            cv::Point pt1 = cv::Point2f( kl.startPointX, kl.startPointY );
            cv::Point pt2 = cv::Point2f( kl.endPointX, kl.endPointY );
     
            /* draw line */
            cv::line( output, pt1, pt2, cv::Scalar( B, G, R ), 3 );
        }
    }
 
    /* show lines on image */
    cv::imshow( "BP lines", output );
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

    mKeyLines            = frame.mKeyLines;
    mLandmarkLinesIndexs = frame.mLandmarkLinesIndexs;
    mLandmarkPlanes.resize(frame.mLandmarkPlanes.size());
    for (int i = 0; i < frame.mLandmarkPlanes.size(); i++) {
        mLandmarkPlanes[i] = frame.mLandmarkPlanes[i].clone();
    }
    mLandmarkIntersectPts   = frame.mLandmarkIntersectPts;
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

void Frame::ExtractLandmarkLinePlane(CameraIntrinsic* K)
{
    // mLandmarkPlanes.resize(0);
    // cv::Mat plane_landmarkline0 = ExtractLinePlane(mKeyLines[mLandmarkLinesIndexs[0]], K, mR, mt);
    // mLandmarkPlanes.push_back(plane_landmarkline0);

    mLandmarkPlanes.resize(mLandmarkLinesIndexs.size());
    for (int i = 0; i < mLandmarkPlanes.size(); i++) {
        if (mLandmarkLinesIndexs[i] >= 0) {
            cv::Mat landmarkPlaneNorm = ExtractLinePlane(mKeyLines[mLandmarkLinesIndexs[i]], K, mR, mt);
            mLandmarkPlanes[i] = landmarkPlaneNorm.clone();
        }
        else {
            mLandmarkPlanes[i] = cv::Mat();
        }
    }
}

cv::Mat Frame::ExtractLinePlane(cv::line_descriptor::KeyLine line, CameraIntrinsic* K, cv::Mat R, cv::Mat t)
{
    cv::Mat res = cv::Mat::zeros(4, 1, CV_32FC1);

    // 1. find two end points of the line segmentation.
    VecPosition posCentroid(line.pt.x, line.pt.y);
    VecPosition posEnd0 = posCentroid + VecPosition::GetVecPositionFromPolar(line.lineLength * .5, line.angle);
    VecPosition posEnd1 = posCentroid + VecPosition::GetVecPositionFromPolar(line.lineLength * .5, line.angle - PI);

    // Log_info("2 end points of the landmark0: [{} {}],[{} {}]", posEnd0.GetX(), posEnd0.GetY(), posEnd1.GetX(), posEnd1.GetY());
    float fx = K->fx;
    float fy = K->fy;
    float cx = K->cx;
    float cy = K->cy;

    float z0 = 1; 
    float x0 = (posEnd0.GetX() - cx) / fx * z0;
    float y0 = (posEnd0.GetY() - cy) / fy * z0;

    float z1 = 1;
    float x1 = (posEnd1.GetX() - cx) / fx * z1;
    float y1 = (posEnd1.GetY() - cy) / fy * z1;
    
    cv::Mat vec0 = (cv::Mat_<float>(3,1) << x0, y0, z0);
    cv::Mat vec1 = (cv::Mat_<float>(3,1) << x1, y1, z1);
    vec0 = vec0 * (1. / cv::norm(vec0));
    vec1 = vec1 * (1. / cv::norm(vec1));

    cv::Mat planeNorm = vec0.cross(vec1);
    // std::cout << mR << std::endl;
    // std::cout << mt << std::endl;
    cv::Mat planeNormWorld = mR.t() * planeNorm;
    // std::cout << vec0.t() << std::endl << vec1.t() << std::endl << planeNorm.t() << std::endl << planeNormWorld.t() << std::endl;
    planeNormWorld = planeNormWorld * (1. / cv::norm(planeNormWorld));
    res = planeNorm.clone();

    return res;
}

void Frame::FindLandmarkLines()
{   
    FindValidLines();
}

int Frame::FindLandmarkLine0() 
{
    std::vector<cv::line_descriptor::KeyLine>& lines = mKeyLines;
    // find the length of the image diag
    float lengthImageDiag = sqrt(mImgPyr[0].cols*mImgPyr[0].cols + mImgPyr[0].rows*mImgPyr[0].rows);
    Log_info("Image diag length: {}", lengthImageDiag);
    
    // find the longest line seg in the image. the line seg should be nearly parallel to the long side of the image.
    int id_match = 0;
    float lineLength_longest = 0;
    for (int i = 0; i < lines.size(); i++) {
        if (lines[i].lineLength > lineLength_longest) {
            lineLength_longest = lines[i].lineLength;
            id_match = i;
        }
    }
    
    if (lines[id_match].lineLength < lengthImageDiag/3
        || AngleDiff(lines[id_match].angle, 0.0001) > PI/6)
        return false;
    
    int id_landmarkline0 = id_match;
    
    Log_info("id_landmarkline0: {},  lineLength_landmarkline0: {}, angle_landmarkline0: {}, pt_landmarkline0: [{},{}]", 
        id_landmarkline0, lines[id_landmarkline0].lineLength, lines[id_landmarkline0].angle, 
        lines[id_landmarkline0].pt.x, lines[id_landmarkline0].pt.y);
    
    return id_landmarkline0;
}

int Frame::FindValidLines()
{
    std::vector<cv::line_descriptor::KeyLine>& lines = mKeyLines;

    std::vector<float> lineLengthes(lines.size());
    std::vector<int>   lineIndexs(lineLengthes.size());
    std::size_t n(0);
    std::generate(std::begin(lineIndexs), std::end(lineIndexs), [&]{ return n++; });

    for (int i = 0; i < lines.size(); i++) {
        lineLengthes[i] = lines[i].lineLength;
    }

    std::sort(  std::begin(lineIndexs), std::end(lineIndexs), [&](int i1, int i2) { return lineLengthes[i1] > lineLengthes[i2]; } );
    for (int i = 0; i < lineIndexs.size(); i++) {
        if ( lineLengthes[lineIndexs[i]] > 0.3 * lineLengthes[lineIndexs[0]] ) {
            mLandmarkLinesIndexs.push_back(lineIndexs[i]);
        }
    }

}

void Frame::TrackLandmarkLineRefFrame(Frame& refframe)
{
    std::vector<cv::line_descriptor::KeyLine>  reflines; 
    std::vector<cv::line_descriptor::KeyLine>& lines    = mKeyLines;
    std::vector<int> indexs;

    for (int i = 0; i < refframe.mLandmarkLinesIndexs.size(); i++) {
        reflines.push_back(refframe.mKeyLines[refframe.mLandmarkLinesIndexs[i]]);
    }

    float diagLength = sqrt(mImgPyr[0].cols * mImgPyr[0].cols + mImgPyr[0].rows * mImgPyr[0].rows);
    TrackLandmarkLines(reflines, lines, indexs, 5, diagLength);
    mLandmarkLinesIndexs = indexs;
}

void Frame::TrackLandmarkLines(std::vector<cv::line_descriptor::KeyLine>  reflines, 
                                std::vector<cv::line_descriptor::KeyLine> lines,
                                std::vector<int>&                         indexs,
                                int                                       trackNum,
                                float                                     diagLength)
{
    indexs.resize(0);
    for (int i = 0; i < reflines.size() && i < trackNum; i++) {
        int id_match = MatchLineFromScene(reflines[i], lines, diagLength);
        indexs.push_back(id_match);
    }
}

int Frame::MatchLineFromScene(cv::line_descriptor::KeyLine refline, std::vector<cv::line_descriptor::KeyLine> lines, float diagLength)
{
    int max_id = -1;
    float max_sc = 0.0;
    for (int i = 0; i < lines.size(); i++) {
        float inters = MatchBetweenTwoLines(refline, lines[i]);
        float dist = cv::norm(lines[i].pt - refline.pt);
        dist = 1 - dist / diagLength; 
        
        float sc = dist * inters;
        if (sc > max_sc && sc > 0.5) {
            max_sc = sc;
            max_id = i;
        }
        // Log_info("line {}:   \t{}   \t{}   \t{}", i, inters, dist, sc);
    }
    
    return max_id;
}

float Frame::MatchBetweenTwoLines(cv::line_descriptor::KeyLine line1, cv::line_descriptor::KeyLine line2)
{
    float res = 0;
    VecPosition c1(line1.pt.x, line1.pt.y);
    VecPosition c2(line2.pt.x, line2.pt.y);
    float an1 = line1.angle;
    float an2 = line2.angle;
    float L1 = line1.lineLength;
    float L2 = line2.lineLength;
    Line l1 = Line::MakeLineFromPositionAndAngle(c1, an1);
    Line l2 = Line::MakeLineFromPositionAndAngle(c2, an2);
    
    VecPosition p1u, p1d;
    VecPosition p11 = c1 + VecPosition::GetVecPositionFromPolar( L1/2, an1);
    VecPosition p12 = c1 + VecPosition::GetVecPositionFromPolar(-L1/2, an1);
    if (p11.GetY() > p12.GetY()) {
        p1u = p11;
        p1d = p12;
    }
    else {
        p1u = p12;
        p1d = p11;
    }
    
    VecPosition p2u, p2d;
    VecPosition p21 = c2 + VecPosition::GetVecPositionFromPolar( L2/2, an2);
    VecPosition p22 = c2 + VecPosition::GetVecPositionFromPolar(-L2/2, an2);
    if (p21.GetY() > p22.GetY()) {
        p2u = p21;
        p2d = p22;
    }
    else {
        p2u = p22;
        p2d = p21;
    }
    
    VecPosition q2u, q2d;
    VecPosition q21 = l1.GetPointOnLineClosestTo(p2u);
    VecPosition q22 = l1.GetPointOnLineClosestTo(p2d);
    if (q21.GetY() > q22.GetY()) {
        q2u = q21;
        q2d = q22;
    }
    else {
        q2u = q22;
        q2d = q21;
    }
    
    // Get the intersection region
    if (q2u.GetY() < p1d.GetY() || q2d.GetY() > p1u.GetY()) {
        res = 0;
    }
    
    else if (q2d.GetY() < p1d.GetY() && q2u.GetY() > p1u.GetY()) { 
        res = L1 * 2 / (L1 + L2);
    }
        
    else if (q2d.GetY() < p1d.GetY() && q2u.GetY() < p1u.GetY()){
            res = (q2u - p1d).GetMagnitude() * 2 / (L1 + L2);
    }
    
    else if (q2d.GetY() > p1d.GetY() && q2u.GetY() < p1u.GetY()){
            res = (q2u - q2d).GetMagnitude() * 2 / (L1 + L2);
    }
    
    else if (q2d.GetY() > p1d.GetY() && q2u.GetY() > p1u.GetY()){
            res = (p1u - q2d).GetMagnitude() * 2 / (L1 + L2);
    }
    

    return res;
}

int Frame::GetAllIntersectionPointsFromLandmarks()
{
    int res;
    if (mLandmarkLinesIndexs.size() < 2) {
        return 0;
    }

    for (int i = 0; i < mLandmarkLinesIndexs.size(); i++) {
        for (int j = 0; j < mLandmarkLinesIndexs.size(); j++) {
            if (i > j) {
                if (mLandmarkLinesIndexs[i] < 0 || mLandmarkLinesIndexs[j] < 0) {
                    continue;
                }

                cv::Point2f intersectPt = IntersectionOfTwoLines(mKeyLines[mLandmarkLinesIndexs[i]],
                                                                 mKeyLines[mLandmarkLinesIndexs[j]]);
                // std::cout << i << ": " << mKeyLines[mLandmarkLinesIndexs[i]].pt << " " << mKeyLines[mLandmarkLinesIndexs[i]].angle << ",\t"
                //           << j << ": " << mKeyLines[mLandmarkLinesIndexs[j]].pt << " " << mKeyLines[mLandmarkLinesIndexs[j]].angle << ",\t"
                //           << intersectPt << std::endl;

                mLandmarkIntersectPts[i*1000+j] = intersectPt;
            }
        }
    }
}

cv::Point2f Frame::IntersectionOfTwoLines(cv::line_descriptor::KeyLine line1, cv::line_descriptor::KeyLine line2)
{
    cv::Point2f res(0,0);

    Line l1 = Line::MakeLineFromPositionAndAngle(VecPosition(line1.pt.x, line1.pt.y), line1.angle);
    Line l2 = Line::MakeLineFromPositionAndAngle(VecPosition(line2.pt.x, line2.pt.y), line2.angle);

    VecPosition intersection = l1.GetIntersection(l2);
    res.x = intersection.GetX();
    res.y = intersection.GetY();

    return res;
}