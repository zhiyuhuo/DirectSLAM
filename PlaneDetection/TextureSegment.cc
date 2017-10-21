#include "TextureSegment.h"

TextureSegment::TextureSegment(cv::Mat image, int gridX, int gridY)
{
    InitData(image, gridX, gridY);
}

void TextureSegment::InitGaborFilters()
{
    cv::Size ksize(3, 3);
    double pi_value = 3.14159265;
    std::vector<double> sigma = {1, 2, 3, 4, 5};
    std::vector<double> theta = {pi_value * 0.125, pi_value * 0.25, pi_value * 0.375, pi_value * 0.50, 
                                 pi_value * 0.625, pi_value * 0.75, pi_value * 0.875, pi_value * 1.0};
    std::vector<double> lambd = {1};
    std::vector<double> gamma = {0.02};
    int count = 0;
    for (int s = 0; s < sigma.size(); s++) {
        for (int t = 0; t < theta.size(); t++) {
            for (int l = 0; l < lambd.size(); l++) {
                for (int g = 0; g < gamma.size(); g++) {
                    cv::Mat kGabor = cv::getGaborKernel(ksize, sigma[s], theta[t], lambd[l], gamma[g], 0, CV_32F);
                    mGaborFilters.push_back(cv::Mat());
                    kGabor.convertTo(kGabor, CV_32FC1);
                    mGaborFilters[count++] = kGabor.clone();
                }
            }
        }
    }

    // for (int i = 0; i < mGaborFilters.size(); i++) {
    //     cv::imshow("filter", mGaborFilters[i]);
    //     cv::waitKey(-1);
    // }

}

void TextureSegment::InitData(cv::Mat image, int gridX, int gridY)
{
    mImage = image.clone();
    mGridX = gridX;
    mGridY = gridY;
    mGridNumX = image.cols / mGridX;
    mGridNumY = image.rows / mGridY;
    mGridGaborFeatures.resize(mGridNumY);
    mTextureMap = cv::Mat(mGridNumY, mGridNumX, CV_32SC1);
    mGrayScaleMap = cv::Mat(mGridNumY, mGridNumX, CV_32FC1);
    for (int i = 0; i < mGridNumY; i++) {
        mGridGaborFeatures[i].resize(mGridNumX);
        for (int j = 0; j < mGridNumX; j++) {
            mTextureMap.at<int>(i, j) = i * mGridNumX + j;
        }
    }
    InitGaborFilters();
}

void TextureSegment::ComputeGridFeatures()
{
    // cv::imshow("mImage", mImage);
    // std::cout << mGridX << " " << mGridY << " " << mGridNumX << " " << mGridNumY << std::endl;
    // std::cout << mGridGaborFeatures.size() << " " << mGridGaborFeatures[0].size() << std::endl;
    for (int v = 0; v <= mImage.rows - mGridY; v+=mGridY) {
        for (int u = 0; u <= mImage.cols - mGridX; u+=mGridX) {
            // std::cout << v << " " << u << " " << v/mGridY << " " << u/mGridX << std::endl;
            cv::Mat patch = mImage(cv::Rect(u, v, mGridX, mGridY));
            // cv::imshow("patch", patch);
            cv::Mat f = ComputeAGridFeature(patch);
            mGridGaborFeatures[v/mGridY][u/mGridX] = f.clone();

            mGrayScaleMap.at<float>(v/mGridY, u/mGridX) = (cv::mean(patch)).val[0] * (1/255.);
            // cv::waitKey(-1);
        }
    }
    ConnectSimilarGrids();
}

cv::Mat TextureSegment::ComputeAGridFeature(cv::Mat img)
{
    cv::Mat res(mGaborFilters.size(), 1, CV_32FC1);
    float* pres = res.ptr<float>(0);

    cv::Mat input;
    img.convertTo(input, CV_32FC1);
    input *= (1/255.);

    int filterX = mGaborFilters[0].cols;
    int filterY = mGaborFilters[0].rows;
    int filterD = filterX * filterY;
    int count = 0;
    for (int v = 0; v < img.rows-filterY; v++) {
        for (int u = 0; u < img.cols-filterX; u++) {
            for (int i = 0; i < mGaborFilters.size(); i++) {
                cv::Mat win = input(cv::Rect(u, v, filterX, filterY));
                float* pwin  = win.ptr<float>(0);
                float* pfilter = mGaborFilters[i].ptr<float>(0);
                float  response = 0.;
                for (int j = 0; j < filterD; j++) {
                    response += pwin[j] * pfilter[j];
                }
                pres[i] += response;      
            } 
            count ++;
        }
    }

    for (int i = 0; i < res.rows; i++) {
        pres[i] /= count;
        // std::cout << std::setprecision(3) << pres[i] << " ";
    }   // std::cout << std::endl;

    // std::cout << res.t() << std::endl;
    return res;
}

cv::Mat TextureSegment::ConnectSimilarGrids()
{
    float Threshold = 8.0;
    int IterMax = 30;

    bool ifConverge = false;
    // for (int y = 1; y < mGridNumY-1; y++) {
    //     for (int x = 1; x < mGridNumX-1; x++) {
    //         std::cout << x << ", " << y << "    "
    //                 << cv::norm(mGridGaborFeatures[y][x], mGridGaborFeatures[y+1][x]) << " "
    //                 << cv::norm(mGridGaborFeatures[y][x], mGridGaborFeatures[y-1][x]) << " "
    //                 << cv::norm(mGridGaborFeatures[y][x], mGridGaborFeatures[y][x+1]) << " "
    //                 << cv::norm(mGridGaborFeatures[y][x], mGridGaborFeatures[y][x-1]) << std::endl;        
    //     }
    // }

    // generate 12 seeds aroung the center of the view
    std::vector<cv::Point2i> seeds = {  cv::Point2i(mGridNumX/2-7, mGridNumY/2-5), 
                                        cv::Point2i(mGridNumX/2-2, mGridNumY/2-5),
                                        cv::Point2i(mGridNumX/2+2, mGridNumY/2-5),
                                        cv::Point2i(mGridNumX/2+7, mGridNumY/2-5),
                                        cv::Point2i(mGridNumX/2-7, mGridNumY/2),
                                        cv::Point2i(mGridNumX/2-2, mGridNumY/2),
                                        cv::Point2i(mGridNumX/2+2, mGridNumY/2),
                                        cv::Point2i(mGridNumX/2+7, mGridNumY/2),
                                        cv::Point2i(mGridNumX/2-7, mGridNumY/2+5),
                                        cv::Point2i(mGridNumX/2-2, mGridNumY/2+5),
                                        cv::Point2i(mGridNumX/2+2, mGridNumY/2+5),
                                        cv::Point2i(mGridNumX/2+7, mGridNumY/2+5)  };

    cv::Mat checkMap = -cv::Mat::ones(mGridNumY, mGridNumX, CV_32SC1);

    int classID = 0;
    std::vector<int> classIDSeeds(seeds.size(), -1);
    classIDSeeds[0] = 0;
    for (int i = 1; i < seeds.size(); i++) {
        for (int j = 0; j < i; j++) {
            float distGabor = cv::norm(mGridGaborFeatures[seeds[j].y][seeds[j].x], mGridGaborFeatures[seeds[i].y][seeds[i].x]);
            float distColor = fabs(mGrayScaleMap.at<float>(seeds[j].y,seeds[j].x) - mGrayScaleMap.at<float>(seeds[i].y,seeds[i].x));
            std::cout << i << " " << j << ": " << distGabor << " " << distColor << std::endl;
            if ( 
                distGabor < Threshold
                && distColor < 0.1 
            ) {             
                classIDSeeds[i] = classIDSeeds[j];
            }
        }
        if (classIDSeeds[i] < 0) {
            classIDSeeds[i] = ++classID;
        }
    }
    mTextureID = classID;

    for (int i = 0; i < classIDSeeds.size(); i++) {
        // std::cout << classIDSeeds[i] << " ";
        checkMap.at<int>(seeds[i].y, seeds[i].x) = classIDSeeds[i];
    }   // std::cout << std::endl;
    // std::cout << checkMap << std::endl;

    int dx[4] = {-1, 1,  0, 0};
    int dy[4] = { 0, 0, -1, 1};

    int Iter = 0;
    for (; Iter < 30; Iter++) {
        bool ifChange = false;

        for (int y = 1; y < mGridNumY-1; y++) {
            for (int x = 1; x < mGridNumX-1; x++) {
                if (checkMap.at<int>(y,x) < 0) {
                    float minValue = 999999.;
                    int   minIndex = -1;
                    for (int i = 0; i < 4; i++) {
                        if (checkMap.at<int>(y+dy[i],x+dx[i]) >= 0) {

                            float distGabor = cv::norm(mGridGaborFeatures[y][x] - mGridGaborFeatures[y+dy[i]][x+dx[i]]);
                            float distColor = fabs(mGrayScaleMap.at<float>(y,x) - mGrayScaleMap.at<float>(y+dy[i],x+dx[i]));
                            // std::cout << distGabor << " " << distColor << std::endl;
                            if (distGabor < minValue && distColor < 0.05) {
                                minValue = distGabor;
                                minIndex = checkMap.at<int>(y+dy[i],x+dx[i]);
                            }

                        }
                    }

                    if ( minValue < 2*Threshold )
                    {
                        checkMap.at<int>(y,x) = minIndex;
                        ifChange = true;
                        break;
                    }
                }
            }
        }

        if (!ifChange) {
            break;
        }
    } 

    // std::cout << "Iter: " << Iter << std::endl;
    // std::cout << checkMap << std::endl;
    checkMap.copyTo(mTextureMap);

    #ifndef __ANDROID__
    cv::Mat imgShow;
    cv::cvtColor(mImage, imgShow, CV_GRAY2BGR);
    std::vector<cv::Scalar> colours(classID+1);
    int step = 255 / (classID+1);
    for (int i = 0; i <= classID; i++) {
        colours[i] = cv::Scalar(step*i, 128-step*i*std::pow(-1,i), 128+step*i*std::pow(-1,i));
    }
    for (int y = 1; y < mGridNumY-1; y++) {
        for (int x = 1; x < mGridNumX-1; x++) {
            if (checkMap.at<int>(y, x) >= 0) {
                cv::circle(imgShow, cv::Point2i(x*mGridX+mGridX/2, y*mGridY+mGridY/2), 
                                    std::min(mGridX, mGridY)/8, colours[checkMap.at<int>(y, x)], 1);
            }
        }
    }
    cv::imshow("image segmented", imgShow);
    #endif
}

void TextureSegment::GetTextureRegions()
{
    mTextureRegions.resize(mTextureID+1);
    mTextureRegionPortions.resize(mTextureID+1, 0);
    for (int i = 0; i < mTextureRegions.size(); i++) {
        mTextureRegions[i] = std::pair<cv::Point2f, cv::Point2f>( cv::Point2f(mImage.cols-1, mImage.rows-1), cv::Point2f(0, 0) );
    }

    int id;
    float u,v;
    for (int y = 0; y < mTextureMap.rows; y++) {
        for (int x = 0; x < mTextureMap.cols; x++) {
            if (mTextureMap.at<int>(y, x) >= 0) {
                id = mTextureMap.at<int>(y, x);
                u = x * mGridX + mGridX/2;
                v = y * mGridY + mGridY/2;
                
                if (u < mTextureRegions[id].first.x) 
                    mTextureRegions[id].first.x = u;
                if (u > mTextureRegions[id].second.x) 
                    mTextureRegions[id].second.x = u;
                if (v < mTextureRegions[id].first.y) 
                    mTextureRegions[id].first.y = v;
                if (v > mTextureRegions[id].second.y) 
                    mTextureRegions[id].second.y = v;

                mTextureRegionPortions[id]++;
            }
        }
    }

    for (int i = 0; i < mTextureRegions.size(); i++) {
        if (mTextureRegionPortions[i] < mGridNumX/3*mGridNumY/3)
        {
            mTextureRegionPortions[i] = 0;
            continue;
        }

        mTextureRegionPortions[i] /= ((mTextureRegions[i].second.x-mTextureRegions[i].first.x) / mGridX + 1) *
                                     ((mTextureRegions[i].second.y-mTextureRegions[i].first.y) / mGridY + 1); 

        std::cout << "textureID " << i << ": " << mTextureRegions[i].first 
                                               << mTextureRegions[i].second 
                                               << mTextureRegionPortions[i]
                                               << std::endl;
    }
}