#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <thread>

#include <opencv2/opencv.hpp>
#include <zconf.h>
#include "Frame.h"
#include "PlaneDetector.h"
#include "TextureSegment.h"
#include "Viewer.h"

using namespace std;

bool stopped = false;
struct timeval time_begin;
struct timeval time_end;
double time_difference;

void callback()
{
    stopped = true;
    printf("track local stoped and callback called!\n");
}

int ReadImageFiles(const string& fileName, std::vector<std::string>& imgFiles, std::vector<long>& timeStamps);

int ReadImuData(const string& fileName, std::vector<std::vector<float> >& imuVecs, std::vector<long>& timeStamps);

int ReadTransformOfFrames(const string& fileName, 
                          std::vector<std::vector<float> >& rotations, 
                          std::vector<std::vector<float> >& translations, 
                          std::vector<long>& timeStamps);

std::vector<float> GetImuTSforImage(long imgTimeStamp, const std::vector<std::vector<float> >& imuVecs, const std::vector<long>& timeStamps);



int main(int argc, char** argv)
{
    float fps = 57.5f;

    /*--------Dataset to use------*/
    std::vector<std::string>          imgFiles;
    std::vector<long>                 imgTimeStamps;
    std::vector<std::vector<float> >  imuVecs;
    std::vector<long>                 imuTimeStamps;
    std::vector<std::vector<float> >  resRotVecs;
    std::vector<std::vector<float> >  resTransVecs;
    std::vector<long>                 resTimeStamps;
    
    cv::Mat img_input;
    ReadImageFiles(argv[2], imgFiles, imgTimeStamps);
    ReadImuData(argv[3], imuVecs, imuTimeStamps);
    ReadTransformOfFrames(argv[4], resRotVecs, resTransVecs, resTimeStamps);
    std::cout << imgFiles.size() << ", " << imgTimeStamps.size() << ", " << imuVecs.size() << ", " << imuTimeStamps.size() << ", "
              << resRotVecs.size() << ", " << resTransVecs.size() << ", " << resTimeStamps.size() << std::endl;

    /*-------- SLAM System --------*/
    // TO DO
    CameraIntrinsic* K = new CameraIntrinsic(argv[1]);
    PlaneDetector* pPD = new PlaneDetector(K);
    // Viewer*   pViewer = new Viewer(pHPD);
    // std::thread* ptViewer = new std::thread(&Viewer::run, pViewer);
    // ptViewer->detach();

    char cmd = ' ';
    bool started = false;
    bool callAlgorithm = false;
    
    int nImg = 0;
    
    while (true) {
        img_input = cv::imread(imgFiles[nImg].c_str(), CV_LOAD_IMAGE_GRAYSCALE);
        double timeStamp = double(imgTimeStamps[nImg]);
        std::vector<float> imu = GetImuTSforImage(imgTimeStamps[nImg], imuVecs, imuTimeStamps);

        // if (nImg == 0) {
        //     cv::imshow("img_input", img_input);
        //     TextureSegment* pTextureSeg = new TextureSegment(img_input, 30, 30);
        //     pTextureSeg->ComputePatchFeatures();
        // }

        if (cmd == 's') {
            if (started) {
                printf("The program breaks. You cannot restart the program\n");
            } else {
                started = true;
            }
        } else {                                
            if (started) {

                if (!callAlgorithm && resTransVecs[nImg-1][0] == 0 && resTransVecs[nImg][0] != 0) {
                    pPD->TrackMono(img_input, resRotVecs[nImg], resTransVecs[nImg]);
                    callAlgorithm = true;
                    // cv::destroyWindow("img_input");
                } else {
                    if (callAlgorithm) {
                        pPD->TrackMono(img_input, resRotVecs[nImg], resTransVecs[nImg]);
                    }
                }
                cv::imshow("img_input", img_input);
            } else {
                cv::imshow("img_input", img_input);
            }
        }

        if (cmd == 'q') {
            // pViewer->Stop();
            stopped = true;
            break;
        }

        if (started)
            cmd = cv::waitKey(100);
        else
            cmd = cv::waitKey(-1);
        
        nImg++;
    }

    usleep(100000);

    while (!stopped) {
        printf("wait for stop ! \n");
        usleep(30000);
    }

    return 0;
}

int ReadImageFiles(const string& fileName, std::vector<std::string>& imgFiles, std::vector<long>& timeStamps)
{
    imgFiles.clear();
    timeStamps.clear();
    
    ifstream file;
    file.open(fileName);
    if ( !file.is_open() ) {
        printf("Open dataset %s failed!\n", fileName.c_str());
        return -1;
    }

    // test read the first image
    cv::Mat Frame;
    std::string frameName; 
    if (getline(file, frameName) ) {
        stringstream ss(frameName);
        ss >> frameName;
        Frame = cv::imread(frameName);
        if (Frame.empty()) {
            printf("Open dataset %s failed! Test first image %s failed!\n", fileName.c_str(), frameName.c_str());
            return -1;
        }
        printf("Open dataset %s succeed! Test first image %s succeed!\n", fileName.c_str(), frameName.c_str());
        file.clear();
        file.seekg(0, ios::beg);
    } else {
        printf("Open dataset %s failed! Get line failed!\n", fileName.c_str());
        return -1;
    }
    
    std::cout << "img dataset can be openned!" << std::endl;
    
    while (true) {
        if (getline(file, frameName)) {
            stringstream ss(frameName);

            ss >> frameName;
            
//             Frame = cv::imread(frameName);
//             if (Frame.empty()) {
//                 printf("Read image %s failed!\n", frameName.c_str());
//                 break;
//             }

            int pos_dash = frameName.find_last_of('/');
            int pos_dot = frameName.find_last_of('.');
            string s_timeStamp(frameName.substr(pos_dash + 1, pos_dot - (pos_dash + 1) ) );
            long timeStamp = stol(s_timeStamp.c_str());
            
            // copy to result;
            imgFiles.push_back(frameName);
            timeStamps.push_back(timeStamp);
            // printf("Read image %s at %ld succeed!\n", frameName.c_str(), timeStamp);
            
        } else {
            printf("Reach end of the dataset!\n");
            break;
        }
    }
    
    return 0;
}

int ReadImuData(const string& fileName, std::vector<std::vector<float> >& imuVecs, std::vector<long>& timeStamps)
{
    std::cout << fileName << std::endl;
    imuVecs.clear();
    timeStamps.clear();
    
    ifstream file;
    file.open(fileName);
    if ( !file.is_open() ) {
        printf("Open imu dataset %s failed!\n", fileName.c_str());
        return -1;
    }
    
    while (true) {
        long timeStamp;
        std::vector<float> imu(6);
        std::vector<char>  stop(7);
        string strLine;
        if (getline(file, strLine)) {
            stringstream ss(strLine);

            ss >> stop[0] >> imu[0] >> stop[1] >> imu[1] >> stop[2] >> imu[2] >> stop[3] >> imu[3]
               >> stop[4] >> imu[4] >> stop[5] >> imu[5] >> stop[6] >> timeStamp;
            
            // std::cout << imu[0] << " " << imu[1] << " " << imu[2] << " " << imu[3] << " " << imu[4] << " " << imu[5] << std::endl;
            // copy to result;
            imuVecs.push_back(imu);
            timeStamps.push_back(timeStamp);
            // printf("Read imu has %dDOF at %ld succeed!\n", imu.size(), timeStamp);
            
        } else {
            printf("Reach end of the imu dataset!\n");
            break;
        }
    }
    return 0;
}

std::vector<float> GetImuTSforImage(long imgTimeStamp, const std::vector<std::vector<float> >& imuVecs, const std::vector<long>& timeStamps)
{
    std::vector<float> res;
    
    int n = 1;
    int N = imuVecs.size();
    while (n < N) {
        if (timeStamps[n] > imgTimeStamp)
        {
            res = abs(timeStamps[n]-imgTimeStamp) < abs(timeStamps[n-1]-imgTimeStamp)?   imuVecs[n] : imuVecs[n-1] ;
            break;
        }
        n++;
    }
    
    return res;
}

int ReadTransformOfFrames(const string& fileName, 
                          std::vector<std::vector<float> >& rotations, 
                          std::vector<std::vector<float> >& translations, 
                          std::vector<long>& timeStamps)
{
    rotations.resize(0);
    translations.resize(0);
    timeStamps.resize(0);
    std::vector<float> rotation(9);
    std::vector<float> translation(3);
    long timeStamp = 0;

    ifstream file;
    file.open(fileName);
    if ( !file.is_open() ) {
        printf("Open SLAM results dataset %s failed!\n", fileName.c_str());
        return -1;
    }
    
    while (true) {
        double timeStamp;
        string strLine;
        if (getline(file, strLine)) {
            stringstream ss(strLine);

            ss >> timeStamp  
               >> rotation[0] >> rotation[1] >> rotation[2] 
               >> rotation[3] >> rotation[4] >> rotation[5] 
               >> rotation[6] >> rotation[7] >> rotation[8]
               >> translation[0] >> translation[1] >> translation[2];        
            
            // std::cout << long(timeStamp) << " " 
            //           << rotation[0] << " " << rotation[1] << " " << rotation[2] << " " 
            //           << rotation[3] << " " << rotation[4] << " " << rotation[5] << " "
            //           << rotation[6] << " " << rotation[7] << " " << rotation[8] << " "
            //           << translation[0] << " " << translation[1] << " " << translation[2]
            //           << std::endl;
            // copy to result;
            rotations.push_back(rotation);
            translations.push_back(translation);
            timeStamps.push_back(long(timeStamp));
            
        } else {
            printf("Reach end of the SLAMResult dataset!\n");
            break;
        }
    }
}