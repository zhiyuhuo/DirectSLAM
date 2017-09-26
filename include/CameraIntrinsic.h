#ifndef CAMERAINTRINSIC_H
#define CAMERAINTRINSIC_H


#include <iostream>
#include <fstream>
#include <sstream>

class CameraIntrinsic {

public:
    float fx;
    float fy;
    float cx;
    float cy;
    float k1;
    float k2;
    float p1;
    float p2;
    float k3;
    float FPS;

    int width;
    int height;

    CameraIntrinsic() = default;
    CameraIntrinsic(float _cx, float _cy, float _fx, float _fy, float _k1, float _k2, float _k3,
        int _width, int _height, float _FPS);
    explicit CameraIntrinsic(const std::string& fileName);
    void loadFromFile(const std::string& fileName);

};

#endif