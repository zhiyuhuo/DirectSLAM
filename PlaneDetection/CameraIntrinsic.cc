#include "CameraIntrinsic.h"
#include "Log.h"

CameraIntrinsic::CameraIntrinsic(float _cx, float _cy, float _fx, float _fy, float _k1,
                float _k2, float _k3, int _width, int _height, float _FPS)
        : cx(_cx), cy(_cy), fx(_fx), fy(_fy), k1(_k1), k2(_k2), k3(_k3), width(_width), height(_height), FPS(_FPS)
{
    Log_info("=== Camera Intrinsic ===");
    Log_info("cx: {}", cx);
    Log_info("cy: {}", cy);
    Log_info("fx: {}", fx);
    Log_info("fy: {}", fy);
    Log_info("k1: {}", k1);
    Log_info("k2: {}", k2);
    Log_info("k3: {}", k3);
    Log_info("width: {}", width);
    Log_info("height: {}", height);
    Log_info("FPS: {}", FPS);

}

CameraIntrinsic::CameraIntrinsic(const std::string& filename) {
    loadFromFile(filename);
}

void CameraIntrinsic::loadFromFile(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        Log_error("Open Camera Intrinsic file {} failed, exit!", filename);
        exit(0);
    }

    std::string line;
    getline(file, line);
    std::stringstream ss(line);
    ss >> fx >> fy >> cx >> cy >> k1 >> k2 >> k3 >> width >> height >> FPS;

    Log_info("=== Camera Intrinsic ===");
    Log_info("cx: {}", cx);
    Log_info("cy: {}", cy);
    Log_info("fx: {}", fx);
    Log_info("fy: {}", fy);
    Log_info("k1: {}", k1);
    Log_info("k2: {}", k2);
    Log_info("k3: {}", k3);
    Log_info("width: {}", width);
    Log_info("height: {}", height);
    Log_info("FPS: {}", FPS);

}
