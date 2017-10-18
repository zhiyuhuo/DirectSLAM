cmake_minimum_required(VERSION 3.4.1)
project(DirectSLAM)

message(STATUS "Build Not for Android Platform")

#add_definitions( -DNO_TIMING=1 )   # change this for timing switch
#add_definitions( -DNO_LOG=1 )      # change this for log switch

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/" "${PROJECT_SOURCE_DIR}/cmake_modules")
SET(CMAKE_BUILD_TYPE Release)
SET(MARCH native)
MESSAGE(STATUS "Build type: " ${CMAKE_BUILD_TYPE})
MESSAGE(STATUS "March type: " ${MARCH})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -g -O2 -ffast-math -fopenmp -mavx -mtune=native -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -O2 -ffast-math -fopenmp -mavx -mtune=native -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_definitions(-DCOMPILEDWITHC11)
message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
add_definitions(-DCOMPILEDWITHC0X)
message(STATUS "Using flag -std=c++0x.")
else()
message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
endif()

# find required packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Pangolin REQUIRED)

# include dir
include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/thirdparty/
${OpenCV_INCLUDE_DIRS}
${CERES_INCLUDE_DIR}
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
	src/Log.cc
	src/CameraIntrinsic.cc
	src/Frame.cc
	src/Optimization.cc
	src/Viewer.cc
	src/Statistic.cc
	src/PlaneDetector.cc
	src/TextureSegment.cc
	src/GeometryR.cc
)

add_library(PlaneDetection SHARED
	PlaneDetection/Frame.cc
	PlaneDetection/CameraIntrinsic.cc
	PlaneDetection/Log.cc
	PlaneDetection/TextureSegment.cc
	PlaneDetection/Viewer.cc
	PlaneDetection/PlaneDetector.cc
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBRARIES}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
ceres
)

target_link_libraries(PlaneDetection
${OpenCV_LIBRARIES}
${Pangolin_LIBRARIES}
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

link_directories(${CERES_LIBRARY})
link_directories("/home/rokid/Loitor_VI_Sensor_SDK_V1.3.1/SDK/src")

add_executable(appTestPlaneDetection app/appTestPlaneDetection.cc)
target_link_libraries(appTestPlaneDetection loitorusbcam loitorimu usb-1.0 PlaneDetection )

add_executable(appTestHomographyTextureBayesPlaneRecognization app/appTestHomographyTextureBayesPlaneRecognization.cc)
target_link_libraries(appTestHomographyTextureBayesPlaneRecognization loitorusbcam loitorimu usb-1.0 ${PROJECT_NAME} )

add_executable(patent app/patent.cc)
target_link_libraries(patent ${PROJECT_NAME} )
    
