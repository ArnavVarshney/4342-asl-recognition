cmake_minimum_required(VERSION 3.27)
project(ncnn_deployment)
add_executable(mytarget ncnn.cpp)

set(OpenCV_DIR "/Users/pregabalin/Documents/Studies/Year_3/Sem2/ELEC4342/project/technical_project/build_opencv/")
set(ncnn_DIR "/Users/pregabalin/Documents/Studies/Year_3/Sem2/ELEC4342/project/technical_project/ncnn/build/install/lib/cmake/ncnn/")

find_package(ncnn REQUIRED)
find_package(OpenCV REQUIRED)



target_link_libraries(mytarget ncnn ${OpenCV_LIBS})