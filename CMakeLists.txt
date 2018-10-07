cmake_minimum_required(VERSION 3.10)
project(gates-locator)

set(CMAKE_CXX_STANDARD 11)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( gates-locator main.cpp include/ImgprocPipeline.h src/ImgprocPipeline.cpp include/ImgprocLogicUtil.h src/ImgprocLogicUtil.cpp src/GatesDetector.cpp include/GatesDetector.h src/GatesDescriptor.cpp include/GatesDescriptor.h)
target_link_libraries( gates-locator ${OpenCV_LIBS} )