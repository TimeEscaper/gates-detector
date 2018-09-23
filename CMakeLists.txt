cmake_minimum_required(VERSION 3.10)
project(gates-locator)

set(CMAKE_CXX_STANDARD 11)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( gates-locator main.cpp include/ImgprocPipeline.h src/ImgprocPipeline.cpp include/ImgprocLogic.h src/ImgprocLogic.cpp include/Utils.h src/Utils.cpp)
target_link_libraries( gates-locator ${OpenCV_LIBS} )