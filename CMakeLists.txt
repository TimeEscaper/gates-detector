cmake_minimum_required(VERSION 3.10)
project(gates-locator)

set(CMAKE_CXX_STANDARD 11)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( gates-locator main.cpp )
target_link_libraries( gates-locator ${OpenCV_LIBS} )