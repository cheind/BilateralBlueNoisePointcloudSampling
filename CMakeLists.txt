cmake_minimum_required(VERSION 2.8)

project(BilateralBlueNoisePointcloudSampling)

# Update CMAKE module paths to locate custom finders
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")

# Find Eigen
find_package(Eigen REQUIRED)
add_definitions(${Eigen_DEFINITIONS})
include_directories(${Eigen_INCLUDE_DIRS})

# Setup library
set(BBN_SOURCES
	inc/bbn/differential.h
	inc/bbn/dart_throwing.h
	inc/bbn/scaling.h
	
	src/differential.cpp
	src/dart_throwing.cpp
	src/scaling.cpp
)

include_directories(inc)
add_library(bbn ${BBN_SOURCES})

# Setup tests
add_executable(resample test/io_pointcloud.h test/resample.cpp)
target_link_libraries(resample bbn)