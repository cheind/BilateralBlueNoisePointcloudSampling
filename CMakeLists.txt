
# User options
set(BBN_VERBOSE TRUE)

cmake_minimum_required(VERSION 2.8)
project(BilateralBlueNoisePointcloudSampling)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")

# Handle user options.
if (BBN_VERBOSE)
	add_definitions("-DBBN_VERBOSE_OUTPUT")
endif()

# Find Eigen
find_package(Eigen REQUIRED)
add_definitions(${Eigen_DEFINITIONS})
include_directories(${Eigen_INCLUDE_DIRS})

# Setup library
set(BBN_SOURCES
	inc/bbn/eigen_types.h
	inc/bbn/scaling.h
	inc/bbn/dart_throwing.h	
	inc/bbn/util.h
	inc/bbn/bruteforce_locator.h
	inc/bbn/hashtable_locator.h
	inc/bbn/augmentation.h
	
	src/scaling.cpp
	src/augmentation.cpp
)

include_directories(inc)
add_library(bbn ${BBN_SOURCES})

# Setup tests
add_executable(resample test/io_pointcloud.h test/resample.cpp)
target_link_libraries(resample bbn)