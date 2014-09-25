
# Copyright (C) 2014 Christoph Heindl <christoph.heindl@gmail.com>
#
# BBNPS is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or any later version.
#
# BBNPS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# User options
set(BBN_VERBOSE TRUE)

# ---------------------------------------------

cmake_minimum_required(VERSION 2.8)
project(BilateralBlueNoisePointcloudSampling)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")

# Dependencies
find_package(Eigen REQUIRED)
add_definitions(${Eigen_DEFINITIONS})
include_directories(${Eigen_INCLUDE_DIRS})

# Compiler specifics
if (MSVC)
	add_definitions("-D_CRT_SECURE_NO_WARNINGS")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
	set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++")
endif()

# Handle user options.
if (BBN_VERBOSE)
	add_definitions("-DBBN_VERBOSE_OUTPUT")
endif()

# Setup library
set(BBN_SOURCES
	inc/bbn/eigen_types.h
	inc/bbn/task_traits.h
	inc/bbn/meta.h
	inc/bbn/util.h
	inc/bbn/stacking.h
	inc/bbn/bruteforce_locator.h
	inc/bbn/hashtable_locator.h	
	inc/bbn/normalization.h
	inc/bbn/dart_throwing.h	
	inc/bbn/energy_minimization.h	

	src/normalization.cpp
)

include_directories(inc)
add_library(bbn ${BBN_SOURCES})

# Setup tests
add_executable(resample test/io_pointcloud.h test/resample.cpp)
target_link_libraries(resample bbn)