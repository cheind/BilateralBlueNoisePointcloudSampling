// This file is part of BilateralBlueNoisePointcloudSampling (BBNPS).
//
// Copyright (C) 2014 Christoph Heindl <christoph.heindl@gmail.com>
//
// BBNPS is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// BBNPS is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#include <Eigen/Dense>
#include "io_pointcloud.h"
#include <iostream>

int main(int argc, const char **argv) {
    
    if (argc != 3) {
        std::cerr << "Usage: " << std::endl;
        std::cerr << argv[0] << " <input.xyz> <output.xyz>" << std::endl;
        return -1;
    }
    
    std::vector<Eigen::Vector3f> points, normals;
    if (!loadPointcloudFromXYZFile(argv[1], points, normals)) {
        std::cerr << "Failed to load pointcloud from file" << std::endl;
    }
    
    if (!savePointcloudToXYZFile(argv[2], points, normals)) {
        std::cerr << "Failed to load pointcloud from file" << std::endl;
    }
    
    
    
    return 0;
}