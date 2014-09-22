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
#include <iostream>
#include "io_pointcloud.h"

#include <bbn/scaling.h>
#include <bbn/dart_throwing.h>
#include <bbn/differential.h>


int main(int argc, const char **argv) {
    
    if (argc != 3) {
        std::cerr << "Usage: " << std::endl;
        std::cerr << argv[0] << " <input.xyz> <output.xyz>" << std::endl;
        return -1;
    }
    
	// Load from file.
    std::vector<Eigen::Vector3f> points, normals;
    if (!loadPointcloudFromXYZFile(argv[1], points, normals)) {
        std::cerr << "Failed to load pointcloud from file" << std::endl;
    }

	// Rescale point cloud.
    Eigen::Affine3f rescale;
    if (!bbn::scalePointcloudToUnitBox(points, normals, rescale)) {
        std::cerr << "Failed to rescale pointcloud" << std::endl;
    }
    
	// Resample by dart throwing.
    std::vector<Eigen::Vector3f> resampledPoints, resampledNormals;    
    bbn::DartThrowing dt;
    dt.setConflictRadius(0.1f);
    
	if (!dt.resample(points, normals, resampledPoints, resampledNormals, bbn::BilateralAugmentativeDifferential(0.5))) {
        std::cerr << "Failed to throw darts." << std::endl;
    }
    
	// Restore original dimensions.
    if (!bbn::restoreScaledPointcloud(resampledPoints, resampledNormals, rescale)) {
        std::cerr << "Failed to undo pointcloud scaling" << std::endl;
    }
    
	// Save result.
    if (!savePointcloudToXYZFile(argv[2], resampledPoints, resampledNormals)) {
        std::cerr << "Failed to load pointcloud from file" << std::endl;
    }
    
    
    
    return 0;
}