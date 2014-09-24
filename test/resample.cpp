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
#include <bbn/augmentation.h>
#include <bbn/bruteforce_locator.h>
#include <bbn/hashtable_locator.h>

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

	// Normalize input
	Eigen::Affine3f undoRotTrans(Eigen::Affine3f::Identity()), undoScale(Eigen::Affine3f::Identity());
	if (!bbn::normalizeOrientationAndTranslation(points, normals, undoRotTrans)) {
        std::cerr << "Failed to normalize position / orientation of pointcloud" << std::endl;
    }

	if (!bbn::normalizeSize(points, normals, undoScale)) {
		std::cerr << "Failed to normalize size of pointcloud" << std::endl;
	}
    
	// Resample by dart throwing.
	std::vector<Eigen::Vector6f> stackedInput, stackedOutput;
	std::vector<size_t> stackedOutputIds;
	bbn::stackPointsAndNormalsWeighted(points, normals, stackedInput, 1.0f, 0.05f);

	// Locator to be used.
	typedef bbn::HashtableLocator<Eigen::Vector6f> LocatorType;

	bbn::AugmentedDartThrowing adt;
	adt.setConflictRadius(0.01f);
	adt.setRandomSeed(10);
    
	if (!adt.resample<LocatorType>(stackedInput, stackedOutputIds)) {
        std::cerr << "Failed to throw darts." << std::endl;
    }

	// Create output
	std::vector<Eigen::Vector3f> resampledPoints, resampledNormals;
	for (size_t i = 0; i < stackedOutputIds.size(); ++i) {
		resampledPoints.push_back(points[stackedOutputIds[i]]);
		resampledNormals.push_back(normals[stackedOutputIds[i]]);
	}
    
	// Restore original dimensions.
	Eigen::Affine3f undoCombined = undoRotTrans * undoScale;
	if (!bbn::applyTransform(resampledPoints, resampledNormals, undoCombined)) {
        std::cerr << "Failed to undo pointcloud scaling" << std::endl;
    }
    
	// Save result.
    if (!savePointcloudToXYZFile(argv[2], resampledPoints, resampledNormals)) {
        std::cerr << "Failed to load pointcloud from file" << std::endl;
    }
    
    
    
    return 0;
}