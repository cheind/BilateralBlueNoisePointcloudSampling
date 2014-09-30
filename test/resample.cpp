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

#include <bbn/task_traits.h>
#include <bbn/normalization.h>
#include <bbn/dart_throwing.h>
#include <bbn/energy_minimization.h>

int main(int argc, const char **argv) {
    
    if (argc != 3) {
        std::cerr << "Usage: " << std::endl;
        std::cerr << argv[0] << " <input.xyz> <output.xyz>" << std::endl;
        return -1;
    }
    
	// Load from file.
	std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > points, normals;
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

	typedef bbn::TaskTraits< Eigen::Vector3f, Eigen::Vector3f, true> R3Traits;

   
	// Resample by dart throwing.
	std::vector<size_t> outputIds;

	bbn::DartThrowing<R3Traits> adt;
	adt.setConflictRadius(0.01f);
	adt.setRandomSeed(10);
    
	if (!adt.resample(points, normals, outputIds)) {
        std::cerr << "Failed to throw darts." << std::endl;
    }

	// Create output
	R3Traits::ArrayOfPositionVector resampledPoints, resampledNormals;
	for (size_t i = 0; i < outputIds.size(); ++i) {
		resampledPoints.push_back(points[outputIds[i]]);
		resampledNormals.push_back(normals[outputIds[i]]);
	}

	R3Traits::PositionLocator ploc;
	ploc.add(points.begin(), points.end());

	bbn::EnergyMinimization<R3Traits> em;
	em.setKernelSigma(0.01f);
	em.setStepSize(0.45f * 0.01f * 0.01f);
	em.setMaximumSearchRadius(0.2f);
	em.minimize(resampledPoints, resampledNormals, resampledPoints, resampledNormals, [&](R3Traits::PositionVector &p, R3Traits::FeatureVector &f) {
		size_t idx;
		float dist2;
		if (!ploc.findClosestWithinRadius(p, 0.1f, idx, dist2))
			return;

		p = points[idx];
		f = normals[idx];

	}, 10);
    
	// Restore original dimensions.
	Eigen::Affine3f undoCombined = undoRotTrans;// * undoScale;
	if (!bbn::applyTransform(resampledPoints, resampledNormals, undoCombined)) {
        std::cerr << "Failed to undo pointcloud scaling" << std::endl;
    }
    
	// Save result.
    if (!savePointcloudToXYZFile(argv[2], resampledPoints, resampledNormals)) {
        std::cerr << "Failed to load pointcloud from file" << std::endl;
    }
    
    return 0;
}