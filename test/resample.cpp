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

typedef bbn::TaskTraits<float, 3, 3> R3Traits;
typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > ArrayOfVector;
typedef bbn::Stacking<Eigen::Vector3f, Eigen::Vector3f> Stacker;

class PointSampler {
public:	
	PointSampler(ArrayOfVector &points, ArrayOfVector &normals, Stacker &s)
		:_points(points), _normals(normals), _s(s)
	{
		_sampleIndices.reserve(_points.size());
		for (size_t i = 0; i < _points.size(); ++i)
			_sampleIndices.push_back(i);
		std::random_shuffle(_sampleIndices.begin(), _sampleIndices.end());
	}

	R3Traits::Vector operator()(void) {
		size_t pointId = _sampleIndices[_index++];
		return _s(_points[pointId], _normals[pointId]);				
	}

private:
	std::vector<size_t> _sampleIndices;
	ArrayOfVector &_points, &_normals;
	Stacker &_s;
	size_t _index;
};

int main(int argc, const char **argv) {
    
    if (argc != 3) {
        std::cerr << "Usage: " << std::endl;
        std::cerr << argv[0] << " <input.xyz> <output.xyz>" << std::endl;
        return -1;
    }
    
	// Load from file.
	ArrayOfVector points, normals;
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
	R3Traits traits;
	bbn::DartThrowing<R3Traits> adt;
	adt.setTaskTraits(traits);
	adt.setConflictRadius(0.01f);
	adt.setMaximumAttempts(points.size());

	Stacker stacker(Stacker::Params(1.0f, 0.05f));

	PointSampler sampler(points, normals, stacker);
	std::vector<R3Traits::Vector> sampled;
    
	if (!adt.resample(sampler, std::back_inserter(sampled))) {
        std::cerr << "Failed to throw darts." << std::endl;
    }

	bbn::HashtableLocator<Eigen::Vector3f> ploc;
	ploc.add(points.begin(), points.end());

	bbn::EnergyMinimization<R3Traits> em;
	em.setTaskTraits(traits);
	em.setKernelSigma(0.03f);
	em.setStepSize(0.45f * 0.005f *0.005f);
	em.setMaximumSearchRadius(0.02f);
	em.minimize(sampled.begin(), sampled.end(), sampled.begin(), [&](R3Traits::VectorLike p) {

		size_t idx;
		float dist2;
		if (!ploc.findClosestWithinRadius(p.topRows(3), 0.1f, idx, dist2))
			return;

		Eigen::Vector3f x = p.topRows(3) - points[idx];
		Eigen::Vector3f xdash = points[idx] + (x - x.dot(normals[idx]) * normals[idx]);

		p = stacker(xdash, normals[idx]);

	}, 3);

	ArrayOfVector resampledPoints, resampledNormals;
	for (size_t i = 0; i < sampled.size(); ++i) {
		resampledPoints.push_back(sampled[i].topRows(3));
		resampledNormals.push_back(sampled[i].bottomRows(3).normalized()); // remove weight.
	}
    
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