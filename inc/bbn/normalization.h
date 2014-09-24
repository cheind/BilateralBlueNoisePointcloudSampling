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

#ifndef BBN_NORMALIZATION_H
#define BBN_NORMALIZATION_H

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>

namespace bbn {

	/* Normalizes the pointclouds position and orientation using PCA. Returns the inverse transform. */
	bool normalizeOrientationAndTranslation(std::vector<Eigen::Vector3f> &points, std::vector<Eigen::Vector3f> &normals, Eigen::Affine3f &invTransform);

	/* Normalizes the pointclouds size through a uniform scaling such that the longest side of the AABB becomes unit length. Assumes normalized rotation/translation.  */
	bool normalizeSize(std::vector<Eigen::Vector3f> &points, std::vector<Eigen::Vector3f> &normals, Eigen::Affine3f &invTransform);

	/* Apply a general affine transformation to a pointcloud. */
	bool applyTransform(std::vector<Eigen::Vector3f> &points, std::vector<Eigen::Vector3f> &normals, const Eigen::Affine3f &t);
   
}

#endif