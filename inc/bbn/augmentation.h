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

#ifndef BBN_AUGMENTATION_H
#define BBN_AUGMENTATION_H

#include <Eigen/Dense>
#include <vector>
#include <bbn/eigen_types.h>

namespace bbn {

	/*	Stacks points and normals in a 6 dimensional tuple. Applies weights for points and normals separately which 
		ultimately affect importance of points and normals in sampling. */
	bool stackPointsAndNormalsWeighted(
		const std::vector<Eigen::Vector3f> &points, const std::vector<Eigen::Vector3f> &normals, 
		std::vector<Eigen::Vector6f> &stacked, 
		float pointWeight = 1.f, float normalWeight = 1.f);	
    
}

#endif