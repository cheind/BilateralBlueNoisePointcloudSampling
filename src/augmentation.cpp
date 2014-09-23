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

#include <bbn/augmentation.h>

namespace bbn {

	bool stackPointsAndNormalsWeighted(
		const std::vector<Eigen::Vector3f> &points, const std::vector<Eigen::Vector3f> &normals,
		std::vector<Eigen::Vector6f> &stacked,
		float pointWeight, float normalWeight)
	{
		for (size_t i = 0; i < points.size(); ++i) {
			Eigen::Vector6f v;
			v.block<3, 1>(0, 0) = points[i] * pointWeight;
			v.block<3, 1>(3, 0) = normals[i] * normalWeight;
			stacked.push_back(v);
		}

		return true;
	}
}