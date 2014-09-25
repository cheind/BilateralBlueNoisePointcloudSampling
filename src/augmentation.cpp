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
#include <bbn/stacking.h>

namespace bbn {

	bool stackPointsAndNormalsWeighted(
		const std::vector<Eigen::Vector3f> &points, const std::vector<Eigen::Vector3f> &normals,
		std::vector<Eigen::Vector6f> &stacked,
		float pointWeight, float normalWeight)
	{
		Stacking<Eigen::Vector3f, Eigen::Vector3f> s(pointWeight, normalWeight);

		for (size_t i = 0; i < points.size(); ++i) {
			stacked.push_back(s(points[i], normals[i]));
		}

		return true;
	}
}