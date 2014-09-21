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

#ifndef BBN_SCALING_H
#define BBN_SCALING_H

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>

namespace bbn {
    
    /** Scale oriented pointcloud so that they are contained within a box of unit size. 
        Transforms the points in place and returns the inverse scaling transformation to restore the original pointcloud. */
    bool scalePointcloudToUnitBox(std::vector<Eigen::Vector3f> &points, std::vector<Eigen::Vector3f> &normals, Eigen::Affine3f &invTransform);
    
    /** Restores the pointcloud. */
    bool restoreScaledPointcloud(std::vector<Eigen::Vector3f> &points, std::vector<Eigen::Vector3f> &normals, const Eigen::Affine3f &invTransform);
}

#endif