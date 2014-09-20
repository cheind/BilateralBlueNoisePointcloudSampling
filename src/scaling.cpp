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

#include <bbn/scaling.h>
#include <iostream>


namespace bbn {
    
    bool scalePointcloudToUnitBox(std::vector<Eigen::Vector3f> &points, std::vector<Eigen::Vector3f> &normals, Eigen::Affine3f &invTransform)
    {
        if (points.empty())
            return false;
        
        // Perform PCA on input to determine a canoncial coordinate frame for the given point cloud.
        Eigen::Matrix3Xf::MapType pointsInMatrix(points.at(0).data(), 3, static_cast<int>(points.size()));
        const Eigen::Vector3f centroid = pointsInMatrix.rowwise().mean();
        pointsInMatrix = pointsInMatrix.colwise() - centroid;
        
        const Eigen::Matrix3f cov = pointsInMatrix * pointsInMatrix.transpose();
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(cov);
        const Eigen::Matrix3f rot = eig.eigenvectors().transpose();
        for (size_t i = 0; i < points.size(); ++i) {
            points[i] = rot * points[i];
            normals[i] = rot * normals[i];
        }
        
        // Now that we have the pointcloud in a canonical frame, calculate axis aligned bounding box of input
        Eigen::AlignedBox3f aabb;
        for (size_t i = 0; i < points.size(); ++i) {
            aabb.extend(points[i]);
        }
        
        // Calculate isotropic scaling to that the longest side becomes unit length
        const float s = 1.f / aabb.diagonal().maxCoeff();
        
        for (size_t i = 0; i < points.size(); ++i) {
            points[i] *= s;
        }
        
        // Assemble inverse transform.
        invTransform = Eigen::Affine3f::Identity();
        invTransform = invTransform.scale(s).rotate(rot).translate(-centroid); // applied in right to left order.
        invTransform = invTransform.inverse();
            
        return true;
    }
}