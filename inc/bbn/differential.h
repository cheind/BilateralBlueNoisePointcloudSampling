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

#ifndef BBN_DIFFERENTIAL_H
#define BBN_DIFFERENTIAL_H

#include <Eigen/Dense>

namespace bbn {
    
    /** Classic differential based on point position only. */
    class PositionalDifferential {
    public:
        inline Eigen::Vector2f::Scalar operator()(const Eigen::Vector3f &p0, const Eigen::Vector3f &n0,
                                                  const Eigen::Vector3f &p1, const Eigen::Vector3f &n1) const
        {
            return (p1 - p0).norm();
        }
    };
    
    /** Calculates the augmentative bilateral differential as described in section 3.1 */
    class BilateralAugmentativeDifferential {
    public:
        inline BilateralAugmentativeDifferential()
        :_sigma(1 / 25.f)
        {}
        
        inline BilateralAugmentativeDifferential(float normalSigma)
        :_sigma(1.f / normalSigma)
        {}
        
        inline Eigen::Vector2f::Scalar operator()(const Eigen::Vector3f &p0, const Eigen::Vector3f &n0,
                                                  const Eigen::Vector3f &p1, const Eigen::Vector3f &n1) const
        {
            
            return Eigen::Vector2f((p1 - p0).norm(), (n1.dot(-n0) * 0.5f + 0.5f) * _sigma).norm();
        }
        
    private:
        float _sigma;
    };
    
}

#endif