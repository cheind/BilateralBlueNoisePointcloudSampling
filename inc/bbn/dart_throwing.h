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

#ifndef BBN_DART_THROWING_H
#define BBN_DART_THROWING_H

#include <Eigen/Dense>
#include <vector>
#include <time.h>
#include <bbn/util.h>

namespace bbn {
    
    /** Resample through dart throwing. */    
    class DartThrowing {
    public:
        /** Default constructor. */
        DartThrowing()
        : _conflictRadius(0.01f), _n(100000)
        {}
        
        
        /** Set the conflict radius that determines the resampling resolution. */
        void setConflictRadius(float r) {
            _conflictRadius = r * r;
        }
        
        /** Resampling stops after n consecutive samples failed to contribute. */
        void setMaximumAttempts(int n) {
            _n = n;
        }
        
        /** Resample input point cloud. */
		template<class BilateralDifferential>
        bool resample(const std::vector<Eigen::Vector3f> &sourcePoints, const std::vector<Eigen::Vector3f> &sourceNormals,
                      std::vector<Eigen::Vector3f> &outputPoints, std::vector<Eigen::Vector3f> &outputNormals,
					  const BilateralDifferential &bd)
        {
            if (sourcePoints.empty())
                return false;
            
            // Build a random array of sample indices
            std::vector<size_t> sampleIndices;
            sampleIndices.reserve(sourcePoints.size());
            for (size_t i = 0; i < sourcePoints.size(); ++i)
                sampleIndices.push_back(i);
            
            srand(unsigned(time(NULL)));
            std::random_shuffle(sampleIndices.begin(), sampleIndices.end());
            
            outputPoints.clear();
            outputNormals.clear();
            
            // Loop over samples and try to add one after another.
            
            int attempt = 0;
            size_t id = 0;
            while (id < sampleIndices.size() && attempt < _n) {
                const Eigen::Vector3f &p = sourcePoints[sampleIndices[id]];
                const Eigen::Vector3f &n = sourceNormals[sampleIndices[id]];
                
                if (!isInConflict(p, n, outputPoints, outputNormals, bd)) {
                    outputPoints.push_back(p);
                    outputNormals.push_back(n);
                    attempt = 0;
                } else {
                    attempt++;
                }
                
                if (id % 5000 == 0) {
                    BBN_LOG("Processed %.2f - Generated %d of possible %d output samples\n",
                            (float)id / sourcePoints.size() * 100, (int)outputPoints.size(), (int)id);
                }
                ++id;
            }
            
            if (attempt >= _n) {
                BBN_LOG("Failed to generate new samples, giving up.\n");
            }
            
            return true;
        }
        
    private:
        
        /** Test if the given sample is in conflict with the previous ones. */
		template<class BilateralDifferential>
        bool isInConflict(const Eigen::Vector3f &p, const Eigen::Vector3f &n,
                          const std::vector<Eigen::Vector3f> &previousPoints, const std::vector<Eigen::Vector3f> &previousNormals,
						  const BilateralDifferential &bd) const
        {
            for (size_t i = 0; i < previousPoints.size(); ++i) {
                const float d = bd(p, n, previousPoints[i], previousNormals[i]);
                if (d < _conflictRadius) {
                    return true;
                }
            }
            
            return false;
        }
        
        float _conflictRadius;
        int _n;
    };
}

#endif