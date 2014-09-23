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
#include <bbn/util.h>
#include <bbn/eigen_types.h>

namespace bbn {
    
    /** Resample through dart throwing. */    
    class AugmentedDartThrowing {
    public:
        /** Default constructor. */
		AugmentedDartThrowing()
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

		/* Set random seed for sampling. */
		void setRandomSeed(unsigned int s) {
			srand(s);
		}
        
        /** Resample input point cloud. */
		template<class Locator>
        bool resample(const std::vector<Eigen::Vector6f> &source,					  
					  std::vector<size_t> &outputIds)
        {
			if (source.empty())
                return false;

			
            // Build a random array of sample indices
            std::vector<size_t> sampleIndices;
			sampleIndices.reserve(source.size());
			for (size_t i = 0; i < source.size(); ++i)
                sampleIndices.push_back(i);
            std::random_shuffle(sampleIndices.begin(), sampleIndices.end());
                      
            // Loop over samples and try to add one after another.
			//Locator loc(100);
			Locator loc;
			outputIds.clear();
            
            int attempt = 0;            
			size_t id = 0;
            while (id < sampleIndices.size() && attempt < _n) {
				size_t pointId = sampleIndices[id];
				const Eigen::Vector6f &p = source[pointId];
               
				if (!loc.findAnyWithinRadius(p, _conflictRadius)) {
					loc.add(p);
					outputIds.push_back(pointId);
					attempt = 0;
                } else {
                    attempt++;
                }
                
                if (id % 5000 == 0) {
                    BBN_LOG("Processed %.2f - Generated %d of possible %d output samples\n",
                            (float)id / source.size() * 100, (int)outputIds.size(), (int)id);
                }
                ++id;
            }
            
            if (attempt >= _n) {
                BBN_LOG("Failed to generate new samples, giving up.\n");
            }
            
            return true;
        }
        
    private:
        
        float _conflictRadius;
        int _n;
    };
}

#endif