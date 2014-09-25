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
#include <bbn/task_traits.h>
#include <bbn/util.h>

namespace bbn {
    
    /** Resample by dart throwing. */    
	template<class TaskTraitsType>
    class DartThrowing {
    public:
        /** Default constructor. */
		DartThrowing()
        : _conflictRadius(0.01f), _n(100000)
        {}        
        
        /** Set the conflict radius that determines the resampling resolution. */
        void setConflictRadius(float r) {
            _conflictRadius = r;
        }
        
        /** Resampling stops after n consecutive samples failed to contribute. */
        void setMaximumAttempts(int n) {
            _n = n;
        }

		/* Set random seed for sampling. */
		void setRandomSeed(unsigned int s) {
			srand(s);
		}

		/* Set parameters specific to traits. */
		void setTaskTraits(const TaskTraitsType &t) {
			_traits = t;
		}
        
        /** Resample input point cloud. */
		bool resample(const std::vector<typename TaskTraitsType::PositionVector> &positions,
					  const std::vector<typename TaskTraitsType::FeatureVector> &features,
					  std::vector<size_t> &outputIds)
        {
			if (positions.empty() || positions.size() != features.size())
                return false;
			
            // Build a random array of sample indices
            std::vector<size_t> sampleIndices;
			sampleIndices.reserve(positions.size());
			for (size_t i = 0; i < positions.size(); ++i)
                sampleIndices.push_back(i);
            std::random_shuffle(sampleIndices.begin(), sampleIndices.end());
                     
            // Loop over samples and try to add one after another.			
			typename TaskTraitsType::StackedLocator loc(_traits.stackedLocatorParams);
			typename TaskTraitsType::Stacker s(_traits.stackerParams);
			outputIds.clear();
            
            int attempt = 0;            
			size_t id = 0;
            while (id < sampleIndices.size() && attempt < _n) {
				size_t pointId = sampleIndices[id];
				typename TaskTraitsType::StackedVector sv = s(positions[pointId], features[pointId]);
               
				if (!loc.findAnyWithinRadius(sv, _conflictRadius)) {
					loc.add(sv);
					outputIds.push_back(pointId);
					attempt = 0;
                } else {
                    attempt++;
                }
                
                if (id % 5000 == 0) {
                    BBN_LOG("Processed %.2f - Generated %d of possible %d output samples\n",
							(float)id / positions.size() * 100, (int)outputIds.size(), (int)id);
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
		TaskTraitsType _traits;
    };
}

#endif