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
	template<class Traits>
    class DartThrowing {
    public:
		
		typedef typename Traits::Scalar Scalar;
		typedef typename Traits::Vector Vector;

        /** Default constructor. */
		DartThrowing()
        : _conflictRadius(Scalar(0.01)), _n(100000)
        {}        
        
        /** Set the conflict radius that determines the resampling resolution. */
        void setConflictRadius(float r) {
            _conflictRadius = r;
        }
        
        /** Resampling stops after n consecutive samples failed to contribute. */
        void setMaximumAttempts(size_t n) {
            _n = n;
        }

		/* Set parameters specific to traits. */
		void setTaskTraits(const Traits &t) {
			_traits = t;
		}
        
        /** Resample input point cloud. */
		template<typename SamplerFnc, typename VectorOutputIterator>
		bool resample(SamplerFnc &sampler, VectorOutputIterator outputIter)
        {
			typename Traits::Locator loc(_traits.getLocatorParams());

			int valids = 0;
			for (size_t n = 0; n < _n; ++n) {
				Vector v = sampler(); // Ask for a new sample.

				if (!loc.findAnyWithinRadius(v, _conflictRadius)) {
					loc.add(v);
					*outputIter++ = v;
					++valids;
				}		

				if (n % 5000 == 0) {
					BBN_LOG("Dart throwing %.2f%% - Found %d in %d attempts\r",
						(float)n / _n * 100, valids, n);
				}
			}

			BBN_LOG("Dart throwing 100.00%% - Found %d in %d attempts\n",
				valids, _n);

			return valids > 0;
        }
        
    private:
        
        Scalar _conflictRadius;
        size_t _n;
		Traits _traits;
    };
}

#endif