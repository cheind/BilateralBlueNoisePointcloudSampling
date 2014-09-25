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

#ifndef BBN_ENERGY_MINIMIZATION_H
#define BBN_ENERGY_MINIMIZATION_H

#include <Eigen/Dense>
#include <vector>
#include <bbn/util.h>
#include <bbn/stacking.h>

namespace bbn {
    
    /** Point based relaxation based on energy minimization. */    
	template<class TaskTraitsType>
    class EnergyMinimization {
    public:
        /** Default constructor. */
		EnergyMinimization()
			: _sigma(typename TaskTraitsType::Scalar(0.1f)),
			_stepSize(typename TaskTraitsType::Scalar(0.45f) * _sigma)
        {}        
        
        /* Set the conflict radius that determines the resampling resolution. */
		void setKernelSigma(typename TaskTraitsType::Scalar s) {
            _sigma = s;
        }

		/* Set step size for gradient descent. */
		void setStepSize(typename TaskTraitsType::Scalar s) {
			_stepSize = s;
		}
        
		/* Set parameters specific to traits. */
		void setTaskTraits(const TaskTraitsType &t) {
			_traits = t;
		}
        
        /** Minimize samples based on energy formulation. */
		bool minimize(const std::vector<typename TaskTraitsType::PositionVector> &positions,
					  const std::vector<typename TaskTraitsType::FeatureVector> &features,
					  std::vector<typename TaskTraitsType::PositionVector> &resultPositions,
					  std::vector<typename TaskTraitsType::FeatureVector> &resultFeatures,
					  size_t nIterations)
        {
			if (positions.empty() || positions.size() != features.size())
                return false;

			const size_t nElements = positions.size();
			typename TaskTraitsType::Stacker stack(_traits.stackerParams);
			typename TaskTraitsType::StackedLocator sloc(_traits.stackedLocatorParams);
			
			// During iterations we alternate between pointers
			std::vector<typename TaskTraitsType::PositionVector> pposes[2];
			std::vector<typename TaskTraitsType::FeatureVector>  pfeatures[2];
			
			// Initialize
			pposes[0] = positions;
			pposes[1].resize(positions.size());
			pfeatures[0] = features;
			pfeatures[1].resize(features.size());

			// Loop
			int index = 0, nextIndex = 1;
			for (size_t iter = 0; iter < nIterations; ++iter) {
				std::vector<typename TaskTraitsType::PositionVector> &curPositions = pposes[index];
				std::vector<typename TaskTraitsType::FeatureVector> &curFeatures = pfeatures[index];
				std::vector<typename TaskTraitsType::PositionVector> &nextPositions = pposes[nextIndex];
				std::vector<typename TaskTraitsType::FeatureVector> &nextFeatures = pfeatures[nextIndex];

				// Build locator for stacked elements
				sloc.reset();
				for (size_t i = 0; i < nElements; ++i) {
					sloc.add(stack(curPositions[i], curFeatures[i]));
				}

				// For each element
				for (size_t i = 0; i < nElements; ++i) {

					// Determine energy gradient as described in equation 14.
					typename TaskTraitsType::PositionVector g = energyGradient(i, sloc);
					
					// Move sample position / feature
					nextPositions[i] = curPositions[i] - _stepSize * energyGradient;
					nextFeatures[i] = curFeatures[i];

					// Constrain sample position /feature
				}

				index = nextIndex;
				nextIndex = (nextIndex + 1) % 2;
			}

			return true;
			
        }
        
    private:

		typename TaskTraitsType::StackedVector energyGradient(size_t queryIndex, const typename TaskTraitsType::StackedLocator &loc) const
		{
			typename TaskTraitsType::StackedVector g = TaskTraitsType::StackedVector::Zero(loc.dims());

			std::vector<size_t> neighborIds;
			std::vector<TaskTraitsType::StackedVector::Scalar> neighborDists2;
			const typename TaskTraitsType::StackedVector &query = loc.get(queryIndex);

			if (!loc.findAllWithinRadius(query, _sigma * 5, neighborIds, neighborDists2))
				return g;

			const typename TaskTraitsType::Scalar sigmaSquared = _sigma * _sigma;
			const typename TaskTraitsType::Scalar oneOverSigmaSquared = 1 / sigmaSquared;

			for (size_t nidx = 0; nidx < neighborIds.size(); ++nidx) {
				if (neighborIds[nidx] == index)
					continue; // don't include self
				
				const typename TaskTraitsType::StackedVector &n = loc.get(neighborIds[nidx]);
				g += (n - query) * oneOverSigmaSquared * exp(-neighborDists2[nidx] * typename TaskTraitsType::Scalar(0.5) * oneOverSigmaSquared);
			}

			return g;
		}


		typename TaskTraitsType::Scalar _sigma, _stepSize;
        TaskTraitsType _traits;
    };
}

#endif