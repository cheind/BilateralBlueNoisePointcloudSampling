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
#include <bbn/task_traits.h>
#include <bbn/util.h>

namespace bbn {
    
    /** Point based relaxation based on energy minimization. */    
	template<class TaskTraitsType>
    class EnergyMinimization {
    public:

		typedef typename TaskTraitsType::Scalar Scalar;
		typedef typename TaskTraitsType::ArrayOfPositionVector ArrayOfPositionVector;
		typedef typename TaskTraitsType::ArrayOfFeatureVector ArrayOfFeatureVector;
		typedef typename TaskTraitsType::StackedVector StackedVector;
		typedef typename TaskTraitsType::StackedLocator StackedLocator;
		typedef typename TaskTraitsType::PositionVector PositionVector;

        /** Default constructor. */
		EnergyMinimization()
			: _sigma(Scalar(0.1f)),
			_stepSize(Scalar(0.45f) * _sigma * _sigma)
        {}        
        
        /* Set the conflict radius that determines the resampling resolution. */
		void setKernelSigma(Scalar s) {
            _sigma = s;
			_stepSize = Scalar(0.45f) * _sigma * _sigma;
        }
       
		/* Set parameters specific to traits. */
		void setTaskTraits(const TaskTraitsType &t) {
			_traits = t;
		}
        
        /** Minimize samples based on energy formulation. */
		bool minimize(const ArrayOfPositionVector &positions,
					  const ArrayOfFeatureVector &features,
					  ArrayOfPositionVector &resultPositions,
					  ArrayOfFeatureVector &resultFeatures,
					  size_t nIterations)
        {
			if (positions.empty() || positions.size() != features.size())
                return false;

			const size_t nElements = positions.size();
			typename TaskTraitsType::Stacker stack(_traits.stackerParams);
			StackedLocator sloc(_traits.stackedLocatorParams);
			
			// During iterations we alternate between pointers
			ArrayOfPositionVector pposes[2];
			ArrayOfFeatureVector pfeatures[2];
			
			// Initialize
			pposes[0] = positions;
			pposes[1] = positions;
			pfeatures[0] = features;
			pfeatures[1] = features;

			// Loop
			int index = 0, nextIndex = 1;
			const PositionVector::Index nPositionRows = positions.front().rows();
			for (size_t iter = 0; iter < nIterations; ++iter) {
				
				ArrayOfPositionVector &curPositions = pposes[index];
				ArrayOfFeatureVector &curFeatures = pfeatures[index];
				ArrayOfPositionVector &nextPositions = pposes[nextIndex];
				ArrayOfFeatureVector &nextFeatures = pfeatures[nextIndex];

				// Build locator for stacked elements
				sloc.reset();
				for (size_t i = 0; i < nElements; ++i) {
					sloc.add(stack(curPositions[i], curFeatures[i]));
				}

				// For each element
				for (size_t i = 0; i < nElements; ++i) {

					// Determine energy gradient as described in equation 14.
					StackedVector g = energyGradient(i, sloc);
					
					// Move sample position / feature
					nextPositions[i] = curPositions[i] - _stepSize * g.topRows(nPositionRows);
					nextFeatures[i] = curFeatures[i];

					if (i % 100 == 0) {
						BBN_LOG("Processed %.2f in energy minimization \n",
							(float)i / nElements * 100);
					}

					// Constrain sample position /feature
					// TODO
				}

				index = nextIndex;
				nextIndex = (nextIndex + 1) % 2;
			}

			resultPositions = pposes[index];
			resultFeatures = pfeatures[index];


			return true;
			
        }
        
    private:

		StackedVector energyGradient(size_t queryIndex, const StackedLocator &loc) const
		{
			StackedVector g = StackedVector::Zero(loc.dims());

			std::vector<size_t> neighborIds;
			std::vector<Scalar> neighborDists2;
			const StackedVector &query = loc.get(queryIndex);

			if (!loc.findAllWithinRadius(query, _sigma * 2, neighborIds, neighborDists2))
				return g;

			const Scalar sigmaSquared = _sigma * _sigma;
			const Scalar oneOverSigmaSquared = 1 / sigmaSquared;

			for (size_t nidx = 0; nidx < neighborIds.size(); ++nidx) {
				if (neighborIds[nidx] == queryIndex)
					continue; // don't include self
				
				const StackedVector &n = loc.get(neighborIds[nidx]);			
				g += (n - query) * oneOverSigmaSquared * exp(-neighborDists2[nidx] * Scalar(0.5) * oneOverSigmaSquared);
			}

			return g;
		}


		Scalar _sigma, _stepSize;
        TaskTraitsType _traits;
    };
}

#endif