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
	template<class Traits>
    class EnergyMinimization {
    public:

		typedef typename Traits::Scalar Scalar;
		typedef typename Traits::Vector Vector;
		typedef typename Traits::Matrix Matrix;
		typedef typename Traits::Locator Locator;
		
        /** Default constructor. */
		EnergyMinimization()
			: _sigma(Scalar(0.03f)),
			  _stepSize(Scalar(0.03f) * _sigma * _sigma),
			  _maxSearchRadius(Scalar(0.03f) * Scalar(2.576))
        {}        
        
        /* Set the conflict radius that determines the resampling resolution. */
		void setKernelSigma(Scalar s) {
            _sigma = s;
        }

		/* Set gradient descent step size. */
		void setStepSize(Scalar s) {
			_stepSize = s;
		}

		/* Set maximum search radius for neighbors. */
		void setMaximumSearchRadius(Scalar s) {
			_maxSearchRadius = s;
		}
       
		/* Set parameters specific to traits. */
		void setTaskTraits(const Traits &t) {
			_traits = t;
		}

        /** Minimize samples based on energy formulation. */
		template<typename ConstrainFnc, typename VectorInputIterator, typename VectorOutputIterator>
		bool minimize(VectorInputIterator samplesBegin,
					  VectorInputIterator samplesEnd,
					  VectorOutputIterator refinedSamplesIter,
					  const ConstrainFnc &fnc,
					  size_t nIterations)					  
        {

			const size_t nElements = static_cast<size_t>(std::distance(samplesBegin, samplesEnd));
			if (nElements == 0)
				return false;

			typename Traits::Locator loc(_traits.getLocatorParams());
			typename Traits::Matrix positions[2] = {
				Traits::Matrix(_traits.getStackedDims(), nElements),
				Traits::Matrix(_traits.getStackedDims(), nElements)
			};

			
			// Initialize
			VectorInputIterator sampleIter = samplesBegin;
			for (size_t i = 0; i != nElements; ++i) {
				positions[0].col(i) = *sampleIter;
				positions[1].col(i) = *sampleIter;
				++sampleIter;
			}

			// Loop
			int index = 0, nextIndex = 1;
			Vector gradient;
			Scalar totalEnergy = 0;
			for (size_t iter = 0; iter < nIterations; ++iter) {

				Matrix &curPositions = positions[index];
				Matrix &nextPositions = positions[nextIndex];
				
				// Build locator for modified elements
				loc.reset();
				for (size_t i = 0; i < nElements; ++i) {
					loc.add(curPositions.col(i));
				}

				// For each element
				totalEnergy = 0;				
				for (size_t i = 0; i < nElements; ++i) {

					// Determine energy gradient as described in equation 14.

					totalEnergy += energy(i, loc, gradient);
					
					// Move sample position / feature
					nextPositions.col(i) = curPositions.col(i);
					nextPositions.col(i).topRows(_traits.getPositionDims()) -= _stepSize * gradient.topRows(_traits.getPositionDims());

					// Constrain sample position / feature
					fnc(nextPositions.col(i));
				}

				BBN_LOG("Energy minimization %.2f%% - Total energy %.2f\r",
					(float)iter / nIterations * 100, totalEnergy);

				index = nextIndex;
				nextIndex = (nextIndex + 1) % 2;
			}

			BBN_LOG("Energy minimization 100.00%% - Total energy %.2f\n", totalEnergy);

			for (size_t i = 0; i != nElements; ++i) {
				*refinedSamplesIter++ = positions[index].col(i);
			}

			return true;
			
        }
        
    private:

		Scalar energy(size_t queryIndex, const Locator &loc, Vector &gradient) const
		{
			gradient = Vector::Zero(loc.dims());
			Scalar energy = 0;

			std::vector<size_t> neighborIds;
			std::vector<Scalar> neighborDists2;
			const Vector &query = loc.get(queryIndex);

			if (!loc.findAllWithinRadius(query, _maxSearchRadius, neighborIds, neighborDists2))
				return energy;

			const Scalar sigmaSquared = _sigma * _sigma;
			const Scalar oneOverSigmaSquared = 1 / sigmaSquared;

			for (size_t nidx = 0; nidx < neighborIds.size(); ++nidx) {
				if (neighborIds[nidx] == queryIndex)
					continue; // don't include self
				
				const Vector &n = loc.get(neighborIds[nidx]);
				const Scalar e = exp(-neighborDists2[nidx] * Scalar(0.5) * oneOverSigmaSquared);

				energy += e;
				gradient += (n - query) * oneOverSigmaSquared * e;
			}

			return energy;
		}


		Scalar _sigma, _stepSize, _maxSearchRadius;
        Traits _traits;
    };
}

#endif