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

#ifndef BBN_BRUTEFORCE_LOCATOR_H
#define BBN_BRUTEFORCE_LOCATOR_H

#include <vector>
#include <limits>
#include <Eigen/Dense>
#include <iostream>

namespace bbn {

	/* Provides nearest neighbor search in n-dimensions using exhaustive search and L2 metric. */
	template<class VectorT>
	class BruteforceLocator {
	public:
		/* Construct empty locator*/
		inline BruteforceLocator()
		{}

		/** Add a new point. */
		void add(const VectorT &point)
		{
			_points.push_back(point);
		}

		/** Add a range of points. */
		template<class VectorTIter>
		void add(VectorTIter begin, VectorTIter end)
		{
			_points.insert(_points.end(), begin, end);
		}

		/* Find any neighbor within the specified radius.*/
		inline bool findAnyWithinRadius(const VectorT &query, typename VectorT::Scalar radius, size_t *index = 0, typename VectorT::Scalar *dist2 = 0) {			
			typename VectorT::Scalar bestDist2 = radius * radius;
			size_t bestIndex = std::numeric_limits<size_t>::max();
			
			for (size_t i = 0; i < _points.size(); ++i) {
				const float d = (query - _points[i]).squaredNorm();
				if (d <= bestDist2) {
					bestDist2 = d;
					bestIndex = i;
					break;
				}
			}

			if (dist2) *dist2 = bestDist2;
			if (index) *index = bestIndex;

			return bestIndex != std::numeric_limits<size_t>::max();
		}

	private:
		typedef std::vector<VectorT, Eigen::aligned_allocator<VectorT> > ArrayOfVectorT;
		ArrayOfVectorT _points;
	};

}

#endif