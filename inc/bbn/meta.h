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

#ifndef BBN_META_H
#define BBN_META_H

#include <Eigen/Dense>
#include <bbn/bruteforce_locator.h>
#include <bbn/hashtable_locator.h>

namespace bbn {
	namespace detail {

		// Derive number of stacked dimensions.

		template<int X, int Y>
		struct StackedSizeAtCompileTime { enum { size = X + Y }; };

		template<int Y>
		struct StackedSizeAtCompileTime<Eigen::Dynamic, Y> { enum { size = Eigen::Dynamic }; };

		template<int X>
		struct StackedSizeAtCompileTime<X, Eigen::Dynamic> { enum { size = Eigen::Dynamic }; };

		template<>
		struct StackedSizeAtCompileTime<Eigen::Dynamic, Eigen::Dynamic> { enum { size = Eigen::Dynamic }; };

		/* Stacked vector type provide. When stacking the positional (A) and feature (B) vector into a single vector,
		   this struct provides the resulting type. */
		template<typename A, typename B>
		struct StackedVectorType {
			typedef Eigen::Matrix<
				typename A::Scalar,
				StackedSizeAtCompileTime<A::RowsAtCompileTime, B::RowsAtCompileTime>::size,
				1
			> type;
		};


		template<typename Vector, bool UseAcceleration>
		struct LocatorType {
			typedef BruteforceLocator<Vector> type;
		};

		template<typename Vector>
		struct LocatorType<Vector, true> {
			typedef HashtableLocator<Vector> type;
		};


	}
}

#endif