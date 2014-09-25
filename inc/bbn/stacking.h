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

#ifndef BBN_STACKING_H
#define BBN_STACKING_H

#include <Eigen/Dense>
#include <bbn/meta.h>
#include <functional>

namespace bbn {
	
	/** Provides stacking of positional and feature vectors into a single combined vector. This closesly
		reassembles the augmentative version of the bilateral differential. */
	template<class Position, class Feature>
	class Stacking : std::binary_function<Position, Feature, typename detail::StackedVectorType<Position, Feature>::type>
	{
	public:

		/** Configuration Parameters */
		struct Params {
			/** Defaults */
			Params()
				: positionWeight(1), featureWeight((result_type::Scalar)0.05)
			{}

			Params(typename result_type::Scalar pweight, typename result_type::Scalar fweight)
				: positionWeight(pweight), featureWeight(fweight)
			{}

			typename result_type::Scalar positionWeight;
			typename result_type::Scalar featureWeight;
		};

		/** Create stacking function with defaults. */
		Stacking()
			:_wPosition(1), _wFeature(result_type::Scalar(0.05))
		{}

		/** Create stacking function with custom weights. */
		Stacking(const Params &p)
			:_wPosition(p.positionWeight), _wFeature(p.featureWeight)
		{}

		/** Stack position and feature vector into a single vector. */
		inline result_type operator() (const Position &p, const Feature &f) 
		{ 
			result_type s(p.rows() + f.rows());
			if (result_type::SizeAtCompileTime != Eigen::Dynamic) {
				s.block<Position::RowsAtCompileTime, 1>(0, 0) = p * _wPosition;
				s.block<Feature::RowsAtCompileTime, 1>(p.rows(), 0) = f * _wFeature;
			} else {
				s.block(0, 0, p.rows(), 1) = p * _wPosition;
				s.block(p.rows(), 0, f.rows(), 1) = f * _wFeature;
			}

			return s;
		}

	private:
		typename result_type::Scalar _wPosition, _wFeature;
	};
}

#endif