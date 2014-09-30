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

#ifndef BBN_TASK_TRAITS_H
#define BBN_TASK_TRAITS_H

#include <bbn/meta.h>
#include <bbn/stacking.h>
#include <vector>

namespace bbn {
	
	/** Traits and options for working with algorithms. */
	template<
		typename Scalar,								/** Scalar value type. I.e float, double, ... */
		int PositionDims = Eigen::Dynamic,				/** Number of positional dimensions at compile time. */
		int FeatureDims = Eigen::Dynamic,				/** Number of positional dimensions at compile time. */
		bool UseAcceleration = true						/** Use acceleration structures for faster nearest neighbor queries. */
	> class TaskTraits
	{
	public:
		enum { 
			PositionDimsAtCompileTime = PositionDims, 
			FeatureDimsAtCompileTime = FeatureDims,
			StackedDimsAtCompileTime = detail::StackedSizeAtCompileTime<PositionDims, FeatureDims>::size
		};

		typedef Scalar Scalar;																		/** Scalar type */
		typedef typename Eigen::Matrix<Scalar, StackedDimsAtCompileTime, 1> Vector;					/** Vector type (position + feature) */
		typedef typename Eigen::Ref<Vector> VectorLike;													/** Vector type (position + feature) */
		typedef typename Eigen::Matrix<
			Scalar,  
			StackedDimsAtCompileTime, 
			Eigen::Dynamic,
			Eigen::ColMajor> Matrix;																/** Matrix type holding n Vectors in rows */
		typedef typename detail::LocatorType<Vector, UseAcceleration>::type Locator;				/** Locator type */

		TaskTraits()
			: _posDims(0), _featureDims(0)
		{}

		TaskTraits(typename Vector::Index posDims, typename Vector::Index featureDims)
			: _posDims(posDims), _featureDims(featureDims)
		{}

		typename Vector::Index getPositionDims() const {
			if (PositionDimsAtCompileTime != Eigen::Dynamic) {
				return PositionDimsAtCompileTime;
			} else {
				return _posDims;
			}
		}

		typename Vector::Index getFeatureDims() const {
			if (FeatureDimsAtCompileTime != Eigen::Dynamic) {
				return FeatureDimsAtCompileTime;
			}
			else {
				return _featureDims;
			}
		}

		typename Vector::Index getStackedDims() const {
			return getPositionDims() + getFeatureDims();			
		}

		void setPositionDims(typename Vector::Index d) {
			if (PositionDimsAtCompileTime == Eigen::Dynamic) {
				_posDims = d;
			}
		}

		void setFeatureDims(typename Vector::Index d) {
			if (FeatureDimsAtCompileTime == Eigen::Dynamic) {
				_featureDims = d;
			}
		}

		typename Locator::Params getLocatorParams() const {
			return _locatorParams;
		}

		void setLocatorParams(const typename Locator::Params &p) {
			_locatorParams = p;
		}

		
	private:
		typename Vector::Index _posDims, _featureDims;
		typename Locator::Params _locatorParams;
	};

}

#endif