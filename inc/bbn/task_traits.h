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

namespace bbn {

	/** Traits for the sampling task. */
	template<
		typename PositionType, /** Vector type representing positional attributes. */
		typename FeatureType,  /** Vector type representing feature attributes. */
		bool UseAcceleration = true /** Use acceleration structure for fast nearest neighbor queries. */
	> struct TaskTraits 
	{
		typedef PositionType PositionVector;
		typedef FeatureType FeatureVector;
		typedef typename detail::StackedVectorType<PositionVector, FeatureVector>::type StackedVector;		
		typedef typename detail::LocatorType<StackedVector, UseAcceleration>::type StackedLocator;
		typedef typename detail::LocatorType<PositionVector, UseAcceleration>::type PositionLocator;
		typedef Stacking<PositionVector, FeatureVector> Stacker;
		typedef typename StackedVector::Scalar Scalar;

		TaskTraits()
		{}

		TaskTraits(const typename StackedLocator::Params &stackedParams, 
				   const typename PositionLocator::Params &positionParams,
				   const typename Stacker::Params &stackParams)
				   :stackedLocatorParams(stackedParams), positionLocatorParams(positionParams), stackerParams(stackParams)
		{}

		typename StackedLocator::Params stackedLocatorParams;
		typename PositionLocator::Params positionLocatorParams;
		typename Stacker::Params stackerParams;		 
	};

}

#endif