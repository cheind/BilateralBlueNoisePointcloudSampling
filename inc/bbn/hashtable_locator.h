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

#ifndef BBN_HASHTABLE_LOCATOR_H
#define BBN_HASHTABLE_LOCATOR_H

#include <vector>
#include <unordered_map>
#include <limits>
#include <Eigen/Dense>
#include <bbn/eigen_types.h>

namespace bbn {

	/* Provides nearest neighbor search in n-dimensions using bucket hashing and L2 metric. */
	template<class VectorT>
	class HashtableLocator {
	public:
		/* Construct empty locator*/
		inline HashtableLocator()
			: _bucketSize(0.05f), _invBucketResolution(1.f / 0.05f)
		{}

		/* Construct with resolution */
		inline HashtableLocator(typename VectorT::Scalar resolution)
			: _bucketSize(resolution), _invBucketResolution(1.f / resolution)
		{}

		/** Add a new point. */
		void add(const VectorT &point)
		{
			size_t index = _points.size();
			_points.push_back(point);
			
			Bucket b = toBucket(point, _invBucketResolution);
			_bucketHash[b].push_back(index);
		}

		/** Add a range of points. */
		template<class VectorTIter>
		void add(VectorTIter begin, VectorTIter end)
		{
			for (VectorTIter i = begin; i != end; ++i) {
				add(*i);
			}
		}

		/* Find any neighbor within the specified radius.*/
		inline bool findAnyWithinRadius(const VectorT &query, typename VectorT::Scalar radius, size_t *index = 0, typename VectorT::Scalar *dist2 = 0) {			
			typename VectorT::Scalar bestDist2 = radius * radius;
			size_t bestIndex = std::numeric_limits<size_t>::max();

			Bucket minCorner, maxCorner;
			ballToBuckets(query, radius, _invBucketResolution, minCorner, maxCorner);

			BucketRangeIterator begin = BucketRangeIterator(minCorner, maxCorner);
			BucketRangeIterator end;

			bool found = false;
			for (BucketRangeIterator biter = begin; biter != end && !found; ++biter) {

				if (!testBallOverlapsBucket(query, radius, *biter, _bucketSize))
					continue;

				typename BucketHash::const_iterator iter = _bucketHash.find(*biter);
				if (iter != _bucketHash.end()) {
					for (size_t i = 0; i < iter->second.size(); ++i) {
						const float d = (query - _points[iter->second[i]]).squaredNorm();
						if (d <= bestDist2) {
							bestDist2 = d;
							bestIndex = iter->second[i];
							found = true;
							break;
						}
					}
				}
			}


			if (dist2) *dist2 = bestDist2;
			if (index) *index = bestIndex;

			return bestIndex != std::numeric_limits<size_t>::max();
		}

	private:	

		/* An arbitrary bucket in n-dimensions. */
		typedef typename Eigen::Matrix<int, VectorT::RowsAtCompileTime, 1> Bucket;

		/* Provides hashing comparison of bucket objects. */
		struct BucketHasher {

			/* Hash bucket*/
			inline std::size_t operator()(const Bucket& k) const
			{	
				std::size_t seed = 0;

				// Mimics boost::hash_combine for arbitrary sized vectors.
				for (int i = 0; i < k.rows(); ++i) {
					seed ^= static_cast<size_t>(scalarHasher(k(i))) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
				}

				return seed;
			}

			/* Compare two buckets for equality. */
			inline bool operator()(const Bucket &k0, const Bucket &k1) const
			{
				for (typename Bucket::Index i = 0; i < k0.rows(); ++i) {
					if (k0(i) != k1(i))
						return false;
				}
				return true;
			}

			std::hash<typename Bucket::Scalar> scalarHasher;
		};

		/* Provides n-dimensional iteration over bucket indices. */
		class BucketRangeIterator {
		public:
			/* Construct iterator from range to iterate. Both corners are inclusive. */
			BucketRangeIterator(const Bucket &minCorner, const Bucket &maxCorner)
				:_n(minCorner.rows()), _current(minCorner), _minCorner(minCorner), _maxCorner(maxCorner)
			{
				if (((_maxCorner - _minCorner).array() < 0).any()) {
					_n = 0;
				}
			}

			/* Construct invalid or end iterator. */
			BucketRangeIterator()
				:_n(0)
			{}

			const Bucket &operator*() const 
			{
				return _current;
			}

			/* Increment iterator to next position. */
			BucketRangeIterator &operator++() 
			{
				// Pop elements that correspond to maximum corner.
				while (_n > 0 && _current(_n - 1) >= _maxCorner(_n - 1)) {
					--_n;
				}

				// Increment position and fill up remainder
				if (_n > 0) {
					_current(_n - 1) += 1;

					if (_n < _minCorner.rows()) {
						const typename Bucket::Index nRowsToFill = _current.rows() - _n;
						_current.tail(nRowsToFill) = _minCorner.tail(nRowsToFill);
						_n = _minCorner.rows();
					}
				}

				return *this;
			}

			/* Test for equality. */
			bool operator==(const BucketRangeIterator &other) const
			{
				if (_n == 0 || other._n == 0) {
					return _n == 0 && other._n == 0;
				}
				else {
					return _current == other._current;
				}
			}

			/* Test for inequality. */
			bool operator!=(const BucketRangeIterator &other) const
			{
				return !operator==(other);
			}

		private:
			typename Bucket::Index _n;
			Bucket _current, _minCorner, _maxCorner;
		};
		
		/** Hash from bucket to list of points in bucket. */
		typedef std::unordered_map<Bucket, std::vector<size_t>, BucketHasher, BucketHasher> BucketHash;
		/** Array of points. */
		typedef std::vector<VectorT, Eigen::aligned_allocator<VectorT> > ArrayOfVectorT;


		/* Converts a point to a bucket. */
		static inline Bucket toBucket(const VectorT &point, typename VectorT::Scalar invResolution) 
		{ 
			Bucket b(point.rows(), 1);
			for (typename VectorT::Index i = 0; i < point.rows(); ++i) {
				b(i) = static_cast<int>(std::floor(point(i) * invResolution));
			}
			return b;
			
		}

		/* Converts a bucket back to a world point. The worldpoint describes the buckets min-corner*/
		static inline VectorT toWorldPoint(const Bucket &b, typename VectorT::Scalar resolution) {
			return b.template cast<typename VectorT::Scalar>() * resolution;
		}

		/** Converts a n-dimensional ball search to a list of buckets to search. Note that declaring the range of buckets as AABB is not ideal
			leads to possibly more buckets to search, especially in higher dimensions. */
		static inline void ballToBuckets(const VectorT &point, typename VectorT::Scalar radius, typename VectorT::Scalar invResolution, Bucket &minCorner, Bucket &maxCorner)
		{
			minCorner = toBucket(point - VectorT::Constant(radius), invResolution);
			maxCorner = toBucket(point + VectorT::Constant(radius), invResolution);
		}

		/* Test for intersection between an n-dimensional sphere and bounds.
		   Based on "On faster sphere box overlap testing" 
		   http://www.mrtc.mdh.se/projects/3Dgraphics/paperF.pdf
		 */
		static inline bool testBallOverlapsBucket(const VectorT &center, typename VectorT::Scalar radius, const Bucket &minCorner, typename VectorT::Scalar cellSize)
		{
			typedef typename VectorT::Scalar Scalar;

			VectorT worldMinCorner = toWorldPoint(minCorner,  cellSize);

            Scalar d = 0;
			for (typename Bucket::Index i = 0; i < minCorner.rows(); ++i) {
				// On the first glance this seems like it misses a case, when the center is inside the bounds in the current dimension.
				// But that's not the case, as in this scenerio the closest value is the center value itself, leading to zero error term.
				Scalar e = std::max<Scalar>(worldMinCorner(i) - center(i), 0) + 
						   std::max<Scalar>(center(i) - (worldMinCorner(i) + cellSize), 0);

				// In the paper it seems like there is a typo at this point.
				if (e > radius)
					return false;
				d += e*e;
			}

			return d <= radius * radius;
			return true;
		}

		
		BucketHash _bucketHash;
		ArrayOfVectorT _points;
		typename VectorT::Scalar _bucketSize, _invBucketResolution;
	};

}

#endif