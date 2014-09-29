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

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <bbn/task_traits.h>
#include <bbn/normalization.h>
#include <bbn/dart_throwing.h>
#include <bbn/energy_minimization.h>

#include <iostream>

int main(int argc, const char **argv) 
{  
	const int imageSize = 500;
	const float spacing = 1.f / imageSize;

	typedef bbn::TaskTraits< Eigen::Vector2f, Eigen::Vector1f, true> ImageTraits;

	ImageTraits::ArrayOfPositionVector positions;
	ImageTraits::ArrayOfFeatureVector features;


	for (float x = 0; x < 1.f; x += spacing) {
		for (float y = 0; y < 1.f; y += spacing) {
			positions.push_back(Eigen::Vector2f(x, y));
			positions.push_back(Eigen::Vector2f(x, y));
			
			Eigen::Vector1f f;
			f(0) = 0;
			features.push_back(f);

			f(0) = 1;
			features.push_back(f);			
		}
	}

   
	// Resample by dart throwing.
	std::vector<size_t> outputIds;

	bbn::DartThrowing<ImageTraits> adt;
	adt.setConflictRadius(0.05f);
	adt.setRandomSeed(10);
	adt.setMaximumAttempts(1000000);
    
	if (!adt.resample(positions, features, outputIds)) {
        std::cerr << "Failed to throw darts." << std::endl;
    }

	cv::Mat img(imageSize, imageSize, CV_8UC3);
	img.setTo(255);

	static cv::Scalar colors[] = {
		cv::Scalar(0, 0, 0),
		cv::Scalar(0, 255, 0),
		cv::Scalar(255, 0, 0),		
		cv::Scalar(0, 0, 255)
	};

	for (size_t i = 0; i < outputIds.size(); ++i) {
		cv::Point pix(
			cvRound(positions[outputIds[i]].x()*imageSize),
			cvRound(positions[outputIds[i]].y()*imageSize));

		cv::Scalar c = colors[cvRound(features[outputIds[i]].x())];
		cv::circle(img, pix, 2, c, CV_FILLED);
	}

	cv::imshow("result", img);
	cv::waitKey();
    
    return 0;
}