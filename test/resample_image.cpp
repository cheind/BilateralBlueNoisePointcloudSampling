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
#include <ctime>
#include <iostream>
#include <random>

typedef bbn::TaskTraits<float, 2, 1> ImageTraits;

cv::Mat createImage(const std::vector<ImageTraits::Vector> &sampled, int imageSize)
{
	cv::Mat gray(imageSize, imageSize, CV_8UC1);
	gray.setTo(255);

	cv::Mat mask(imageSize, imageSize, CV_8UC1);
	mask.setTo(255);

	for (size_t i = 0; i < sampled.size(); ++i) {
		cv::Point pix(
			cvRound(sampled[i].x()*gray.cols),
			cvRound(sampled[i].y()*gray.rows));

		cv::circle(gray, pix, 2, cvRound(sampled[i].z() * 255), CV_FILLED);
		cv::circle(mask, pix, 2, 0, CV_FILLED);
	}

	cv::Mat colorized;
	cv::applyColorMap(gray, colorized, cv::COLORMAP_JET);
	colorized.setTo(255, mask);
	return colorized;
}


class PixelSampler {
public:
	PixelSampler() 
	{
		_gen = std::mt19937(_rd());
		_disPos = std::uniform_real_distribution<float>(0, 1);
		_disWeight = std::discrete_distribution<>({50,50});
	}

	ImageTraits::Vector operator()(void) {
		ImageTraits::Vector v(3);
		v(0) = _disPos(_gen);
		v(1) = _disPos(_gen);
		v(2) = _disWeight(_gen) / 2.f;
		return v;
	}
private:
	std::random_device _rd;
	std::mt19937 _gen;
	std::uniform_real_distribution<float> _disPos;
	std::discrete_distribution<> _disWeight;
};

int main(int argc, const char **argv) 
{  
	const int imageSize = 500;
	const float spacing = 1.f / imageSize;
	
	ImageTraits it;
	PixelSampler sampler;

	bbn::DartThrowing<ImageTraits> adt;
	adt.setTaskTraits(it);
	adt.setConflictRadius(0.08f);
	adt.setMaximumAttempts(100000);	
	
	std::vector<ImageTraits::Vector> sampled;
    
	if (!adt.resample(sampler, std::back_inserter(sampled))) {
        std::cerr << "Failed to throw darts." << std::endl;
    }

	cv::Mat img = createImage(sampled, imageSize);
	cv::imshow("result", img);
	cv::waitKey();

	bbn::EnergyMinimization<ImageTraits> em;
	em.setTaskTraits(it);
	em.setKernelSigma(0.03f);
	em.setStepSize(0.45f * 0.03f * 0.03f);
	em.setMaximumSearchRadius(0.2f);

	while (true) {
		em.minimize(sampled.begin(), sampled.end(), sampled.begin(), [&](ImageTraits::VectorLike p) {
			p.x() = std::max<float>(0, std::min<float>(p.x(), 1));
			p.y() = std::max<float>(0, std::min<float>(p.y(), 1));
		}, 1);

		img = createImage(sampled, imageSize);
		cv::imshow("result", img);
		cv::waitKey();
	}

    
    return 0;
}