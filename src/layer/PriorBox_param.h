/*
 * Relu_param.h
 *
 *  Created on: Jun 17, 2019
 *      Author: doyle
 */

#ifndef LAYER_PriorBox_PARAM_H_
#define LAYER_PriorBox_PARAM_H_
#include <string.h>
#include <iostream>
#include <vector>

namespace tmnet
{
	class PriorBoxParam
	{
	public:
		PriorBoxParam();
		~PriorBoxParam();

	    int zero;
	    int one;
	    int two;
	    int third;
		float variances[4];
		int flip;
		int clip;
		int image_width;
		int image_height;
		float step_width;
		float step_height;
		float offset;
		std::vector<float> min_sizes;
		std::vector<float> max_sizes;
		std::vector<float> aspect_ratios;
	};
}

#endif /* LAYER_RELU_PARAM_H_ */
