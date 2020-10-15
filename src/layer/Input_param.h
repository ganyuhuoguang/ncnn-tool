/*
 * Input_param.h
 *
 *  Created on: Jun 17, 2019
 *      Author: doyle
 */

#ifndef LAYER_INPUT_PARAM_H_
#define LAYER_INPUT_PARAM_H_

namespace tmnet
{
	class Inputparam
	{
	public:
		Inputparam();
		~Inputparam();

		unsigned int feature_w;
		unsigned int feature_h;
		unsigned int feature_d;
	};
}

#endif /* LAYER_INPUT_PARAM_H_ */
