/*
 * Relu_param.h
 *
 *  Created on: Jun 17, 2019
 *      Author: doyle
 */

#ifndef LAYER_Upsample_PARAM_H_
#define LAYER_Upsample_PARAM_H_
#include <string.h>
#include <iostream>
#include <vector>

namespace tmnet
{
	class UpsampleParam
	{
	public:
		UpsampleParam();
		~UpsampleParam();

	    int scale;
	};
}

#endif /* LAYER_Upsample_PARAM_H_ */
