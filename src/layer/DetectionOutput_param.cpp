/*
 * Relu_param.cpp
 *
 *  Created on: Jun 17, 2019
 *      Author: doyle
 */

#include "DetectionOutput_param.h"

namespace tmnet
{
	/*************************************************************************
	* Function Name : ReluParam
	* Description   : construct function ,initialize all the variable
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	DetectionOutputParam::DetectionOutputParam()
	{
		num_class = 0;
		nms_threshold = 0.05f;
		nms_top_k = 300;
		keep_top_k = 100;
		confidence_threshold = 0.5f;
		variances[0] = 0.1f;
		variances[1] = 0.1f;
		variances[2] = 0.2f;
		variances[3] = 0.2f;
	}

	/*************************************************************************
	* Function Name : ~ReluParam
	* Description   : deconstruct function
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	DetectionOutputParam::~DetectionOutputParam()
	{

	}
}
