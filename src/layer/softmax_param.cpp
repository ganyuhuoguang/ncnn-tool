/*
 * Relu_param.cpp
 *
 *  Created on: Jun 17, 2019
 *      Author: doyle
 */

#include "softmax_param.h"

namespace tmnet
{
	/*************************************************************************
	* Function Name : ReluParam
	* Description   : construct function ,initialize all the variable
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	SoftmaxParam::SoftmaxParam()
	{
		zero = 0;
		one = 0;
	}

	/*************************************************************************
	* Function Name : ~ReluParam
	* Description   : deconstruct function
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	SoftmaxParam::~SoftmaxParam()
	{

	}
}
