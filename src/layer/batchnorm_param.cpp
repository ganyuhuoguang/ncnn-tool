/*
 * BatchNorm_param.cpp
 *
 *  Created on: Jun 17, 2019
 *      Author: doyle
 */

#include "batchnorm_param.h"

namespace tmnet 
{
	/*************************************************************************
	* Function Name : BatchNormParam
	* Description   : construct function ,initialize all the variable
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	BatchNormParam::BatchNormParam()
	{
		zero = 0;
		one = 0;
	}

	/*************************************************************************
	* Function Name : ~BatchNormParam
	* Description   : deconstruct function
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	BatchNormParam::~BatchNormParam()
	{

	}
}
