/*
 * Relu_param.cpp
 *
 *  Created on: Jun 17, 2019
 *      Author: doyle
 */

#include "region_param.h"

namespace tmnet
{
	/*************************************************************************
	* Function Name : ReluParam
	* Description   : construct function ,initialize all the variable
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	RegionParam::RegionParam()
	{
		zero = 0;
		one = 0;
		two = 0;
		three = 0;
	}

	/*************************************************************************
	* Function Name : ~ReluParam
	* Description   : deconstruct function
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	RegionParam::~RegionParam()
	{

	}
}
