/*
 * Pool_param.cpp
 *
 *  Created on: Jun 17, 2019
 *      Author: doyle
 */

#include "Pool_param.h"

namespace tmnet
{
	/*************************************************************************
	* Function Name : PoolParam
	* Description   : construct function ,initialize all the variable
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	PoolParam::PoolParam()
	{
		pooltype = 0;
		kernel_w = 0;
		stride_w = 0;
		pad_left = 0;
		pad_right = 0;
		global_pooling = 0;
		//register
		uiRegMatSrcAddr = 0;		//R/W
		uiRegMatDstAddr = 0;		//R/W
		uiRegMatRowIn = 0;		//R/W
		uiRegMatColIn = 0;		//R/W
		uiRegMatRowOut = 0;		//R/W
		uiRegMatColOut = 0;		//R/W
		uiRegMatChannel = 0;		//R/W
		uiRegAirthA = 0;			//R/W
		uiRegAirthB = 0;			//R/W
		uiRegAirthScale = 0;		//R/W
		uiRegCtrl = 0;			//R/W
	}

	/*************************************************************************
	* Function Name : ~PoolParam
	* Description   : deconstruct function
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	PoolParam::~PoolParam()
	{

	}
}
