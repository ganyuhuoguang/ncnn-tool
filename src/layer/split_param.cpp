/*
 * Relu_param.cpp
 *
 *  Created on: Jun 17, 2019
 *      Author: doyle
 */

#include "split_param.h"

namespace tmnet
{
	/*************************************************************************
	* Function Name : ReluParam
	* Description   : construct function ,initialize all the variable
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	SplitParam::SplitParam()
	{
		uiRegMatSrcAddr = 0;		//R/W
		uiRegMatDstAddr = 0;		//R/W
		uiRegReluCtrl = 0;		//R/W
		uiRegCubeInWidth = 0;		//R/W
		uiRegCubeInHeight = 0;	//R/W
		uiRegCubeInChannel = 0;	//R/W
		uiRegBsBypass = 0;		//R/W
		uiRegBsAluSrc = 0;		//R/W
		uiRegBsMulSrc = 0;		//R/W
		uiRegBsOprand = 0;		//R/W
		uiRegBsQn = 0;			//R/W
		uiRegClassesLength = 0;	//R/W
		uiRegBsCfg = 0;			//R/W
	}

	/*************************************************************************
	* Function Name : ~ReluParam
	* Description   : deconstruct function
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	SplitParam::~SplitParam()
	{

	}
}
