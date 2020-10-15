/*
 * Soft_param.cpp
 *
 *  Created on: Jun 17, 2019
 *      Author: doyle
 */

#include "Soft_param.h"

namespace tmnet 
{
	/*************************************************************************
	* Function Name : SoftParam
	* Description   : construct function ,initialize all the variable
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	SoftParam::SoftParam()
	{
		uiRegMatSrcAddr = 0;		//R/W
		uiRegMatDstAddr = 0;		//R/W
		uiRegClassesLength = 0;	//R/W
		uiRegSoftCtrl = 0;		//R/W
		uiRegMaxNum1 = 0;			//R
		uiRegMax1Probability = 0;	//R
		uiRegMaxNum2 = 0;			//R
		uiRegMax2Probability = 0;	//R
		uiRegMaxNum3 = 0;			//R
		uiRegMax3Probability = 0;	//R
		uiRegMaxNum4 = 0;			//R
		uiRegMax4Probability = 0;	//R
		uiRegMaxNum5 = 0;			//R
		uiRegMax5Probability = 0;	//R
	}

	/*************************************************************************
	* Function Name : ~SoftParam
	* Description   : deconstruct function
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	SoftParam::~SoftParam()
	{

	}
}

