/*
 * InnerProduct_param.cpp
 *
 *  Created on: Jun 17, 2019
 *      Author: doyle
 */

#include "InnerProduct_param.h"

namespace tmnet
{
	/*************************************************************************
	* Function Name : FcParam
	* Description   : construct function ,initialize all the variable
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	FcParam::FcParam()
	{
		num_output = 0;
		bias_term = 0;
		data_size = 0;
		uiWeightAddr = 0;
		uiBiasAddr = 0;
		//register
		uiRegInDataSrcAddr = 0;		//R/W
		uiRegWDataSrcAddr = 0;			//R/W
		uiRegInDataClassesLength = 0;	//R/W
		uiRegWDataClassesLength = 0;	//R/W
		uiRegOutDataDstAddr = 0;		//R/W
		uiRegFcStart = 0;				//R/W
		uiRegFcFinish = 0;				//R
		uiRegOutData = 0;				//R
		uiRegBiasSrcAddr = 0;			//R/W
		uiRegOutputNumber = 0;			//R/W
	}

	/*************************************************************************
	* Function Name : ~FcParam
	* Description   : deconstruct function
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	FcParam::~FcParam()
	{

	}
}


