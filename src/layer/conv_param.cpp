/*
 * conv_param.cpp
 *
 *  Created on: Jun 17, 2019
 *      Author: doyle
 */
#include "conv_param.h"

namespace tmnet 
{
	/*************************************************************************
	* Function Name : ConvParam
	* Description   : construct function ,initialize all the variable
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	ConvParam::ConvParam()
	{
		num_output = 0;
		kernel_w = 0;
		dilation_size = 0;
		stride_w = 0;
		pad_w = 0;
		bias_term = 0;
		data_size = 0;
		//input scale value
		fInputScale = 0;
		//data DDR address
		uiWeightAddr = 0;
		uiBiasAddr = 0;
		uiAScaleAddr = 0;
		//register
		uiRegInFeatureAdd = 0;	//R/W
		uiRegOutFeatureAdd = 0;	//R/W
		uiRegKerWeightAdd = 0;	//R/W
		uiRegKerSize = 0;			//R/W
		uiRegFeaSize = 0;			//R/W
		uiRegFeaChannel = 0;		//R/W
		uiRegPadCtrl = 0;			//R/W
		uiRegConvCtrl = 0;		//R/W
		uiRegIfmSplTim = 0;		//R/W
		uiRegRowPerLd = 0;		//R/W
		uiRegRowLstLd = 0;		//R/W
	}

	/*************************************************************************
	* Function Name : ~ConvParam
	* Description   : deconstruct function
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	ConvParam::~ConvParam()
	{

	}
}

