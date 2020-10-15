/*
 * Soft_param.h
 *
 *  Created on: Jun 17, 2019
 *      Author: doyle
 */

#ifndef LAYER_SOFT_PARAM_H_
#define LAYER_SOFT_PARAM_H_

namespace tmnet
{
	class SoftParam
	{
	public:
		SoftParam();
		~SoftParam();

		//register
		unsigned long long  uiRegMatSrcAddr;		//R/W
		unsigned long long  uiRegMatDstAddr;		//R/W
		unsigned int  uiRegClassesLength;	//R/W
		unsigned int  uiRegSoftCtrl;		//R/W
		unsigned int  uiRegMaxNum1;			//R
		unsigned int  uiRegMax1Probability;	//R
		unsigned int  uiRegMaxNum2;			//R
		unsigned int  uiRegMax2Probability;	//R
		unsigned int  uiRegMaxNum3;			//R
		unsigned int  uiRegMax3Probability;	//R
		unsigned int  uiRegMaxNum4;			//R
		unsigned int  uiRegMax4Probability;	//R
		unsigned int  uiRegMaxNum5;			//R
		unsigned int  uiRegMax5Probability;	//R
	};
}
#endif /* LAYER_SOFT_PARAM_H_ */
