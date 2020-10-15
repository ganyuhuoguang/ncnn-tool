/*
 * InnerProduct_param.h
 *
 *  Created on: Jun 17, 2019
 *      Author: doyle
 */

#ifndef LAYER_INNERPRODUCT_PARAM_H_
#define LAYER_INNERPRODUCT_PARAM_H_

namespace tmnet
{
    class FcParam
    {
    public:
        FcParam();
        ~FcParam();

        unsigned int num_output;
        unsigned int bias_term;
        unsigned int data_size;
        unsigned long long uiWeightAddr;
        unsigned long long uiBiasAddr;
        //register
        unsigned long long uiRegInDataSrcAddr;		//R/W
        unsigned long long uiRegWDataSrcAddr;			//R/W
        unsigned int uiRegInDataClassesLength;	//R/W
        unsigned int uiRegWDataClassesLength;	//R/W
        unsigned long long uiRegOutDataDstAddr;		//R/W
        unsigned int uiRegFcStart;				//R/W
        unsigned int uiRegFcFinish;				//R
        unsigned int uiRegOutData;				//R
        unsigned long long uiRegBiasSrcAddr;			//R/W
        unsigned int uiRegOutputNumber;			//R/W
    };
}

#endif /* LAYER_INNERPRODUCT_PARAM_H_ */
