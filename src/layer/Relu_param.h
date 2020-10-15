/*
 * Relu_param.h
 *
 *  Created on: Jun 17, 2019
 *      Author: doyle
 */

#ifndef LAYER_RELU_PARAM_H_
#define LAYER_RELU_PARAM_H_

namespace tmnet
{
    class ReluParam
    {
    public:
        ReluParam();
        ~ReluParam();

        float slope;

        //register
        unsigned long long  uiRegMatSrcAddr;		//R/W
        unsigned long long  uiRegMatDstAddr;		//R/W
        unsigned int  uiRegReluCtrl;		//R/W
        unsigned int  uiRegCubeInWidth;		//R/W
        unsigned int  uiRegCubeInHeight;	//R/W
        unsigned int  uiRegCubeInChannel;	//R/W
        unsigned int  uiRegBsBypass;		//R/W
        unsigned long long  uiRegBsAluSrc;		//R/W
        unsigned long long  uiRegBsMulSrc;		//R/W
        unsigned int  uiRegBsOprand;		//R/W
        unsigned int  uiRegBsQn;			//R/W
        unsigned int  uiRegClassesLength;	//R/W
        unsigned int  uiRegBsCfg;			//R/W

    };
}
#endif /* LAYER_RELU_PARAM_H_ */
