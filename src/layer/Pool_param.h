/*
 * Pool_param.h
 *
 *  Created on: Jun 17, 2019
 *      Author: doyle
 */

#ifndef LAYER_POOL_PARAM_H_
#define LAYER_POOL_PARAM_H_

namespace tmnet
{
    class PoolParam
    {
    public:
        PoolParam();
        ~PoolParam();

        unsigned char pooltype;
        unsigned char kernel_w;
        unsigned char kernel_h; 
        unsigned char stride_w;
        unsigned char stride_h;
        unsigned char pad_left;
        unsigned char pad_right;
        unsigned char pad_top;
        unsigned char pad_bottom;
        unsigned char global_pooling;
        unsigned char pad_mode;
        //register
        unsigned long long  uiRegMatSrcAddr;		//R/W
        unsigned long long  uiRegMatDstAddr;		//R/W
        unsigned int  uiRegMatRowIn;		//R/W
        unsigned int  uiRegMatColIn;		//R/W
        unsigned int  uiRegMatRowOut;		//R/W
        unsigned int  uiRegMatColOut;		//R/W
        unsigned int  uiRegMatChannel;		//R/W
        unsigned int  uiRegAirthA;			//R/W
        unsigned int  uiRegAirthB;			//R/W
        unsigned int  uiRegAirthScale;		//R/W
        unsigned int  uiRegCtrl;			//R/W
    };
}



#endif /* LAYER_POOL_PARAM_H_ */
