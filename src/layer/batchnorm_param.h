/*
 * Relu_param.h
 *
 *  Created on: Jun 17, 2019
 *      Author: doyle
 */

#ifndef LAYER_BATCHNORM_PARAM_H_
#define LAYER_BATCHNORM_PARAM_H_

namespace tmnet
{
    class BatchNormParam
    {
    public:
        BatchNormParam();
        ~BatchNormParam();

        int zero;
        float one;
    };
}

#endif /* LAYER_BATCHNORM_PARAM_H_ */
