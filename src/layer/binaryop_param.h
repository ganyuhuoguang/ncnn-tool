/*
 * Relu_param.h
 *
 *  Created on: Jun 17, 2019
 *      Author: doyle
 */

#ifndef LAYER_BINARYOP_PARAM_H_
#define LAYER_BINARYOP_PARAM_H_

namespace tmnet
{
    class BinaryOpParam
    {
    public:
        BinaryOpParam();
        ~BinaryOpParam();

        int zero;
        int one;
        float two;
    };
}

#endif /* LAYER_BINARYOP_PARAM_H_ */
