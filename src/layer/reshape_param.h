/*
 * Relu_param.h
 *
 *  Created on: Jun 17, 2019
 *      Author: doyle
 */

#ifndef LAYER_Reshape_PARAM_H_
#define LAYER_Reshape_PARAM_H_

namespace tmnet
{
	class ReshapeParam
	{
	public:
		ReshapeParam();
		~ReshapeParam();

	    int dim[3];
	    int dim4;
	};
}
#endif /* LAYER_RELU_PARAM_H_ */
