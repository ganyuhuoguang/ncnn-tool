/*
 * Relu_param.h
 *
 *  Created on: Jun 17, 2019
 *      Author: doyle
 */

#ifndef LAYER_DETECTIONOUTPUT_PARAM_H_
#define LAYER_DETECTIONOUTPUT_PARAM_H_

namespace tmnet{
    typedef struct 
	{
		unsigned int InFeaW;
	    unsigned int InFeaH;
	    unsigned int InputChannel;
		unsigned int OutFeaW;
	    unsigned int OutFeaH;
	    unsigned int OutputChannel;
		unsigned int FeaInAddr;
		unsigned int FeaOutAddr;
		//float LScale;
		unsigned int PSPLFlag;
		unsigned int DataSize;
		unsigned int DataAddr;
        unsigned int mboxLocAddr;
        unsigned int mboxConfAddr;

        int num_output;
        int num_class;
        float nms_threshold;
        int nms_top_k;
        int keep_top_k;
        float confidence_threshold;
        float variances[4];
        

	}DETECTIONOUTPUTHEAD;
    class DetectionOutputParam
    {
    public:
        DetectionOutputParam();
        ~DetectionOutputParam();

        DETECTIONOUTPUTHEAD DetectionOutputhead;
        unsigned int  data_size;
        int       num_output;
        int num_class;
        float nms_threshold;
        int nms_top_k;
        int keep_top_k;
        float confidence_threshold;
        float variances[4];
    };
}



#endif /* LAYER_DetectionOutput_PARAM_H_ */
