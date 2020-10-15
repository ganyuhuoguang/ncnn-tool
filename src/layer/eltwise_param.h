/*
 * Relu_param.h
 *
 *  Created on: Jun 17, 2019
 *      Author: doyle
 */

#ifndef LAYER_ELTWISE_PARAM_H_
#define LAYER_ELTWISE_PARAM_H_

namespace tmnet
{
	
	typedef struct 
	{
		unsigned int head;
		unsigned int  Conv_Ctrl;	//R/W
		unsigned int  uiEWRegFeatureSrcAdd;	//R/W
		unsigned int  uiEWRegFeatureDstAdd;	//R/W
		unsigned int  uiRegWeightSrcAdd = 0;	//R/W
		unsigned int  uiRegBiasSrcAdd = 0;	//R/W
		unsigned int  uiRegFeaSizeWidth;			//R/W
		unsigned int  uiRegFeaSizeSeg0;			//R/W
		unsigned int  uiRegFeaSizeSeg1;			//R/W
		unsigned int  uiRegFeaChannel1;		//R/W
		unsigned int  uiRegArithA;			//R/W
		unsigned int  uiRegArithB;		//R/W
		unsigned int  uiRegArithC;		//R/W
		unsigned int  uiRegArithD;		//R/W
		unsigned int  uiRegArithE;		//R/W
		unsigned int  uiRegArithF;		//R/W
		unsigned int  reserveConv =0;
		//sdp
	    unsigned int  Sdp_Ctrl;	//R/W
		unsigned int  Sdp_Row_Col_Chanel;	//R/W
		float  Qn_A;	//R/W
		unsigned int  Sdp_Data_Num;	//R/W
		unsigned int  Sdp_Seg_Size_Num_Ew;	//R/W
		unsigned int  Sdp_Ew1_Src_Addr;	//R/W
		unsigned int  Sdp_Ew2_Src_Addr;	//R/W
		unsigned int  Sdp_Ew3_Src_Addr = 0;	//R/W
		unsigned int  Sdp_Dst_Addr;	//R/W
		float  Sdp_Ew1_Bs_A;	//R/W
		float  Sdp_Ew2_Bs_A;
		float  Sdp_Ew3_Bs_A;	//R/W
		unsigned int  Sdp_Matrix_Row_Col_In;	//R/W
		unsigned int  Sdp_Matrix_Row_Col_Out;	//R/W
		unsigned int  reserveSdp[2] ={0};

		//cdp
		unsigned int  Cdp_Ctrl;	//R/W
		unsigned int  FeaSrcAddr;	//R/W
		unsigned int  FeaDstAddr;	//R/W
		unsigned int  FeaSegSize;	//R/W
		unsigned int  FeaChannel;	//R/W
		unsigned int  Cdp_Arith_A;	//R/W
		unsigned int  Cdp_Arith_B;	//R/W
		unsigned int  Cdp_Arith_C;	//R/W
		unsigned int  reserveCdp[4] ={0};
	
		//softmax
		unsigned int  Softmax_Ctrl;	//R/W
		unsigned int  Classes_length;	//R/W
		unsigned int  reserveSoftmax[2] ={0};

		//reserve
		unsigned int  reserve[15] ={0};

	}EWREGISTERS;

	class EltwiseParam
	{
	public:
		EltwiseParam();
		~EltwiseParam();
		EWREGISTERS ewregisters;
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

	    unsigned int row_in;
	    unsigned int col_in;
	    unsigned int row_out;
	    unsigned int col_out;
	    unsigned int channel_in;
	    unsigned int channel_out;
	    unsigned int col_seg;
	    unsigned int col_seg_in_f;
	    unsigned int col_seg_in_m;
	    unsigned int col_seg_in_l;
		unsigned char conv_bypass;
		unsigned int col_seg_out;
	    unsigned char kernel_size;
	    unsigned char stride;
	    unsigned char padding;
		unsigned int fp16Enable;
	    // unsigned char pooling_size;
		/* 0: normal mode
		 * 1: depthwise convolution mode
		 * 2: deconvolution mode
		 * 3: de-depthwise convolution mode
		 */
	    unsigned char conv_mode;
		/* convolution start signal for DPU RTL */
	    unsigned char conv_start;
		unsigned char cdp_bypass;
		unsigned char softmax_bypass;

		unsigned char PoolingSize;
		unsigned char  PoolStride;
		unsigned char  PoolPadding;	

		//ew
		// float qn_a;
	    unsigned char int32_to_fp32_enable;
	    unsigned char bn_enable;
	    unsigned char relu_enable;
	    unsigned char prelu_enable;
	    unsigned char fp32_to_int8_enable;
	    unsigned char qn_enable;
		unsigned char din_select;
	    unsigned char ew_enable;
	    unsigned char pooling_enable;
	    unsigned char sdp_bypass;
	
	    unsigned int seg_size;
		unsigned int seg_num;
		unsigned int ew_num;
		unsigned int ew1_src_addr;
		unsigned int ew2_src_addr;
		unsigned int ew3_src_addr;
		unsigned int ew_dst_addr;
		unsigned int ew1_bs_a;
		unsigned int ew2_bs_a;
		unsigned int ew3_bs_a;
	    unsigned char weight_seg_enable;
		unsigned int feature_seg_disable;
	};
}

#endif /* LAYER_RELU_PARAM_H_ */
