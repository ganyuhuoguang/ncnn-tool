/*
 * conv_param.h
 *
 *  Created on: Jun 17, 2019
 *      Author: doyle
 */

#ifndef LAYER_CONV_PARAM_H_
#define LAYER_CONV_PARAM_H_

namespace tmnet
{
	typedef struct 
	{
		unsigned int head;
		unsigned int  Conv_Ctrl;	//R/W
		unsigned int  uiRegFeatureSrcAdd;	//R/W
		unsigned int  uiRegFeatureDstAdd;	//R/W
		unsigned int  uiRegWeightSrcAdd;	//R/W
		unsigned int   uiRegBiasSrcAdd;	//R/W
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
		unsigned int  uiRegArithG;
		
		//sdp
		unsigned int  Sdp_Ctrl;	//R/W
		unsigned int  Sdp_Row_Col_Chanel;	//R/W
		float  Qn_A;	//R/W
		unsigned int  Sdp_Data_Num;	//R/W
		unsigned int  Sdp_Seg_Size_Num_Ew;	//R/W
		unsigned int  Sdp_Ew1_Src_Addr = 0;	//R/W
		unsigned int  Sdp_Ew2_Src_Addr = 0;	//R/W
		unsigned int  Sdp_Ew3_Src_Addr = 0;	//R/W
		unsigned int  Sdp_Dst_Addr = 0;	//R/W
		unsigned int  Sdp_Ew1_Bs_A =0;	//R/W
		unsigned int  Sdp_Ew2_Bs_A =0;
		unsigned int  Sdp_Ew3_Bs_A =0;	//R/W
		unsigned int  Sdp_Matrix_Row_Col_In;	//R/W
		unsigned int  Sdp_Matrix_Row_Col_Out;	//R/W
		unsigned int  zero1[2] ={0};

		//cdp
		unsigned int  Cdp_Ctrl;	//R/W
		unsigned int  FeaSrcAddr;	//R/W
		unsigned int  FeaDstAddr;	//R/W
		unsigned int  FeaSegSize;	//R/W
		unsigned int  FeaChannel;	//R/W
		unsigned int  Cdp_Arith_A;	//R/W
		unsigned int  Cdp_Arith_B;	//R/W
		unsigned int  Cdp_Arith_C;	//R/W
		unsigned int  zero2[4] ={0};
		
		//softmax
		unsigned int  Softmax_Ctrl;	//R/W
		unsigned int  Classes_length;	//R/W
		unsigned int  zero3[2] ={0};

		//reserve
		unsigned int  zero4[15] ={0};
	}CONVREGISTERS;

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
		float LScale;
		unsigned int PSPLFlag;
		unsigned int DataSize;
		unsigned int DataAddr;

	}CONVHEAD;

	class ConvParam
	{
		public:
			ConvParam();
			~ConvParam();
			CONVREGISTERS convregisters;
			CONVHEAD convhead;
			unsigned int num_output;
			unsigned int kernel_w;
			unsigned int kernel_h;
			unsigned int dilation_size;
			unsigned int stride_w;
			unsigned int stride_h;
			unsigned int pad_w;
			unsigned int pad_h;
			unsigned int bias_term;
			unsigned int data_size;
			unsigned int inputsWeightsDatasIsInt8;
			//input scale value
			float	fInputScale;
			//data DDR address
			unsigned long long  uiWeightAddr;
			unsigned long long  uiBiasAddr;
			unsigned long long  uiAScaleAddr;
			//register
			unsigned long long  uiRegInFeatureAdd;	//R/W
			unsigned long long  uiRegOutFeatureAdd;	//R/W
			unsigned long long  uiRegKerWeightAdd;	//R/W
			unsigned int  uiRegKerSize;			//R/W
			unsigned int  uiRegFeaSize;			//R/W
			unsigned int  uiRegFeaChannel;		//R/W
			unsigned int  uiRegPadCtrl;			//R/W
			unsigned int  uiRegConvCtrl;		//R/W
			unsigned int  uiRegIfmSplTim;		//R/W
			unsigned int  uiRegRowPerLd;		//R/W
			unsigned int  uiRegRowLstLd;		//R/W

			unsigned int row_in;
			unsigned int col_in;
			unsigned int row_out;
			unsigned int col_out;
			unsigned int channel_in;
			unsigned int channel_out;
			unsigned int col_seg;
			unsigned int seg_num;
			unsigned int convSegNum;
			unsigned int col_seg_in_f;
			unsigned int col_seg_in_m;
			unsigned int col_seg_in_l;
			unsigned int colSegOutF;
			unsigned int colSegOutM;
			unsigned int colSegOutL;
			unsigned int conv_bypass;
			unsigned int col_seg_out;
			unsigned int kernel_size;
			unsigned int stride;
			unsigned int padding;
			unsigned int fp16Enable;
			unsigned int conv_mode;
			unsigned int conv_start;
			unsigned char cdp_bypass;
			unsigned char softmax_bypass;
			unsigned char PoolingSize;
			unsigned char  PoolStride;
			unsigned char  PoolPadding;	
			unsigned int upsampleMode;
			bool upsampleEnable;
			//ew
			// float qn_a;
			unsigned int int32_to_fp32_enable;
			unsigned int bn_enable;
			unsigned int relu_enable;
			unsigned int prelu_enable;
			unsigned int fp32_to_int8_enable;
			unsigned int qn_enable;
			unsigned int din_select;
			unsigned int ew_enable;
			unsigned int pooling_enable;
			unsigned int sdp_bypass;
			unsigned int weight_seg_enable;
			int iOverlapHeight;
			unsigned int feature_seg_disable;
			unsigned int channelOutAlign;
	};
}

#endif /* LAYER_CONV_PARAM_H_ */
