/*
 * conv_param.h
 *
 *  Created on: Jun 17, 2019
 *      Author: doyle
 */

#ifndef LAYER_PCONVDW_PARAM_H_
#define LAYER_PCONVDW_PARAM_H_

namespace tmnet
{
	typedef struct 
	{
		unsigned int head;
		unsigned int  Conv_Ctrl;	//R/W
		unsigned int  uiDWRegFeatureSrcAdd;	//R/W
		unsigned int  uiDWRegFeatureDstAdd;	//R/W
		unsigned int  uiDWRegWeightSrcAdd;	//R/W
		unsigned int  uiRegBiasSrcAdd;	//R/W
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
		unsigned int  zero =0;
		//sdp
	    unsigned int  Sdp_Ctrl;	//R/W
		unsigned int  Sdp_Row_Col_Chanel;	//R/W
		float  Qn_A;	//R/W
		unsigned int  Sdp_Data_Num;	//R/W
		unsigned int  Sdp_Seg_Size_Num_Ew ;	//R/W
		unsigned int  Sdp_Ew1_Src_Addr = 0;	//R/W
		unsigned int  Sdp_Ew2_Src_Addr = 0;	//R/W
		unsigned int  Sdp_Ew3_Src_Addr = 0;	//R/W
		unsigned int  Sdp_Dst_Addr = 0;	//R/W
		unsigned int  Sdp_Ew1_Bs_A = 0;	//R/W
		unsigned int  Sdp_Ew2_Bs_A = 0;
		unsigned int  Sdp_Ew3_Bs_A = 0;	//R/W
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

	}PCONVDWREGISTERS;

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

	}PCONVDWHEAD;

	class PConvDwParam
	{
	public:
		PConvDwParam();
		~PConvDwParam();
		PCONVDWREGISTERS convdwregisters;
		PCONVDWHEAD convdwhead;
		int       num_output;
		unsigned char kernel_w;
		unsigned char dilation_size;
		unsigned char stride_w;
		unsigned char pad_left;
		unsigned char pad_w;
		unsigned char bias_term;
		unsigned int  data_size;
		unsigned int  group;
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
	    unsigned int col_seg_in_f;
	    unsigned int col_seg_in_m;
	    unsigned int col_seg_in_l;
		unsigned int conv_bypass;
		unsigned int col_seg_out;
	    unsigned int kernel_size;
	    unsigned int stride;
	    unsigned int padding;
	    unsigned int conv_mode;
	    unsigned int conv_start;
		unsigned int cdp_bypass;
		unsigned int softmax_bypass;
		unsigned int PoolingSize;
		unsigned int PoolStride;
		unsigned int PoolPadding;	

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
		unsigned int weight_seg_enable;
		int iOverlapHeight;
		unsigned int feature_seg_disable;
	};
}



#endif /* LAYER_CONV_PARAM_H_ */
