/*
 * ReLU.cpp
 *
 *  Created on: Jun 12, 2019
 *      Author: doyle
 */

#include "Eltwise.h"
#include <iostream>

namespace tmnet
{
	/*************************************************************************
	* Function Name : ReLU
	* Description   : construct function ,initialize all the variable
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	Eltwise::Eltwise()
	{
		//quantize value initialize
		iQuantizeFlag = 0;
		fBsQn = 0;
		//int32->fp32 value initialize
		iFp32Flag = 0;
		uiBsAluSrc = 0;
		uiBsMulSrc = 0;
		fOprand = 0;
		cRunFlag = 1;
		cConcatFlag = 0;
	}

	/*************************************************************************
	* Function Name : ~ReLU
	* Description   : deconstruct function
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	Eltwise::~Eltwise()
	{

	}

	/*************************************************************************
	* Function Name : loadParam
	* Description   : layer load param data
	* Parameters    : fileFp -- input param file
	* Returns       : 0 -- success
	**************************************************************************/
	int Eltwise::loadParam(FILE* fileFp, int output_num)
	{
		int id = 0;
		char cValue[50];

		while (fscanf(fileFp, "%d=%s", &id, &cValue) == 2)
		{
			switch (id)
			{
				case 0:
					break;
				default:
					break;
			}
		}
		return 0;
	}

	/*************************************************************************
	* Function Name : calculateFeaSize
	* Description   : calculate layer feature size
	* Parameters    : iIw -- input width
	* 				  iIh -- input height
	* 				  iIc -- input channel
	* Returns       : 0 -- success
	**************************************************************************/
	int Eltwise::calculateFeaSize(std::vector<unsigned int> iIw, std::vector<unsigned int> iIh, std::vector<unsigned int> iIc)
	{
		LayerCommon.iInFeaW = iIw[0];
		LayerCommon.iInFeaH = iIh[0];
		LayerCommon.cInputChannel = iIc[0];

		LayerCommon.iOutFeaW = iIw[0];
		LayerCommon.iOutFeaH = iIh[0];
		LayerCommon.cOutputChannel = iIc[0];

		//input:int8
		LayerCommon.uiInputSize = iIw[0]*iIh[0]*iIc[0];

		//output:int32
		LayerCommon.uiOutputSize = LayerCommon.iOutFeaW*\
								LayerCommon.iOutFeaH*\
								LayerCommon.cOutputChannel*sizeof(unsigned int);
		return 0;
	}

	/*************************************************************************
	* Function Name : fillDDRAddress
	* Description   : fill the bin data ddr address
	* Parameters    : uiLastAddr -- last address
	* Returns       : 0 -- success
	**************************************************************************/
	unsigned long long Eltwise::fillDDRAddress(unsigned long long uiLastAddr,const char* num)
	{
		uiLastAddr=uiLastAddr;
		return uiLastAddr;
	}

	/*************************************************************************
	* Function Name : getInputScale
	* Description   : get layer input scale value from bin file
	* Parameters    : fileFp -- input file
	* Returns       : 0 -- success
	**************************************************************************/
	int Eltwise::getInputScale(FILE *fileFp)
	{
		return 0;
	}

	/*************************************************************************
	* Function Name : setQuantize
	* Description   : set quantize value
	* Parameters    : iQuantize -- quantize enable flag
	*                 fScale -- quantize scale
	* Returns       : uiLastAddr -- next address
	**************************************************************************/
	void Eltwise::setQuantize(int iQuantize,float fIScale)
	{

	}

	/*************************************************************************
	* Function Name : setFp32
	* Description   : set fp32 translate value
	* Parameters    : iFp32 -- fp32 enable flag
	* 				  uiBiasAddr -- bias address
	* 				  uiWScaleAddr -- weight scale address
	* 				  fIScale -- input scale
	* Returns       : NULL
	**************************************************************************/
	void Eltwise::setFp32(int iFp32,unsigned long long uiBiasAddr,unsigned long long uiWScaleAddr,float fIScale)
	{

	}

	/*************************************************************************
	* Function Name : getNextInputOutputAddr
	* Description   : ge tNext Input Output Address
	* Parameters    : uiAddr -- input address
	* Returns       : uiNewAddr -- next layer input address
	**************************************************************************/
	unsigned long long Eltwise::getNextInputOutputAddr(unsigned long long uiAddr,unsigned int uiOneSeg,\
															unsigned int uiOriAddr,char cBufferNum)
	{
		return 0;
	}

	/*************************************************************************
	* Function Name : setRegisterValue
	* Description   : ge tNext Input Output Address
	* Parameters    : uiLastAddr -- last address
	*                 uiOneSeg -- one seg size
	*                 uiOriAddr -- original address
	*                 cBufferNum -- buffer number
	* Returns       : uiNewAddr -- next layer input address
	**************************************************************************/
	unsigned long long Eltwise::setRegisterValue(unsigned long long uiLastAddr,unsigned int uiOneSeg,\
												const unsigned int uiOriAddr,char cBufferNum)
	{
		return 0;
	}

	/*************************************************************************
	* Function Name : setRegisterValue
	* Description   : set layer register value
	* Parameters    : uiInputAddr -- data input address
	*                 uiOutputAddr -- data out address
	* Returns       : success
	**************************************************************************/
	int Eltwise::setRegisterValue(std::vector<unsigned long long> uiInputAddr, std::vector<unsigned long long> uiOutputAddr,const char* num)
	{
		memset(&ewregisters, 0, sizeof(ewregisters));
		row_in = 0;
		col_in = 0;
		row_out = LayerCommon.iOutFeaW;
		col_out = LayerCommon.iOutFeaH;
		channel_in = 0;
		channel_out = LayerCommon.cOutputChannel;
		kernel_size = 0;
		stride = 0;
		padding = 0;
		conv_bypass = 1;
		conv_mode = 0;
		conv_start = 0;
		int32_to_fp32_enable = 1;
		bn_enable = 1;
		prelu_enable = 0;
		fp32_to_int8_enable = 1;
		qn_enable = 1;
		col_seg = 0;
		din_select = 0;
		ew_enable = 1;
		sdp_bypass = 0;
		ew_num = 2;
		weight_seg_enable = 0;
		col_seg_out=0;
		cdp_bypass = 1;
		softmax_bypass = 1;
		PoolingSize = poolingsize;
	    PoolStride = poolingstride;
	    PoolPadding = poolingpadding;
		feature_seg_disable = 0;
		inputScale = row_out * col_out * channel_out * 2;
		
		/*if relu layer under ew layer*/
		if (cReLUFlag == 1)
		{
			relu_enable = 1;
		}
		else
		{
			relu_enable = 0;
		}
		
		if (fp16_enable == true)
		{
			fp16Enable = 1;
		}
		else
		{
			fp16Enable = 0;
		}
		/*if pooling layer under ew layer*/
		if (cPoolingFlag == 1)
		{
			pooling_enable = 1;
		}
		else
		{
			pooling_enable = 0;
		}
		
		if (inputScale > MAX_VALUE_ONCE)
		{
			// seg_num = ( inputScale / MAX_VALUE_ONCE / SEG_NUM_PRAM + 1 ) * SEG_NUM_PRAM;

            if((inputScale % MAX_VALUE_ONCE) != 0)
				seg_num = inputScale / MAX_VALUE_ONCE+1;
			else
			    seg_num = inputScale / MAX_VALUE_ONCE;
			
			for(int j=0; j< 100; j++)
            {
				if((inputScale % seg_num) != 0)
					seg_num = seg_num + 1;
			    else
				{
					break;
				}	
			}
		}
		else
		{
			seg_num = 1;
		}

		if (seg_num != 0)
		{
			seg_size = inputScale / seg_num;
		}
		else
		{
			tmtool_log(LOG_ERROR, "seg_num error");
		} 

		ewregisters.head = 0x2aa55;
		ewregisters.Conv_Ctrl = (feature_seg_disable<<22)+(qn_enable<<17)+(weight_seg_enable<<16)+(conv_mode<<13)+(padding<<9)+\
							  (stride<<5)+(kernel_size<<1)+(conv_bypass<<15)+conv_start;
		ewregisters.uiEWRegFeatureSrcAdd = uiInputAddr[1];
		// printf("EWuiInputAddr:********************0x%llx\n",uiInputAddr);
		ewregisters.uiEWRegFeatureDstAdd = uiOutputAddr[0];
		// printf("EWuiOutputAddr:********************0x%llx\n",uiOutputAddr);
		ewregisters.uiRegFeaSizeWidth = (row_out<<16)+row_in;
		ewregisters.uiRegFeaChannel1 = (channel_out<<16)+channel_in;
		ewregisters.uiRegArithB = row_in*col_in*channel_in;
		ewregisters.uiRegArithE = row_out*col_out*channel_out/(PoolStride*PoolStride);
		ewregisters.uiRegArithF = ((kernel_size*kernel_size)<<23)+(kernel_size*kernel_size*channel_in);

		//sdp
		ewregisters.Sdp_Ctrl = 0x01+(int32_to_fp32_enable<<1)+\
														(bn_enable<<2)+\
														(relu_enable<<3)+\
														(prelu_enable<<4)+\
														(fp32_to_int8_enable<<5)+\
														(qn_enable<<6)+\
														(din_select<<7)+\
														(ew_enable<<8)+\
														(pooling_enable<<9)+\
														(PoolingSize<<10)+\
														(PoolStride<<14)+\
														(PoolPadding<<18)+\
														(sdp_bypass<<23)+\
														(fp16Enable<<27);

		ewregisters.Sdp_Row_Col_Chanel = ((row_out*col_out) << 13) + channel_out;
		ewregisters.Qn_A = Qn_a;
		ewregisters.Sdp_Data_Num = row_out*col_out*channel_out;	
		ewregisters.Sdp_Seg_Size_Num_Ew = (seg_size << 16) + (seg_num << 2) + ew_num;	
		ewregisters.Sdp_Ew1_Src_Addr = uiInputAddr[0];
		ewregisters.Sdp_Ew2_Src_Addr = uiInputAddr[1];
		ewregisters.Sdp_Dst_Addr = uiOutputAddr[0];
		/*no use*/	
		ewregisters.Sdp_Ew1_Bs_A = 1 / ew1BnA;	   
		ewregisters.Sdp_Ew2_Bs_A = 1 / ew2BnA;
		ewregisters.Sdp_Matrix_Row_Col_In = (row_out << 16) + col_out;
		ewregisters.Sdp_Matrix_Row_Col_Out = ((row_out/PoolStride) << 16) + (col_out/PoolStride);
		ewregisters.Cdp_Ctrl = cdp_bypass << 7;
		ewregisters.Softmax_Ctrl = softmax_bypass << 1;
		return 0;
	}

	/*************************************************************************
	* Function Name : writeDDRInfoInputOutput
	* Description   : write input output ddr information to putput file
	* Parameters    : fileOutFp -- output file
	* Returns       : void
	**************************************************************************/
	void Eltwise::writeDDRInfoInputOutput(FILE *fileOutFp)
	{

	}

	/*************************************************************************
	* Function Name : writeBinFile
	* Description   : write layer data to bin file
	* Parameters    : iDropCount -- dropout and input layer number before this layer
	*			      fileInFp -- input ncnn bin file
	*			      fileOutFp -- out tmmodel bin file
	*			      iLayerIndex -- index in ncnn param file
	* Returns       : 0 -- success
	**************************************************************************/
	int Eltwise::writeBinFile(FILE *fileInFp,FILE *fileOutFp,int iLayerIndex,const char* num)
	{
		return 0;
	}

	/*************************************************************************
	* Function Name : writeddrBinFile
	* Description   : write register value to bin file
	* Parameters    : fileRp -- out tmmodel bin file after the layer date		   
	* Returns       : 0 -- success
	**************************************************************************/
	int Eltwise::writeddrBinFile(FILE *fileRp)
	{
		char cConvBufs[sizeof(ewregisters)];
		memcpy(cConvBufs,&ewregisters,sizeof(ewregisters));
		int iEwregistersLength = sizeof(ewregisters)/4;
		for (int i = 0; i < iEwregistersLength; i++)
		{
			for(int k=0; k< 4; k++)
			{
				fwrite(&cConvBufs[i*4+3-k],sizeof(char),1,fileRp);
			}
		}

		return 0;
	}
}

