/*
 * Convolution.cpp
 *
 *  Created on: Jun 11, 2019
 *      Author: doyle
 */

#include "Convolution.h"
#include <iostream>
#include <limits.h>
#include <math.h>

namespace tmnet
{
	/*************************************************************************
	* Function Name : Convolution
	* Description   : construct function ,initialize all the variable
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	Convolution::Convolution()
	{
	//	memset(&convParam,0,sizeof(CONV_PARAM));
		uiIfmSplTim = 0;
		inputsWeightsDatasIsInt8 = 0;
		cRunFlag = 1;
		cConcatFlag = 0;
		reluWriteBinFlag = false;
	}

	/*************************************************************************
	* Function Name : ~Convolution
	* Description   : deconstruct function
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	Convolution::~Convolution()
	{

	}

	/*************************************************************************
	* Function Name : setLayerBinDataSize
	* Description   : set Layer Bin Data Size
	* Parameters    : NULL
	* Returns       : NULL
	**************************************************************************/
	void Convolution::setLayerBinDataSize(void)
	{
		//weight:int8
		//bias:float
		//weight scale:float
		//input scale:float
		if(cFirstFlag == 1)
		{
			uiBinDataSize = (data_size/(RGB_CHANEL_NUM_SIZE))*(DPU_INPUT_CHANEL_NUM_SIZE) + (CONV_OUT_BIAS_SCALE_IN_BIN + CONV_OUT_A_SCALE_IN_BIN)*num_output*sizeof(float);//3　（Ｂias + ascale） 放在并文件中的
		}
		else
		{  
			uiBinDataSize = data_size + (CONV_OUT_BIAS_SCALE_IN_BIN + CONV_OUT_A_SCALE_IN_BIN) * num_output * sizeof(float);
		}
	}

	/*************************************************************************
	* Function Name : loadParam
	* Description   : layer load param data
	* Parameters    : fileFp -- input param file
	* Returns       : 0 -- success
	**************************************************************************/
	int Convolution::loadParam(FILE *fileFp, int output_num)
	{
		int id = 0;
		int value = 0;
		pad_w = UCHAR_MAX;
		pad_h = UCHAR_MAX;
		kernel_w = UCHAR_MAX;
		kernel_h = UCHAR_MAX;
		stride_w = UCHAR_MAX;
		stride_h = UCHAR_MAX;
		num_output = output_num;
		while (fscanf(fileFp, "%d=%d", &id, &value) == 2)  //need modify
		{
			switch (id)
			{
				case 0:
					num_output = value;  //output num
					break;
				case 1:
					kernel_w = value;    //kernel size 
					break;
				case 2:
					dilation_size = value;//
					break;
				case 3:
					stride_w = value;
					break;
				case 4:
					pad_w = value;
					break;
				case 5:
					bias_term = value;
					break;
				case 6:
					data_size = value;
					break; 
				case 8:
				    inputsWeightsDatasIsInt8 = value; //value：1 ConvolutionDepthWise, 2 inputs Weights Data int8, no value: inputs Weights Data FP32
					break;
				case 11:
					kernel_h = value;
					break;
				case 13:
					stride_h = value;
					break;
				case 14:
					pad_h = value;
					break;
				default:
					break;
			}

			if (pad_h == UCHAR_MAX)
			{
				pad_h = pad_w;
			}
			if (kernel_h == UCHAR_MAX)
			{
				kernel_h = kernel_w;
			}
			if (stride_h == UCHAR_MAX)
			{
				stride_h = stride_w;
			}
		}
		setLayerBinDataSize();
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
	int Convolution::calculateFeaSize(std::vector<unsigned int> iIw, std::vector<unsigned int> iIh, std::vector<unsigned int> iIc)
	{
		LayerCommon.iInFeaW = iIw[0];
		LayerCommon.iInFeaH = iIh[0];
		LayerCommon.cInputChannel = iIc[0];
		int dilation_w = 1;//todo
		int dilation_h = 1;

		const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
		const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
		if (pad_w >= 0 || pad_h >= 0)
		{
			LayerCommon.iOutFeaW = (LayerCommon.iInFeaW + 2*pad_w - kernel_w) / stride_w + 1;
			LayerCommon.iOutFeaH = (LayerCommon.iInFeaH + 2*pad_h - kernel_h) / stride_h + 1;
		}
		else if (pad_w == CONV_PADW_PADH_NO_DEFAULT && pad_h == CONV_PADW_PADH_NO_DEFAULT)
		{
			int wpad = kernel_extent_w + (LayerCommon.iInFeaW - 1) / stride_w * stride_w - LayerCommon.iInFeaW;
			int hpad = kernel_extent_h + (LayerCommon.iInFeaH - 1) / stride_h * stride_h - LayerCommon.iInFeaH;
	
			LayerCommon.iOutFeaW = (LayerCommon.iInFeaW + wpad / 2 + wpad - wpad / 2 - kernel_w) / stride_w + 1;
			LayerCommon.iOutFeaH = (LayerCommon.iInFeaW + hpad / 2 + hpad - hpad / 2 - kernel_h) / stride_h + 1;
		}
		LayerCommon.cOutputChannel = num_output;
		//input:int8
		if (cFirstFlag)
		{
			LayerCommon.uiInputSize = iIw[0]*iIh[0]*FirstLayerChannel;//int8
		}
		else
		{
			LayerCommon.uiInputSize = iIw[0]*iIh[0]*iIc[0];
		}
		//output:int32
		LayerCommon.uiOutputSize = LayerCommon.iOutFeaW*\
								LayerCommon.iOutFeaH*\
								LayerCommon.cOutputChannel * 2;
		return 0;
	}

	/*************************************************************************
	* Function Name : getNewAddrFormDataSize
	* Description   : calculate the next bin data address (4K Algin)
	* Parameters    : NULL
	* Returns       : uiNewAddress -- nextt address
	**************************************************************************/
	uint64 Convolution::getNewAddrFormDataSize(unsigned int uiDataSize,uint64 uiAddress)
	{
		const unsigned int uiOne= ALGN_SIZE;
		unsigned int uiNewAddress = 0;
		unsigned int ucStepCount = 0;

		if(uiDataSize%uiOne >=0)
		{
			ucStepCount++;
		}
		ucStepCount += (uiDataSize/uiOne);

		uiNewAddress = uiAddress;
		uiNewAddress += (ucStepCount*uiOne);

		return uiNewAddress;
	}

	/*************************************************************************
	* Function Name : fillDDRAddress
	* Description   : fill the bin data ddr address
	* Parameters    : uiLastAddr -- last address
	* Returns       : 0 -- success
	**************************************************************************/
	uint64 Convolution::fillDDRAddress(uint64 uiLastAddr,const char* num)
	{
		unsigned int uiDataSize;
		float fBuf = 0;
		//write weight address and calculate the size of weight data    type:int8
		uiWeightAddr = uiLastAddr;
		if(cFirstFlag)//the first conv
		{
			fBuf = (float)data_size/LayerCommon.cInputChannel* atoi(num);
			uiDataSize = (unsigned int)fBuf;
		}
		else
		{
			uiDataSize = data_size;
		}

		uiLastAddr = getNewAddrFormDataSize(uiDataSize,uiLastAddr);
		
		//write bias address and calculate the size of bias data    type:float
		uiBiasAddr = uiLastAddr;
		if(cPreluFlag == 1)
		{
			uiDataSize = num_output*sizeof(float)*LAYER_OUT_PUT_BIN_ASCALE_BIAS_PRELU; //(Ascale +  Bias +  Prelu)
		}
		else
		{
			uiDataSize = num_output*sizeof(float)*LAYER_OUT_PUT_BIN_ASCALE_BIAS;
		}
		
		uiLastAddr = getNewAddrFormDataSize(uiDataSize,uiLastAddr);
		return uiLastAddr;
	}

	/*************************************************************************
	* Function Name : getInputScale
	* Description   : get layer input scale value from bin file
	* Parameters    : fileFp -- input file
	* Returns       : 0 -- success
	**************************************************************************/
	int Convolution::getInputScale(FILE *fileFp)
	{
		int rc = 0;
		union
		{
			struct
			{
				unsigned char f0;
				unsigned char f1;
				unsigned char f2;
				unsigned char f3;
			};
			unsigned int tag;
		} flag_struct;
		char *pcCharBuf = (char *)malloc(data_size);
		char *pcConvBias = (char *)malloc(num_output*sizeof(float));
		if (fileFp == NULL)
		{
			return -1;
		}
		//read the tag
		rc = fread(&flag_struct, sizeof(flag_struct), 1,fileFp);//0x000D4B38
		//read weight
		rc = fread(pcCharBuf, data_size, 1, fileFp);
		assert(rc > 0);
		//read bias
		rc = fread(pcConvBias, num_output * sizeof(float), 1, fileFp);
		assert(rc > 0);
		//read weight scale
		rc = fread(pcConvBias, num_output * sizeof(float), 1, fileFp);
		assert(rc > 0);
		//read input scale
		rc = fread(pcConvBias, sizeof(float), 1, fileFp);
		assert(rc > 0);
		//save input scale
		float fBuf = fInputScale;
		memcpy(&fBuf,pcConvBias,sizeof(float));
		fInputScale = fBuf;
		Qn_a = fBuf;
		inputScale = fBuf;
		free(pcCharBuf);
		free(pcConvBias);
		return rc;
	}

	/*************************************************************************
	* Function Name : getNextInputOutputAddr
	* Description   : ge tNext Input Output Address
	* Parameters    : uiAddr -- input address
	* Returns       : uiNewAddr -- next layer input address
	**************************************************************************/
	uint64 Convolution::getNextInputOutputAddr(uint64 uiAddr,unsigned int uiOneSeg,\
															unsigned int uiOriAddr,char cBufferNum)
	{
		unsigned int uiNewAddr = 0;

		if( uiAddr >= ((cBufferNum-1)*uiOneSeg + uiOriAddr) )
		{
			//the last buffer
			uiNewAddr = uiOriAddr;
		}
		else
		{
			uiNewAddr = uiAddr + uiOneSeg;
		}

		return uiNewAddr;
	}

	/*************************************************************************
	* Function Name : setRegisterValue
	* Description   : set layer register value
	* Parameters    : NULL
	* Returns       : uiLastAddr -- next address
	**************************************************************************/
	uint64 Convolution::setRegisterValue(uint64 uiLastAddr,unsigned int uiOneSeg,\
														const unsigned int uiOriAddr,char cBufferNum)
	{
		//conv layer input data addr
		uiRegInFeatureAdd = uiLastAddr;
		uiLastAddr = getNextInputOutputAddr(uiLastAddr,uiOneSeg,uiOriAddr,cBufferNum);
		//conv layer output data addr
		uiRegOutFeatureAdd = uiLastAddr;
		//conv layer weights address
		uiRegKerWeightAdd = uiWeightAddr;
		//conv layer kernel size register
		uiRegKerSize = convRegKernelSize();
		//conv layer feature size
		uiRegFeaSize = convRegFeatureSize();
		//conv layer feature channel
		uiRegFeaChannel = convRegFeatureChannel();
		//conv layer padding control
		uiRegPadCtrl = convRegPadCtrl();
		//conv layer conv ctrl
		uiRegConvCtrl = 1;
		//conv layer IFM_SPL_TIM
		uiRegIfmSplTim = convRegIFM_SPL_TIM();
		//conv layer ROW_PER_LD
		uiRegRowPerLd = convRegROW_PER_LD();
		//conv layer ROW_LST_LD
		uiRegRowLstLd = convRegROW_LST_LD();
		return uiLastAddr;
	}

	/*************************************************************************
	* Function Name : setRegisterValue
	* Description   : set layer register value
	* Parameters    : uiInputAddr -- input data address
	*                 uiOutputAddr -- output data address
	* Returns       : uiLastAddr -- next address
	**************************************************************************/
	int Convolution::setRegisterValue(std::vector<unsigned long long> uiInputAddr, std::vector<unsigned long long> uiOutputAddr,const char* num)
	{   
		memset(&convregisters, 0, sizeof(convregisters));
		unsigned int channelNumber = atoi(num);
		row_in = LayerCommon.iInFeaW;
		col_in = LayerCommon.iInFeaH;
		row_out= LayerCommon.iOutFeaW;
		col_out= LayerCommon.iOutFeaH;
		channel_out = LayerCommon.cOutputChannel;
		kernel_size = kernel_w;
		stride = stride_w;
		padding = pad_w;
		conv_bypass = 0;
		conv_mode = 0;
		conv_start = 1;
		int32_to_fp32_enable = 1;
		bn_enable = 1;
		din_select = 1;
		ew_enable = 0;
		sdp_bypass = 0;
		// col_seg_out=col_seg/stride;
		cdp_bypass = 1;
		PoolingSize = poolingsize;
		//PoolStride = 2;
		PoolStride = poolingstride;
		PoolPadding = poolingpadding;	
    

		if (row_out == 1 && col_out == 1 && row_in == kernel_size)
		{
			stride = kernel_size;
		}
		else
		{
			stride = stride_w;
		}

		if (fp16_enable)
		{
			fp16Enable = 1;
		}
		else
		{
			fp16Enable = 0;
		}
	
		if (cReLUFlag == true)
		{
			relu_enable = 1;
		}
		else
		{
			relu_enable = 0;
		}

		if(cFirstFlag == 1)
		{
			channel_in = FirstLayerChannel;
			// PoolStride = 2;
		}
		else
		{
			channel_in = LayerCommon.cInputChannel;
			// PoolStride = 1;
		}
		
		if (cPreluFlag == 1)
		{
			prelu_enable = 1;
		}
		else
		{
			prelu_enable = 0;
		}

		if (cPoolingFlag == 1)
		{
			pooling_enable = 1;
		}
		else
		{
			pooling_enable = 0;
		}
		
		if (cCloseQnFlag == true)
		{
			fp32_to_int8_enable = 0;
			qn_enable = 0;
		}
		else
		{
			fp32_to_int8_enable = 1;
			qn_enable = 1;
		}

		if (cUpsampleFlag == true)
		{
			upsampleMode = 0;
			upsampleEnable = true;
			PoolingSize = 1;	
		}
		else
		{
			upsampleMode = 0;
			upsampleEnable = false;
		}
		
		
    	//  feature分段使能
    	 if(row_in*col_in > FRAMSIZE)
        {
            if((row_in*col_in%FRAMSIZE) != 0)
            {
                convSegNum =row_in*col_in/FRAMSIZE+1;
            }
            else
            {
                convSegNum =row_in*col_in/FRAMSIZE;
            }
            col_seg=col_in/convSegNum ;
        }
        else
        {
            col_seg =col_in;
            convSegNum =1;
        }
        
        if(convSegNum  > 1)
        {
            for(int j=0; j< 100; j++)
            {
                if(convSegNum  < 3)            //  分2段
                {
                    if(((col_seg+padding-stride)%stride) != 0)
                        col_seg_in_f=col_seg - 1;
                    else
                        col_seg_in_f=col_seg;    
                    
                    col_seg_in_l=col_in-col_seg_in_f;
                    col_seg_in_m=col_seg_in_l;
                    
                    colSegOutF=((col_seg_in_f+padding-stride)/stride)+1;
                    colSegOutL=((col_seg_in_l-kernel_size+padding)/stride)+1;
                    colSegOutM=colSegOutL;
                }
                else                            //  分3段
                {
                    if(((col_seg+padding-stride)%stride) != 0)
                        col_seg_in_f=col_seg - 1;
                    else
                        col_seg_in_f=col_seg;
                    
                    if(((col_seg-stride)%stride) != 0)
                        col_seg_in_m=col_seg+1;
                    else
                        col_seg_in_m=col_seg;
                    
                    col_seg_in_l=col_in-col_seg_in_f-(convSegNum -2)*col_seg_in_m;
                    
                    colSegOutF=((col_seg_in_f+padding-stride)/stride)+1;
                    colSegOutM=((col_seg_in_m-stride)/stride)+1;
                    colSegOutL=((col_seg_in_l-kernel_size+padding)/stride)+1;
                }
                if(!cFirstFlag)
                {
                    if((col_seg_in_f*row_in < FRAMSIZE) & (col_seg_in_m*row_in < FRAMSIZE) & (col_seg_in_l*row_in < FRAMSIZE))
                    {
                        break;
                    }
                    else
                    {
                        convSegNum =convSegNum +1;
                        col_seg=col_in/convSegNum ;
                    }
                }
                else
                {
                    if((colSegOutF*row_out < FRAMSIZE) & (colSegOutM*row_out < FRAMSIZE) & (colSegOutL*row_out < FRAMSIZE))
                    {
                        break;
                    }
                    else
                    {
                        convSegNum =convSegNum +1;
                        col_seg=col_in/convSegNum ;
                    }
                }                    
            }    
        }
        else                 //  不分段
        {
            col_seg_in_f = col_seg;
            col_seg_in_m = col_seg;
            col_seg_in_l = col_seg;
            
            colSegOutF = ((col_seg-kernel_size+2*padding)/stride)+1;
            colSegOutM = ((col_seg-kernel_size+2*padding)/stride)+1;
            colSegOutL = ((col_seg-kernel_size+2*padding)/stride)+1;
        }

		if(channel_out > 0)
		{
			if (channel_out <= channelNumber)
			{
				channelOutAlign = channelNumber;
			}
			else
			{
				if (!(channel_out % channelNumber))
				{
					channelOutAlign = channel_out;
				}
				else
				{
					channelOutAlign = ((channel_out / channelNumber)+1) * channelNumber;
				}
			}
			
		}
		else
		{
			tmtool_log(LOG_ERROR, "channel_in error");
		}
		
		iOverlapHeight = kernel_size - stride;
		if(conv_mode == 0)
	    {
	        if (kernel_size - stride > 1)
            {
                weight_seg_enable = 1;
            }
            else if (kernel_size*kernel_size*channel_in > WRAMSIZE)
            {
                weight_seg_enable = 1;
            }
            else
            {
                weight_seg_enable = 0;
            }
	    }
		else
		{
			weight_seg_enable = 0;
		}
        
		if((row_in * col_in * channel_in / 8) < FRAMSIZE)
		{
			feature_seg_disable = 1;
		}
		else
		{
			feature_seg_disable = 0;
		}

		if(stride > 0)
		{
			col_seg_out=col_seg/stride;
		}
	
		if(iOverlapHeight < 0)
		{
			iOverlapHeight = 0;
		}
		

		if (PoolStride == 0)
		{
			tmtool_log(LOG_ERROR, "PoolStride error");
		} 

		if (col_seg == 0)
		{
			tmtool_log(LOG_ERROR, "col_seg error");
		} 
		// printf("row_in               :%d \n", row_in               );  
		// printf("col_in               :%d \n", col_in               );  
		// printf("row_out              :%d \n", row_out              );  
		// printf("col_out              :%d \n", col_out              );  
		// printf("channel_in           :%d \n", channel_in           );  
		// printf("channel_out          :%d \n", channel_out          );  
		// printf("col_seg              :%d \n", col_seg              );  
		// printf("conv_bypass          :%d \n", conv_bypass          );  
		// printf("kernel_size          :%d \n", kernel_size          );  
		// printf("stride               :%d \n", stride               );  
		// printf("padding              :%d \n", padding              );
		// printf("conv_mode            :%d \n", conv_mode            );
		// printf("int32_to_fp32_enable :%d \n", int32_to_fp32_enable );
		// printf("bn_enable            :%d \n", bn_enable            );  
		// printf("relu_enable          :%d \n", relu_enable          );  
		// printf("prelu_enable         :%d \n", prelu_enable         );  
		// printf("fp32_to_int8_enable  :%d \n", fp32_to_int8_enable  );  
		// printf("qn_enable            :%d \n", qn_enable            );  
		// printf("din_select           :%d \n", din_select           );  
		// printf("ew_enable            :%d \n", ew_enable            );  
		// printf("pooling_enable       :%d \n", pooling_enable       );  
		// printf("sdp_bypass           :%d \n", sdp_bypass           );  
		// printf("PoolingSize          :%d \n", PoolingSize          );  
		// printf("PoolStride           :%d \n", PoolStride           );  
		// printf("PoolPadding          :%d \n",PoolPadding           );  

		convregisters.head = 0x3aa55;
    	convregisters.Conv_Ctrl = (feature_seg_disable<<22)+(qn_enable<<17)+(weight_seg_enable<<16)+(conv_mode<<13)+(padding<<9)+\
								(stride<<5)+(kernel_size<<1)+(conv_bypass<<15)+conv_start;
		convregisters.uiRegFeatureSrcAdd = uiInputAddr[0];
		convregisters.uiRegFeatureDstAdd = uiOutputAddr [0];
		convregisters.uiRegWeightSrcAdd = uiWeightAddr;
		// printf("convregisters uiInputAddr: 0x%lx, uiOutputAddr: 0x%lx.\n", uiInputAddr, uiOutputAddr);
		convregisters.uiRegBiasSrcAdd = uiBiasAddr;
		convregisters.uiRegFeaSizeWidth = (row_out << 16) + row_in;
		convregisters.uiRegFeaSizeSeg0 = (col_seg_in_l << 20) + (col_seg_in_m<<10)+col_seg_in_f;
		convregisters.uiRegFeaSizeSeg1 = ((iOverlapHeight)*row_in<<18)+(convSegNum<<10)+ colSegOutF;
		convregisters.uiRegFeaChannel1 = (channelOutAlign << 16) + channel_in;
		convregisters.uiRegArithA = row_in*col_in;
		convregisters.uiRegArithB = row_in*col_in*channel_in;
		convregisters.uiRegArithC = (col_seg_in_m * row_in << 16) + col_seg_in_f*row_in;
		convregisters.uiRegArithD = col_seg_in_l*row_in;;

		convregisters.uiRegArithF = ((kernel_size*kernel_size) << 23) + (kernel_size*kernel_size*channel_in);
		convregisters.uiRegArithG = (colSegOutL<<10) + colSegOutM;

		//sdp
		convregisters.Sdp_Ctrl = 0x01+(int32_to_fp32_enable<<1)+\
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
														(upsampleEnable<<24)+\
														(upsampleMode<<25)+\
														(fp16Enable<<27);

		convregisters.Sdp_Row_Col_Chanel = ((row_out*col_out)<<13)+channelOutAlign;
		convregisters.Sdp_Data_Num = row_out*col_out*channelOutAlign;
		convregisters.Sdp_Matrix_Row_Col_In = (row_out<<16)+col_out;
		if (cUpsampleFlag == true)
		{
			convregisters.Sdp_Matrix_Row_Col_Out = ((row_out*(PoolingSize + 1))<<16)+(col_out*(PoolingSize + 1));
			if (cCloseQnFlag == false && fp16_enable == false)
			{
				convregisters.uiRegArithE = row_out*(PoolingSize + 1)*col_out*(PoolingSize + 1)*channelOutAlign;
			}
			else if (cCloseQnFlag == true && fp16_enable == true)
			{
				convregisters.uiRegArithE = row_out*(PoolingSize + 1)*col_out*(PoolingSize + 1)*channelOutAlign*2;
			}
			else if (cCloseQnFlag == false && fp16_enable == true)
			{
				convregisters.uiRegArithE = row_out*(PoolingSize + 1)*col_out*(PoolingSize + 1)*channelOutAlign*4;
			}
			else if (cCloseQnFlag == true && fp16_enable == false)
			{
				convregisters.uiRegArithE = row_out*(PoolingSize + 1)*col_out*(PoolingSize + 1)*channelOutAlign*4;
			}	
		}
		else
		{
			convregisters.Sdp_Matrix_Row_Col_Out = ((row_out/PoolStride)<<16)+(col_out/PoolStride);
			if (cCloseQnFlag == false && fp16_enable == false)
			{
				convregisters.uiRegArithE = row_out*col_out*channelOutAlign/(PoolStride*PoolStride);
			}
			else if (cCloseQnFlag == true && fp16_enable == true)
			{
				convregisters.uiRegArithE = row_out*col_out*channelOutAlign/(PoolStride*PoolStride)*2;
			}
			else if (cCloseQnFlag == false && fp16_enable == true)
			{
				convregisters.uiRegArithE = row_out*col_out*channelOutAlign/(PoolStride*PoolStride)*4;
			}
			else if (cCloseQnFlag == true && fp16_enable == false)
			{
				convregisters.uiRegArithE = row_out*col_out*channelOutAlign/(PoolStride*PoolStride)*4;
			}	
				
		}		
		convregisters.Qn_A = Qn_a; //FP32 -> int8
		convregisters.Cdp_Ctrl = cdp_bypass<<7;
		if(cSoftMaxFlag == 1)
		{
			convregisters.Softmax_Ctrl = 1;
		}
		else
		{
			convregisters.Softmax_Ctrl = 2;
		}
		convregisters.Classes_length = SOFTMAX_OUTPUT_LENGTH;
		return 0;
	}

	/*************************************************************************
	* Function Name : writeDDRInfoWeight
	* Description   : write weight data information to output file
	* Parameters    : fileOutFp -- output file
	* Returns       : void
	**************************************************************************/
	void Convolution::writeDDRInfoWeight(FILE *fileOutFp)
	{
		char cPrintfbuf[100];
		sprintf(cPrintfbuf,"conv\n");
		fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
		sprintf(cPrintfbuf,"weight address %llx\n",uiRegKerWeightAdd);
		fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
		sprintf(cPrintfbuf,"bias address %llx\n",uiBiasAddr);
		fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
		sprintf(cPrintfbuf,"weight scale address %llx\n",uiAScaleAddr);
		fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
	}

	/*************************************************************************
	* Function Name : writeDDRInfoInputOutput
	* Description   : write input output data information to output file
	* Parameters    : fileOutFp -- output file
	* Returns       : void
	**************************************************************************/
	void Convolution::writeDDRInfoInputOutput(FILE *fileOutFp)
	{
		char cPrintfbuf[100];

		sprintf(cPrintfbuf,"conv\n");
		fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);

		sprintf(cPrintfbuf,"input address %llx\n",uiRegInFeatureAdd);
		fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);

		sprintf(cPrintfbuf,"output address %llx\n",uiRegOutFeatureAdd);
		fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
	}

	/************************************************************************
	* Function Name : getFp32Infor
	* Description   : get fp32 information
	* Parameters    : uiBias -- output bias address
	* 				  uiWScale -- output weight scale address
	* 				  fIScale -- output scale
	* Returns       : 0 -- success
	**************************************************************************/
	int Convolution::getFp32Infor(uint64 *uiBias,uint64 *uiWScale,float *fIScale)
	{
		*uiBias = uiBiasAddr;
		*uiWScale = uiAScaleAddr;
		*fIScale = fInputScale;

		return 0;
	}

	/*************************************************************************
	* Function Name : getQuantizeInfor
	* Description   : get quantize information
	* Parameters    : fIScale -- output scale
	* Returns       : 0 -- success
	**************************************************************************/
	int Convolution::getQuantizeInfor(float *fIScale)
	{
		*fIScale = fInputScale;
		return 0;
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
	int Convolution::writeBinFile(FILE *fileInFp, FILE *fileOutFp, int iLayerIndex, const char* num)
	{
		char HEAD[sizeof(convhead)];
		unsigned int channelnumber = atoi(num);
		float *wscale = NULL;
		float iscale;
		union
		{
			struct
			{
				unsigned char f0;
				unsigned char f1;
				unsigned char f2;
				unsigned char f3;
			};
			unsigned int tag;
		} flag_struct;

		//write bin file data
		unsigned int iKernelSize = kernel_w*kernel_w;
		char *pcCharBuf = (char *)malloc(data_size);
		char *pcConvBias = (char *)malloc(num_output*sizeof(float));
		char *pcConvWscale = (char *)malloc(num_output*sizeof(float));
		char *pcConvLscale = (char *)malloc(num_output*sizeof(float));
		char cPrintBuf[PRINT_BUF_SIZE];
		int rc = 0;
		unsigned int k = 0;
		unsigned int tempId = 0;
		unsigned int addnum = 0;

		if(0 == num_output % channelnumber)
		{
			addnum = 0;
		}
		else
		{
			addnum = channelnumber - num_output % channelnumber;
		}

		if(pcCharBuf == NULL || pcConvBias == NULL)
		{
			tmtool_log(LOG_ERROR, "malloc wrong: %d", iLayerIndex);
			return -1;
		}
		if(channelnumber == 0)
		{
			tmtool_log(LOG_ERROR, "channelnumber wrong: %d", iLayerIndex);
			return -1;
		}
		//read the tag
		rc = fread(&flag_struct, sizeof(flag_struct), 1, fileInFp);
		assert(rc > 0);
		//read weight
		rc = fread(pcCharBuf, data_size, 1, fileInFp);
		assert(rc > 0);

		int input_num = LayerCommon.cInputChannel;
		int output_num = num_output;
		
		//add layerhead, size, address
		char cLayerHead[LAYER_HEADER_SIZE]={(char)0xAA,(char)0xBB,(char)0xCC,(char)0xDD,01};
		convhead.InFeaW = LayerCommon.iInFeaW;
		convhead.InFeaH = LayerCommon.iInFeaH;
		convhead.InputChannel = LayerCommon.cInputChannel;
		convhead.OutFeaW = LayerCommon.iOutFeaW;
		convhead.OutFeaH = LayerCommon.iOutFeaH;
		convhead.OutputChannel = LayerCommon.cOutputChannel;
		convhead.FeaInAddr = convregisters.uiRegFeatureSrcAdd;
		convhead.FeaOutAddr = convregisters.uiRegFeatureDstAdd;
		convhead.LScale = Qn_a;
		convhead.PSPLFlag = WeightAddrFlag;

		if(cFirstFlag == 1)
		{
			convhead.DataSize = (data_size + addnum*iKernelSize*input_num) / 3 * channelnumber;
		}	
		else
		{
			convhead.DataSize = data_size + addnum*iKernelSize*input_num;
		}
		
		convhead.DataAddr = convregisters.uiRegWeightSrcAdd;
		fwrite(cLayerHead, sizeof(char), sizeof(cLayerHead), fileOutFp);
		memcpy(HEAD, &convhead, sizeof(convhead));
		int ConvheadLength = sizeof(convhead)/4;
		for (int i = 0; i < ConvheadLength; i++)
		{
			for(k=0; k< 4; k++)
			{
				fwrite(&HEAD[i*4+3-k], sizeof(char), 1, fileOutFp);
			}
		}
		//write data conv weights
		if(LayerCommon.cInputChannel < DATA_CHANNEL_NUM)
		{
			for (unsigned int outid=0; outid < output_num / channelnumber; outid++)
			{
				for (unsigned int kid=0; kid < iKernelSize; kid++)
				{
					for (unsigned int w=0; w < channelnumber; w++)
					{
						int i = 0;
						fwrite(&i, sizeof(char), 1, fileOutFp);
						for (k = 0; k < 3; k++)
						{
							fwrite(&pcCharBuf[outid*input_num*iKernelSize*channelnumber + \
											w*input_num*iKernelSize + (2-k)*iKernelSize + kid], sizeof(char), 1, fileOutFp);
						}
						int n = 0;
						if (channelnumber == 8)
						{
							fwrite(&n, sizeof(char), 4, fileOutFp);
						}
						
						if (channelnumber == 16)
						{
							for (int i = 0; i<3; i++)
							{
								fwrite(&n, sizeof(char), 4, fileOutFp);
							}
						}
					}
				}
			}
		}
		else if(iKernelSize == 4*4) //change to 16 channel convlution kernel size of 4*4 -----  8 todo
		{
			char *extraPcCharBuf = (char *)malloc(data_size + addnum * iKernelSize * input_num);
			memset(extraPcCharBuf,0x00,sizeof(char)*(data_size + addnum * iKernelSize * input_num));
			memcpy(extraPcCharBuf,pcCharBuf,data_size);
			for(unsigned int inid=0; inid < (num_output + addnum) / channelnumber; inid++)
			{
				for(unsigned int outid=0; outid<input_num; outid++)
				{
					for(unsigned int kid=0; kid < iKernelSize / channelnumber; kid++)//
					{
						for(unsigned int j=0; j<channelnumber; j++)//channelnumber
						{
							for(k=0; k<channelnumber; k++)
							{
								tempId = inid * channelnumber * input_num * iKernelSize + j * input_num * iKernelSize + \
																outid * iKernelSize + kid * channelnumber;
								if(k<4)
								{
								fwrite(&extraPcCharBuf[tempId + (3-k)],sizeof(char),1,fileOutFp);
								}														
								else if((k>3)&&(k<8))
								{
								fwrite(&extraPcCharBuf[tempId + (11-k)],sizeof(char),1,fileOutFp);
								}
								else if((k>7)&&(k<12))
								{
								fwrite(&extraPcCharBuf[tempId + (19-k)],sizeof(char),1,fileOutFp);
								}
								else if((k>11)&&(k<16))
								{
								fwrite(&extraPcCharBuf[tempId + (27-k)],sizeof(char),1,fileOutFp);
								}	
							}
						}
					}
				}
			}
			free(extraPcCharBuf);
		}
			
		else
		{
			char *extraPcCharBuf = (char *)malloc(data_size + addnum * iKernelSize * input_num);
			memset(extraPcCharBuf,0x00,sizeof(char)*(data_size + addnum * iKernelSize * input_num));

			memcpy(extraPcCharBuf,pcCharBuf,data_size);
			for(unsigned int outid=0; outid<(num_output + addnum)/channelnumber; outid++)//num_output+num_output%channelnumber
			{
				for(unsigned int inid=0; inid<input_num/channelnumber; inid++)
				{
					for(unsigned int kid=0; kid<iKernelSize; kid++)
					{
						for(unsigned int w=0; w<channelnumber; w++)
						{
							for(k=0; k<channelnumber; k++)
							{
								tempId = outid*input_num*iKernelSize*channelnumber + w*input_num*iKernelSize+ \
													inid*iKernelSize*channelnumber + kid;
								if(k<4)
								{
								fwrite(&extraPcCharBuf[tempId + (3-k)*iKernelSize], sizeof(char), 1, fileOutFp);
								}
								else if((k>3) && (k<8))
								{
								fwrite(&extraPcCharBuf[tempId + (11-k)*iKernelSize], sizeof(char), 1, fileOutFp);
								}
								else if((k>7) && (k<12))
								{
								fwrite(&extraPcCharBuf[tempId + (19-k)*iKernelSize], sizeof(char), 1, fileOutFp);
									
								}
								else if((k>11) && (k<16))
								{
								fwrite(&extraPcCharBuf[tempId + (27-k)*iKernelSize], sizeof(char), 1, fileOutFp);
								}
								else
								{
								
								}
							} // end of k
						}// end of w
					}// end of kid
				}
			}
		free(extraPcCharBuf);
		}

		/* write bias data */
		char BiasLayerHead[LAYER_HEADER_SIZE] = {(char)0xAA, (char)0xBB, (char)0xCC, (char)0xDD,2};
		fwrite(BiasLayerHead, sizeof(char), sizeof(BiasLayerHead), fileOutFp);
		memset(&convhead, 0, sizeof(convhead));

		convhead.DataAddr = convregisters.uiRegBiasSrcAdd;
		if (cPreluFlag==1)
		{
			convhead.DataSize = (num_output + addnum) * 3 * 4;
		}
		else
		{
			convhead.DataSize = (num_output + addnum) * 2 * 4;
		}

		//rewrite convolution head tag
		memcpy(HEAD, &convhead, sizeof(convhead));
		ConvheadLength = sizeof(convhead) / 4;
		for (int i = 0; i < ConvheadLength; i++)
		{
			for(k = 0; k < 4; k++)
			{
				fwrite(&HEAD[i*4+3-k], sizeof(char), 1, fileOutFp);
			}
		}    

		/* read bias data from file input file stream */
		rc = fread(pcConvBias, num_output * sizeof(float), 1, fileInFp);
		assert(rc > 0);
		
		//write weight scale
		wscale = (float *)malloc(num_output * 4);
		rc = fread(pcConvWscale, num_output * sizeof(float), 1, fileInFp);
		memcpy(wscale, pcConvWscale, num_output * 4);
		for (unsigned int j=0; j<num_output; j++)
		{
			memcpy(cPrintBuf, &pcConvWscale[sizeof(float) * j], sizeof(cPrintBuf));
		}
		free(pcConvWscale);

		rc = fread(pcConvLscale, sizeof(float), 1, fileInFp);
		assert(rc > 0);
		memcpy(&iscale, pcConvLscale, 4);

		free(pcConvLscale);

		//write a scale
		for (unsigned int j=0; j<num_output + addnum; j++)
		{

			float a = 0.0f;
			if(fabs(wscale[j]) >= 1e-6 && fabs(iscale) >= 1e-6)
			{
				a = 1.0f / (wscale[j]*iscale);
			}
			else
			{
				a = 0.0f;
			}
			memcpy(cPrintBuf, &a, sizeof(cPrintBuf));
			for(k = 0; k < 4 ;k++)
			{
				if((0 != addnum)&&(j >= num_output ))
				{
					int i = 0;//add 0 for num_output%channels!=0
					
					fwrite(&i,sizeof(char),1,fileOutFp);
				}
				else
				{
					fwrite(&cPrintBuf[3-k], sizeof(char), 1, fileOutFp);
				}
			}
		}
			//write bias
		for (unsigned int j=0; j<num_output + addnum; j++)
		{
				memcpy(cPrintBuf, &pcConvBias[sizeof(float)*j], sizeof(cPrintBuf));
				int floatLenghth = sizeof(float);
			for(k = 0; k < sizeof(float); k++)
			{
				if((0 != addnum)&&(j >= num_output ))
				{
					int i = 0;
					fwrite(&i,sizeof(char),1,fileOutFp);
				}
				else
				{
					fwrite(&cPrintBuf[3-k], sizeof(char), 1, fileOutFp);
				}
			}
		}

		free(pcCharBuf);
		free(pcConvBias);
		free(wscale);
		return rc;
	}

	/*************************************************************************
	* Function Name : writeddrBinFile
	* Description   : write register instruction value to bin file
	* Parameters    : fileRp -- out tmmodel bin file after the layer date		   
	* Returns       : 0 -- success
	**************************************************************************/
	int Convolution::writeddrBinFile(FILE *fileRp)
	{
		int glb_length = 12;
		int glb_ctrl = 1;
		char HEAD[sizeof(convhead)];
		char cPrinBuf[4];
		char cConvBufs[sizeof(convregisters)];
		
		if (cFirstFlag == 1)
		{
			//glb
			char GlbLayerHead[LAYER_HEADER_SIZE] = {(char)0xAA,(char)0xBB,(char)0xCC,(char)0xDD,3};
			fwrite(GlbLayerHead, sizeof(char), sizeof(GlbLayerHead), fileRp);
			memset(&convhead, 0, sizeof(convhead));
			convhead.DataSize = glb_length;
			memcpy(HEAD, &convhead, sizeof(convhead));
			int iConvheadLength = sizeof(convhead)/4;
			for (int i = 0; i < iConvheadLength; i++)
			{
				for(int k=0; k< 4; k++)
				{
					fwrite(&HEAD[i*4+3-k],sizeof(char),1,fileRp);
				}
			}    
			//glb regsiters
			memcpy(cPrinBuf, &glb_ctrl, sizeof(cPrinBuf));
			for(int k=0; k< 4; k++)
			{
				fwrite(&cPrinBuf[3-k], sizeof(char), 1, fileRp);
			}
			char INSTR_DDR[4] = {0x4a, 0x00, 0x00, 0x00};
			fwrite(INSTR_DDR,sizeof(char),sizeof(INSTR_DDR),fileRp);
			memcpy(cPrinBuf,&GLB_NUM,sizeof(cPrinBuf));
			for(int k=0; k< 4; k++)
			{
				fwrite(&cPrinBuf[3-k],sizeof(char),1,fileRp);
			}

			//instruction header
			char RegLayerHead[LAYER_HEADER_SIZE] = {(char)0xAA, (char)0xBB, (char)0xCC, (char)0xDD,4};
			fwrite(RegLayerHead, sizeof(char), sizeof(RegLayerHead), fileRp);
			memset(&convhead, 0, sizeof(convhead));
			convhead.DataAddr = 0x4a000000;
			convhead.DataSize = 256 * GLB_NUM;
			memcpy(HEAD,&convhead,sizeof(convhead));
			iConvheadLength = sizeof(convhead)/4;
			for (int i = 0; i < iConvheadLength; i++)
			{
				for(int k=0; k< 4; k++)
				{
					fwrite(&HEAD[i*4+3-k],sizeof(char),1,fileRp);
				}
			}    
		}
		
		/* write instruction regsiters to real bin files */
		memcpy(cConvBufs, &convregisters, sizeof(convregisters));
		int iConvRegistersLength = sizeof(convregisters)/4;
		for (int i = 0; i < iConvRegistersLength; i++)
		{
			for(int k=0; k< 4; k++)
			{
				fwrite(&cConvBufs[i*4+3-k], sizeof(char), 1, fileRp);
			}
		}
		return 0;
	}

	/*************************************************************************
	* Function Name : convRegKernelSize
	* Description   : get the value of the convolution kernel size register value
	* Parameters    : NULL
	* Returns       : uiRegValue -- the value of the register
	**************************************************************************/
	unsigned int Convolution::convRegKernelSize(void)
	{
		unsigned int uiRegValue = 0;

		uiRegValue = kernel_w;
		uiRegValue = uiRegValue << REG_KER_H_BIT;
		uiRegValue += kernel_w;

		return uiRegValue;
	}

	/*************************************************************************
	* Function Name : ConvRegFeatureSize
	* Description   : get the value of the convolution feature size register value
	* Parameters    : NULL
	* Returns       : uiRegValue -- the value of the register
	**************************************************************************/
	unsigned int Convolution::convRegFeatureSize(void)
	{
		unsigned int uiRegValue = 0;

		uiRegValue = LayerCommon.iInFeaH;
		uiRegValue = uiRegValue << REG_FEA_H_BIT;
		uiRegValue += LayerCommon.iInFeaW;

		return uiRegValue;
	}

	/*************************************************************************
	* Function Name : ConvRegFeatureChannel
	* Description   : get the value of the convolution input output channel register value ,
	*                 based on input and output channels.
	* Parameters    : uiInputChannel -- input channel number
	*                 uiOutputChannel -- output channel number
	* Returns       : uiRegValue -- the value of the register
	**************************************************************************/
	unsigned int Convolution::convRegFeatureChannel(void)
	{
		unsigned int uiRegValue = 0;

		uiRegValue = LayerCommon.cOutputChannel;
		uiRegValue = uiRegValue << REG_FEA_CO_BIT;
		if(cFirstFlag)
		{
			uiRegValue += DATA_CHANNEL_NUM;
		}
		else
		{
			uiRegValue += LayerCommon.cInputChannel;
		}
		return uiRegValue;
	}

	/*************************************************************************
	* Function Name : ConvRegPadCtrl
	* Description   : get the value of the convolution padding control register value
	* Parameters    : NULL
	* Returns       : uiRegValue -- the value of the register
	**************************************************************************/
	unsigned int Convolution::convRegPadCtrl(void)
	{
		unsigned int uiRegValue = 0;

		if(pad_w > 0)
		{
			uiRegValue = pad_w;
			uiRegValue = uiRegValue << REG_PAD_SZ_BIT;
			//enable
			uiRegValue += REG_PAD_ENABLE;
		}

		return uiRegValue;
	}

	/*************************************************************************
	* Function Name : ConvRegIFM_SPL_TIM
	* Description   : get the value of the convolution IFM_SPL_TIM register value
	* Parameters    : ucKerW -- feature width
	*                 uiFeaH -- feature height
	* Returns       : uiRegValue -- the value of the register
	**************************************************************************/
	unsigned int Convolution::convRegIFM_SPL_TIM(void)
	{
		unsigned int uiRegValue = 0;
		unsigned int uiFeatureSize = LayerCommon.iInFeaW*LayerCommon.iInFeaH;

		if(uiFeatureSize > CONV_BUFF_SIZE)
		{
			uiRegValue = (uiFeatureSize+(CONV_BUFF_SIZE-1))/CONV_BUFF_SIZE - 1;
			uiIfmSplTim = uiRegValue;
		}

		return uiRegValue;
	}

	/*************************************************************************
	* Function Name : ConvRegROW_PER_LD
	* Description   : get the value of the convolution ROW_PER_LD register value
	* Parameters    : NULL
	* Returns       : uiRegValue -- the value of the register
	**************************************************************************/
	unsigned int Convolution::convRegROW_PER_LD(void)
	{
		unsigned int uiRegValue = 0;
		unsigned int uiFeatureSize = LayerCommon.iInFeaW*LayerCommon.iInFeaH;

		if(uiFeatureSize > CONV_BUFF_SIZE)
		{
			uiRegValue = (LayerCommon.iInFeaH/(uiIfmSplTim + 1));
			uiRowPerLd = uiRegValue;
		}

		return uiRegValue;
	}

	/*************************************************************************
	* Function Name : ConvRegROW_PER_LD
	* Description   : get the value of the convolution ROW_PER_LD register value
	* Parameters    : ucKerW -- feature width
	*                 uiFeaH -- feature height
	*                 uiRowPerLd -- the value of the register ROW_LST_LD
	* Returns       : uiRegValue -- the value of the register
	**************************************************************************/
	unsigned int Convolution::convRegROW_LST_LD(void)
	{
		unsigned int uiRegValue = 0;
		unsigned int uiFeatureSize = LayerCommon.iInFeaW*LayerCommon.iInFeaH;

		if(uiFeatureSize > CONV_BUFF_SIZE)
		{
			uiRegValue = LayerCommon.iInFeaH % uiRowPerLd;
			if(!uiRegValue)
			{
				uiRegValue = uiRowPerLd;
			}
		}
		return uiRegValue;
	}
}
