/*
 * Convolution.cpp
 *
 *  Created on: Jun 11, 2019
 *      Author: doyle
 */

#include "ConvolutionDepthWise.h"
#include <math.h>

namespace tmnet
{
	/*************************************************************************
	* Function Name : Convolution
	* Description   : construct function ,initialize all the variable
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	ConvolutionDepthWise::ConvolutionDepthWise()
	{
		uiIfmSplTim = 0;
		inputsWeightsDatasIsInt8 = 0;
		cRunFlag = 1;
		cConcatFlag = 0;
	}

	/*************************************************************************
	* Function Name : ~Convolution
	* Description   : deconstruct function
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	ConvolutionDepthWise::~ConvolutionDepthWise()
	{

	}

	/*************************************************************************
	* Function Name : setLayerBinDataSize
	* Description   : set Layer Bin Data Size
	* Parameters    : NULL
	* Returns       : NULL
	**************************************************************************/
	void  ConvolutionDepthWise::setLayerBinDataSize(void)
	{
		uiBinDataSize = data_size + 2* num_output*sizeof(float);
	}

	/*************************************************************************
	* Function Name : loadParam
	* Description   : layer load param data
	* Parameters    : fileFp -- input param file
	* Returns       : 0 -- success
	**************************************************************************/
	int ConvolutionDepthWise::loadParam(FILE *fileFp, int output_num)
	{
		int id = 0;
		int value = 0;

		num_output = output_num;
		while (fscanf(fileFp, "%d=%d", &id, &value) == 2)
		{
			switch (id)
			{
				case 0:
					num_output = value;
					break;
				case 1:
					kernel_w = value;
					break;
				case 2:
					dilation_size = value;
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
				case 7:
					group = value;
					break;
				case 8:
					inputsWeightsDatasIsInt8 = value;
					break;
				default:
					break;
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
	int ConvolutionDepthWise::calculateFeaSize(std::vector<unsigned int> iIw, std::vector<unsigned int> iIh, std::vector<unsigned int> iIc)
	{
		LayerCommon.iInFeaW = iIw[0];
		LayerCommon.iInFeaH = iIh[0];
		LayerCommon.cInputChannel = iIc[0];

		LayerCommon.iOutFeaW = (LayerCommon.iInFeaW - kernel_w + 2*ConvDwParam::pad_w)/ConvDwParam::stride_w + 1;
		LayerCommon.iOutFeaH = LayerCommon.iOutFeaW;
		LayerCommon.cOutputChannel = num_output;

		//input:int8
		if(cFirstFlag)
		{
			LayerCommon.uiInputSize = iIw[0]*iIh[0]*8;
		}
		else
		{
			LayerCommon.uiInputSize = iIw[0]*iIh[0]*iIc[0];
		}
		//output:int32
		LayerCommon.uiOutputSize = LayerCommon.iOutFeaW*\
											LayerCommon.iOutFeaH*\
											LayerCommon.cOutputChannel*sizeof(unsigned int);
		return 0;
	}

	/*************************************************************************
	* Function Name : getNewAddrFormDataSize
	* Description   : calculate the next bin data address (4K Algin)
	* Parameters    : NULL
	* Returns       : uiNewAddress -- nextt address
	**************************************************************************/
	uint64 ConvolutionDepthWise::getNewAddrFormDataSize(unsigned int uiDataSize, uint64 uiAddress)
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
	uint64 ConvolutionDepthWise::fillDDRAddress(uint64 uiLastAddr,const char* num)
	{
		unsigned int uiDataSize;
		//write weight address and calculate the size of weight data    type:int8
		uiWeightAddr = uiLastAddr;
		uiDataSize = data_size;
		uiLastAddr = getNewAddrFormDataSize(uiDataSize,uiLastAddr);
		uiBiasAddr = uiLastAddr;
		if(cPreluFlag == 1)
		{
			uiDataSize = num_output*4*3;
		}
		else
		{
			uiDataSize = num_output*4*2;
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
	int ConvolutionDepthWise::getInputScale(FILE *fileFp)
	{
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
		int rc = 0;
		char *pcCharBuf = (char *)malloc(data_size);
		char *pcConvBias = (char *)malloc(num_output*sizeof(float));
		//read the tag
		rc = fread(&flag_struct, sizeof(flag_struct), 1,fileFp);
		assert(rc > 0);
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
	uint64 ConvolutionDepthWise::getNextInputOutputAddr(uint64 uiAddr,unsigned int uiOneSeg,\
															unsigned int uiOriAddr,char cBufferNum)
	{	
		return 0;
	}

	/*************************************************************************
	* Function Name : setRegisterValue
	* Description   : set layer register value
	* Parameters    : NULL
	* Returns       : uiLastAddr -- next address
	**************************************************************************/
	uint64 ConvolutionDepthWise::setRegisterValue(uint64 uiLastAddr,unsigned int uiOneSeg,\
														const unsigned int uiOriAddr,char cBufferNum)
	{
		return 0;
	}

	/*************************************************************************
	* Function Name : setRegisterValue
	* Description   : set layer register value
	* Parameters    : uiInputAddr -- input data address
	*                 uiOutputAddr -- output data address
	* Returns       : uiLastAddr -- next address
	**************************************************************************/
	int ConvolutionDepthWise::setRegisterValue(std::vector<unsigned long long> uiInputAddr, std::vector<unsigned long long> uiOutputAddr,const char* num)
	{
		memset(&convdwregisters, 0, sizeof(convdwregisters));
		unsigned int channelNumber = atoi(num);
		row_in = LayerCommon.iInFeaW;
		col_in= LayerCommon.iInFeaH;
		row_out= LayerCommon.iOutFeaW;
		col_out= LayerCommon.iOutFeaH;
		channel_in= LayerCommon.cInputChannel;
		channel_out= LayerCommon.cOutputChannel;
		kernel_size= kernel_w;
		stride= stride_w;
		padding= pad_w;
		conv_bypass= 0;
		conv_mode= 1;
		conv_start = 1;
		int32_to_fp32_enable= 1;
		bn_enable= 1;
		din_select= 1;
		ew_enable= 0;
		sdp_bypass= 0;
		// col_seg_out=col_seg/stride;
		cdp_bypass = 1;
		softmax_bypass = 1;
		PoolingSize = poolingsize;
		PoolStride = poolingstride;
		PoolPadding = poolingpadding;	
		// printf("col_seg:%d",col_seg);
		// printf("stride:%d",stride);

		if (cReLUFlag == 1)
		{
			relu_enable = 1;
		}
		else
		{
			relu_enable = 0;
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
		
		if (cCloseQnFlag == 1)
		{
			fp32_to_int8_enable = 0;
			qn_enable = 0;
		}
		else
		{
			fp32_to_int8_enable = 1;
			qn_enable = 1;
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

		if (row_out == 1 && col_out == 1)
		{
			stride = kernel_size;
		}
		else
		{
			stride = stride_w;
		}
		
		iOverlapHeight = kernel_size - stride;
		if (conv_mode == 0)
	    {
	        if (kernel_size*kernel_size*channel_in > WRAMSIZE)
	        {
	            weight_seg_enable=1;
	        }
	        else
	        {
	            weight_seg_enable=0;
	        }
	    }
		else
	    {
	        weight_seg_enable=0;
	    }
        
		if ((row_in * col_in * channel_in / 8) < FRAMSIZE)
		{
			feature_seg_disable=1;
		}
		else
		{
			feature_seg_disable=0;
		}

		if (stride > 0)
		{
			col_seg_out=col_seg/stride;
		}
		
		if (iOverlapHeight < 0)
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
	    convdwregisters.head = 0x3aa55;
	    convdwregisters.Conv_Ctrl = (feature_seg_disable<<22)+(qn_enable<<17)+(weight_seg_enable<<16)+(conv_mode<<13)+(padding<<9)+\
							  (stride<<5)+(kernel_size<<1)+(conv_bypass<<15)+conv_start;
		convdwregisters.uiDWRegFeatureSrcAdd = uiInputAddr[0];
		convdwregisters.uiDWRegFeatureDstAdd = uiOutputAddr[0];
		convdwregisters.uiDWRegWeightSrcAdd = uiWeightAddr;
		// printf("CONVuiWeightAddr:********************0x%llx\n",convdwregisters.uiDWRegWeightSrcAdd);
		convdwregisters.uiRegBiasSrcAdd = uiBiasAddr;
		convdwregisters.uiRegFeaSizeWidth = (row_out<<16)+row_in;
		convdwregisters.uiRegFeaSizeSeg0 = (col_seg_in_l<<20)+(col_seg_in_m<<10)+col_seg_in_f;
	    convdwregisters.uiRegFeaSizeSeg1 = ((iOverlapHeight)*row_in<<18)+(convSegNum<<10)+ colSegOutF;
		convdwregisters.uiRegFeaChannel1 = (channelOutAlign<<16)+channel_in;
		convdwregisters.uiRegArithA = row_in*col_in;
	    convdwregisters.uiRegArithB = row_in*col_in*channel_in;
		convdwregisters.uiRegArithC = (col_seg_in_m*row_in<<16)+col_seg_in_f*row_in;
		convdwregisters.uiRegArithD = col_seg_in_l*row_in;
	    convdwregisters.uiRegArithE = row_out*col_out*channelOutAlign/(PoolStride*PoolStride);
		convdwregisters.uiRegArithF = ((kernel_size*kernel_size)<<23)+(kernel_size*kernel_size*channel_in);
		convdwregisters.uiRegArithG = (colSegOutL<<10) + colSegOutM;

		//sdp
		convdwregisters.Sdp_Ctrl = 0x01 + (int32_to_fp32_enable << 1) + \
								(bn_enable << 2) + \
								(relu_enable << 3) + \
								(prelu_enable << 4) + \
								(fp32_to_int8_enable << 5) + \
								(qn_enable << 6) + \
								(din_select << 7) + \
								(ew_enable << 8) + \
								(pooling_enable << 9) + \
								(PoolingSize << 10) + \
								(PoolStride << 14) + \
								(PoolPadding << 18) + \
								(sdp_bypass << 23);
						
		convdwregisters.Sdp_Row_Col_Chanel = ((row_out*col_out) << 13) + channelOutAlign;   // 12 for vgg16; 13 for facenet
		convdwregisters.Qn_A = Qn_a;
		convdwregisters.Sdp_Data_Num = row_out*col_out*channelOutAlign;
		convdwregisters.Sdp_Matrix_Row_Col_In = (row_out << 16) + col_out;
    	convdwregisters.Sdp_Matrix_Row_Col_Out =((row_out/PoolStride)<<16)+(col_out/PoolStride);
		convdwregisters.Cdp_Ctrl = cdp_bypass << 7;
		convdwregisters.Softmax_Ctrl = softmax_bypass << 1;
		convdwregisters.Classes_length = SOFTMAX_OUTPUT_LENGTH;
		return 0;
	}

	/*************************************************************************
	* Function Name : writeDDRInfoWeight
	* Description   : write weight data information to output file
	* Parameters    : fileOutFp -- output file
	* Returns       : void
	**************************************************************************/
	void ConvolutionDepthWise::writeDDRInfoWeight(FILE *fileOutFp)
	{

	}

	/*************************************************************************
	* Function Name : writeDDRInfoInputOutput
	* Description   : write input output data information to output file
	* Parameters    : fileOutFp -- output file
	* Returns       : void
	**************************************************************************/
	void ConvolutionDepthWise::writeDDRInfoInputOutput(FILE *fileOutFp)
	{

	}

	/*************************************************************************
	* Function Name : getFp32Infor
	* Description   : get fp32 information
	* Parameters    : uiBias -- output bias address
	* 				  uiWScale -- output weight scale address
	* 				  fIScale -- output scale
	* Returns       : 0 -- success
	**************************************************************************/
	int ConvolutionDepthWise::getFp32Infor(uint64 *uiBias,uint64 *uiWScale,float *fIScale)
	{
		return 0;
	}

	/*************************************************************************
	* Function Name : getQuantizeInfor
	* Description   : get quantize information
	* Parameters    : fIScale -- output scale
	* Returns       : 0 -- success
	**************************************************************************/
	int ConvolutionDepthWise::getQuantizeInfor(float *fIScale)
	{
		return 0;
	}

	/*************************************************************************
	* Function Name : writeBinFile
	* Description   : write layer data to bin file, should be lined to a -> bias -> prelu
	* Parameters    : iDropCount -- dropout and input layer number before this layer
	*			      fileInFp -- input ncnn bin file
	*			      fileOutFp -- out tmmodel bin file
	*			      iLayerIndex -- index in ncnn param file
	* Returns       : 0 -- success
	**************************************************************************/
	int ConvolutionDepthWise::writeBinFile(FILE *fileInFp,FILE *fileOutFp,int iLayerIndex,const char* num)
	{
		char HEAD[sizeof(convdwhead)];
		int channelnumber = atoi(num);
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
		int iKernelSize = kernel_w * kernel_w;
		char *pcCharBuf = (char *)malloc(data_size);
		char *pcConvBias = (char *)malloc(num_output*sizeof(float));
		char *pcConvWscale = (char *)malloc(num_output*sizeof(float));
		char *pcConvLscale = (char *)malloc(num_output*sizeof(float));
		char cPrintBuf[PRINT_BUF_SIZE];
		unsigned int k = 0, rc = 0;
		
		if(pcCharBuf == NULL || pcConvBias == NULL)
		{
			tmtool_log(LOG_ERROR, "malloc wrong: %d", iLayerIndex);
			return -1;
		}

		//read the tag
		rc = fread(&flag_struct, sizeof(flag_struct), 1,fileInFp);
		assert(rc > 0);
		//write weight
		rc = fread(pcCharBuf, data_size, 1, fileInFp);
		assert(rc > 0);

		int output_num = num_output;
		
		//add layerhead, size, address
		char cLayerHead[LAYER_HEADER_SIZE]={(char)0xAA, (char)0xBB, (char)0xCC, (char)0xDD, 01};
		convdwhead.InFeaW = LayerCommon.iInFeaW;
		convdwhead.InFeaH = LayerCommon.iInFeaH;
		convdwhead.InputChannel = LayerCommon.cInputChannel;
		convdwhead.OutFeaW = LayerCommon.iOutFeaW;
		convdwhead.OutFeaH = LayerCommon.iOutFeaH;
		convdwhead.OutputChannel = LayerCommon.cOutputChannel;
		convdwhead.FeaInAddr = convdwregisters.uiDWRegFeatureSrcAdd;
		convdwhead.FeaOutAddr = convdwregisters.uiDWRegFeatureDstAdd;
		convdwhead.LScale = Qn_a;
		convdwhead.PSPLFlag = WeightAddrFlag;
		convdwhead.DataSize = data_size;
		convdwhead.DataAddr = convdwregisters.uiDWRegWeightSrcAdd;
		fwrite(cLayerHead,sizeof(char),sizeof(cLayerHead),fileOutFp);
		memcpy(HEAD,&convdwhead,sizeof(convdwhead));
		int convdwheadLength = sizeof(convdwhead)/4;
		for (int i = 0; i < convdwheadLength; i++)
		{
			for(int k=0; k< 4; k++)
			{
				fwrite(&HEAD[i*4+3-k],sizeof(char),1,fileOutFp);
			}
		}

		/* write convolution weight data */
		for(int outid=0; outid<output_num/channelnumber; outid++)
		{
			for(int kid=0; kid<iKernelSize; kid++)
			{
				for(int k=0; k<channelnumber; k++)
				{
					if(k < 4)
					{
						fwrite(&pcCharBuf[outid * iKernelSize * channelnumber + \
										(3-k) * iKernelSize + kid], sizeof(char), 1, fileOutFp);
					}
					else if((k > 3) && (k < 8))
					{
						fwrite(&pcCharBuf[outid * iKernelSize * channelnumber + \
										(11-k) * iKernelSize + kid], sizeof(char), 1, fileOutFp);
					}
					else if((k > 7) && (k < 12))
					{
						fwrite(&pcCharBuf[outid * iKernelSize * channelnumber + \
										(19-k) * iKernelSize + kid],sizeof(char), 1, fileOutFp);
					}
					else if(( k > 11) && (k < 16))
					{
						fwrite(&pcCharBuf[outid*iKernelSize*channelnumber + (27-k)*iKernelSize+kid],sizeof(char), 1, fileOutFp);
					}
				}
			}
		}

		/* write bias */
		char BiasLayerHead[LAYER_HEADER_SIZE] = {(char)0xAA, (char)0xBB, (char)0xCC, (char)0xDD, 2};
		fwrite(BiasLayerHead, sizeof(char), sizeof(BiasLayerHead), fileOutFp);
		/* clear struct convdwhead for ready to next scale field */
		memset(&convdwhead, 0, sizeof(convdwhead));

		convdwhead.DataAddr = convdwregisters.uiRegBiasSrcAdd;
		if (cPreluFlag==1)
		{
			convdwhead.DataSize = num_output *3*4;
		}
		else
		{
			convdwhead.DataSize = num_output *2*4;
		}
		
		memcpy(HEAD, &convdwhead, sizeof(convdwhead));
		convdwheadLength = sizeof(convdwhead)/4;
		for (int i = 0; i < convdwheadLength; i++)
		{
			for(int k=0; k< 4; k++)
			{
				fwrite(&HEAD[i*4 + 3 - k],sizeof(char),1,fileOutFp);
			}
		}    

		//write data
		rc = fread(pcConvBias,num_output * sizeof(float), 1, fileInFp);
		assert(rc > 0);

		//write weight scale
		wscale = (float *)malloc(num_output*4);
		rc = fread(pcConvWscale,num_output *  sizeof(float), 1, fileInFp);
		assert(rc > 0);

		memcpy(wscale, pcConvWscale, num_output*4);
		for (int j=0; j<num_output; j++)
		{
			memcpy(cPrintBuf, &pcConvWscale[sizeof(float)*j], sizeof(cPrintBuf));
		}
		free(pcConvWscale);

		//write layer scale
		rc = fread(pcConvLscale,sizeof(float), 1, fileInFp);
		assert(rc > 0);

		memcpy(&iscale, pcConvLscale, 4);
		memcpy(cPrintBuf, pcConvLscale, sizeof(cPrintBuf));
		free(pcConvLscale);

		for (int j=0; j<num_output; j++)
		{
		//float a = 1/(wscale[j]*iscale);
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
			for(int k=0; k< 4 ;k++)
			{
				fwrite(&cPrintBuf[3-k], sizeof(char), 1, fileOutFp);
			}
		}

		//write bias
		for (int j=0; j<num_output; j++)
		{
			memcpy(cPrintBuf,&pcConvBias[sizeof(float)*j],sizeof(cPrintBuf));
			for(k=0; k<sizeof(float); k++)
			{
				fwrite(&cPrintBuf[3-k], sizeof(char), 1, fileOutFp);
			}
		}

		free(pcCharBuf);
		free(pcConvBias);
		free(wscale);
		return rc;
	}

	/*************************************************************************
	* Function Name : writeddrBinFile
	* Description   : write register value to bin file
	* Parameters    : fileRp -- out tmmodel bin file after the layer date		   
	* Returns       : 0 -- success
	**************************************************************************/
	int ConvolutionDepthWise::writeddrBinFile(FILE *fileRp)
	{
		char cConvBufs[sizeof(convdwregisters)];
		memcpy(cConvBufs,&convdwregisters,sizeof(convdwregisters));
		int convdwregistersLength = sizeof(convdwregisters)/4;
		for (int i = 0; i < convdwregistersLength; i++)
		{
			for(int k = 0; k< 4; k++)
			{
				fwrite(&cConvBufs[i*4+3-k],sizeof(char),1,fileRp);
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
	unsigned int ConvolutionDepthWise::convRegKernelSize(void)
	{
		return 0;
	}

	/*************************************************************************
	* Function Name : ConvRegFeatureSize
	* Description   : get the value of the convolution feature size register value
	* Parameters    : NULL
	* Returns       : uiRegValue -- the value of the register
	**************************************************************************/
	unsigned int ConvolutionDepthWise::convRegFeatureSize(void)
	{
		return 0;
	}

	/*************************************************************************
	* Function Name : ConvRegFeatureChannel
	* Description   : get the value of the convolution input output channel register value ,
	*                 based on input and output channels.
	* Parameters    : uiInputChannel -- input channel number
	*                 uiOutputChannel -- output channel number
	* Returns       : uiRegValue -- the value of the register
	**************************************************************************/
	unsigned int ConvolutionDepthWise::convRegFeatureChannel(void)
	{
		return 0;
	}

	/*************************************************************************
	* Function Name : ConvRegPadCtrl
	* Description   : get the value of the convolution padding control register value
	* Parameters    : NULL
	* Returns       : uiRegValue -- the value of the register
	**************************************************************************/
	unsigned int ConvolutionDepthWise::convRegPadCtrl(void)
	{
		return 0;
	}

	/*************************************************************************
	* Function Name : ConvRegIFM_SPL_TIM
	* Description   : get the value of the convolution IFM_SPL_TIM register value
	* Parameters    : ucKerW -- feature width
	*                 uiFeaH -- feature height
	* Returns       : uiRegValue -- the value of the register
	**************************************************************************/
	unsigned int ConvolutionDepthWise::convRegIFM_SPL_TIM(void)
	{
		return 0;
	}

	/*************************************************************************
	* Function Name : ConvRegROW_PER_LD
	* Description   : get the value of the convolution ROW_PER_LD register value
	* Parameters    : NULL
	* Returns       : uiRegValue -- the value of the register
	**************************************************************************/
	unsigned int ConvolutionDepthWise::convRegROW_PER_LD(void)
	{
		return 0;
	}

	/*************************************************************************
	* Function Name : ConvRegROW_PER_LD
	* Description   : get the value of the convolution ROW_PER_LD register value
	* Parameters    : ucKerW -- feature width
	*                 uiFeaH -- feature height
	*                 uiRowPerLd -- the value of the register ROW_LST_LD
	* Returns       : uiRegValue -- the value of the register
	**************************************************************************/
	unsigned int ConvolutionDepthWise::convRegROW_LST_LD(void)
	{
		return 0;
	}
}
