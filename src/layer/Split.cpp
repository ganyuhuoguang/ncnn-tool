/*
 * ReLU.cpp
 *
 *  Created on: Jun 12, 2019
 *      Author: doyle
 */

#include "Split.h"

namespace tmnet 
{
	/*************************************************************************
	* Function Name : ReLU
	* Description   : construct function ,initialize all the variable
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	Split::Split()
	{
		//quantize value initialize
		iQuantizeFlag = 0;
		fBsQn = 0;
		//int32->fp32 value initialize
		iFp32Flag = 0;
		uiBsAluSrc = 0;
		uiBsMulSrc = 0;
		fOprand = 0;
		// cRunFlag = 0;
		cConcatFlag = 0;
		reluWriteBinFlag = false;
	}

	/*************************************************************************
	* Function Name : ~ReLU
	* Description   : deconstruct function
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	Split::~Split()
	{

	}

	/*************************************************************************
	* Function Name : loadParam
	* Description   : layer load param data
	* Parameters    : fileFp -- input param file
	* Returns       : 0 -- success
	**************************************************************************/
	int Split::loadParam(FILE* fileFp, int output_num)
	{
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
	int Split::calculateFeaSize(std::vector<unsigned int> iIw, std::vector<unsigned int> iIh, std::vector<unsigned int> iIc)
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
	unsigned long long Split::fillDDRAddress(unsigned long long uiLastAddr,const char* num)
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
	int Split::getInputScale(FILE *fileFp)
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
	void Split::setQuantize(int iQuantize,float fIScale)
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
	void Split::setFp32(int iFp32,unsigned long long uiBiasAddr,unsigned long long uiWScaleAddr,float fIScale)
	{

	}

	/*************************************************************************
	* Function Name : getNextInputOutputAddr
	* Description   : ge tNext Input Output Address
	* Parameters    : uiAddr -- input address
	* Returns       : uiNewAddr -- next layer input address
	**************************************************************************/
	unsigned long long Split::getNextInputOutputAddr(unsigned long long uiAddr,unsigned int uiOneSeg,\
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
	unsigned long long Split::setRegisterValue(unsigned long long uiLastAddr,unsigned int uiOneSeg,\
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
	int Split::setRegisterValue(std::vector<unsigned long long> uiInputAddr, std::vector<unsigned long long> uiOutputAddr,const char* num)
	{
		if (cRunFlag == 1)
		{
			memset(&splitregister, 0, sizeof(splitregister));
			unsigned int channelNumber = atoi(num);
			row_in = LayerCommon.iInFeaW;
			col_in = LayerCommon.iInFeaH;
			row_out = LayerCommon.iOutFeaW;
			col_out = LayerCommon.iOutFeaH;
			channel_in = LayerCommon.cInputChannel;
			channel_out = LayerCommon.cOutputChannel;
			conv_bypass = TRUE;
			sdp_bypass = TRUE;
			inputScale = row_out * col_out * channel_out;
			
			if(channel_in > 0)
			{
				if (channel_in <= channelNumber)
				{
					featureChannel = channelNumber;
					channelPadding = channelNumber - channel_in;
				}
				else
				{
					if (!(channel_in % channelNumber))
					{
						featureChannel = channel_in;
						channelPadding = 0;
					}
					else
					{
						featureChannel = ((channel_in / channelNumber)+1) * channelNumber;
						channelPadding = featureChannel - channel_in;
					}
				}
				
			}
			else
			{
				tmtool_log(LOG_ERROR, "channel_in error");
			}

			if ((row_in*col_in*channel_in/2) > CDPRAMSIZE)
			{
				if ((( row_in * col_in * channel_in / 2 ) % CDPRAMSIZE) != 0)
				{
					featureSegNum=((row_in*col_in*channel_in/2)/CDPRAMSIZE)+1;
				}
				else
				{
					featureSegNum=((row_in*col_in*channel_in/2)/CDPRAMSIZE);
				}
				if ((col_in % featureSegNum) != 0)
				{
					featureHeightSeg0 = (col_in / featureSegNum) + 1;
				}
				else
				{
					featureHeightSeg0 = (col_in / featureSegNum);
				}

				if ((row_in * featureHeightSeg0 * channel_in / 2) > CDPRAMSIZE)
				{
					featureSegNum = featureSegNum + 1;
					featureHeightSeg0 = (col_in / featureSegNum);
					featureHeightSeg1 = (col_in - featureHeightSeg0) / (featureSegNum - 1);
				}            
				else
				{
					featureSegNum = featureSegNum;
					featureHeightSeg0 = (col_in / featureSegNum);
					featureHeightSeg1 = (col_in - featureHeightSeg0) / (featureSegNum - 1);
				}
			}
			else
			{
				featureSegNum=1;
				featureHeightSeg0=col_in;
				featureHeightSeg1=col_in;
			}
			
			cdpMode = SPLIT;
			cdpBypass = FALSE;

			//conv
			splitregister.Conv_Ctrl = conv_bypass<<15;
			
			//sdp
			splitregister.Sdp_Ctrl = (sdp_bypass<<23);
			
			//cdp
			splitregister.Cdp_Ctrl = 0x01 + (cdpMode << 1) + (cdpBypass << 7);
			splitregister.FeaSrcAddr = uiInputAddr[0];
			splitregister.FeaDstAddr = uiOutputAddr[1];
			splitregister.FeaSegSize = (featureSegNum<<24)+(featureHeightSeg1<<16)+(featureHeightSeg0<<8)+row_in;
			splitregister.FeaChannel = (channelPadding << 16) + featureChannel;
			splitregister.Cdp_Arith_A = row_in * col_in;
			splitregister.Cdp_Arith_B = row_in * col_in * channel_in;
			splitregister.Cdp_Arith_C = ((featureHeightSeg1*row_in)<<16)+(featureHeightSeg0*row_in);
			splitregister.FeaDstAddr2 = uiOutputAddr[0];
			splitregister.Softmax_Ctrl = 2;
			splitregister.head = 0x4aa55;
		}

		return 0;
	}

	/*************************************************************************
	* Function Name : writeDDRInfoInputOutput
	* Description   : write input output ddr information to putput file
	* Parameters    : fileOutFp -- output file
	* Returns       : void
	**************************************************************************/
	void Split::writeDDRInfoInputOutput(FILE *fileOutFp)
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
	// int ReLU::writeBinFile(int iDropCount,FILE *fileInFp,FILE *fileOutFp,int iLayerIndex,char cBit)
	int Split::writeBinFile(FILE *fileInFp,FILE *fileOutFp,int iLayerIndex,const char* num)
	{
		return 0;
	}

	/*************************************************************************
	* Function Name : writeddrBinFile
	* Description   : write register value to bin file
	* Parameters    : fileRp -- out tmmodel bin file after the layer date		   
	* Returns       : 0 -- success
	**************************************************************************/
	int Split::writeddrBinFile(FILE *fileRp)
	{
		if (cRunFlag == 1)
		{
			char cConvBufs[sizeof(splitregister)];
			memcpy(cConvBufs,&splitregister,sizeof(splitregister));
			int isplitregisterLength = sizeof(splitregister)/4;
			for (int i = 0; i < isplitregisterLength; i++)
			{
				for(int k=0; k< 4; k++)
				{
					fwrite(&cConvBufs[i*4+3-k],sizeof(char),1,fileRp);
				}

			}

		}
		return 0;
	}
}

