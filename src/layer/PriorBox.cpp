/*
 * ReLU.cpp
 *
 *  Created on: Jun 12, 2019
 *      Author: doyle
 */

#include "PriorBox.h"
#include <string.h>
#include <iostream>

namespace tmnet
{
	/*************************************************************************
	* Function Name : ReLU
	* Description   : construct function ,initialize all the variable
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	PriorBox::PriorBox()
	{
		//quantize value initialize
		iQuantizeFlag = 0;
		fBsQn = 0;
		//int32->fp32 value initialize
		iFp32Flag = 0;
		uiBsAluSrc = 0;
		uiBsMulSrc = 0;
		fOprand = 0;
		cRunFlag = 0;
		cCpuFlag = 1;
		cConcatFlag = 0;
	}

	/*************************************************************************
	* Function Name : ~ReLU
	* Description   : deconstruct function
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	PriorBox::~PriorBox()
	{

	}

	int PriorBox::getSpecicalParam(std::vector<float> size, char *ptr, char *p)
	{
		size.clear();
		while ( ptr = strsep(&p,","))//get ,
		{
			if ((*ptr >= '0') && (*ptr <= '9'))
			{
				size.resize(atoi(ptr));
			}	
			else
			{
				size.push_back(atof(ptr));
			}
		}
		return 0;
	}

	/*************************************************************************
	* Function Name : loadParam
	* Description   : layer load param data
	* Parameters    : fileFp -- input param file
	* Returns       : 0 -- success
	**************************************************************************/
	int PriorBox::loadParam(FILE* fileFp, int output_num)
	{
		int id = 0;
		char cValue[50];
		int value = 0;
		zero = output_num;
		char *tokenPtr = NULL;
		char *p = NULL;

		while ((fscanf(fileFp, "%d=%s", &id,&cValue) == 2))
		{
			p = cValue;
			switch (id)
			{
			case -23300:
				getSpecicalParam(min_sizes,tokenPtr,p);
				break;
			case -23301:
				getSpecicalParam(max_sizes,tokenPtr,p);
				break;
			case -23302:
				getSpecicalParam(aspect_ratios,tokenPtr,p);
				break;
			case 3:
				variances[0] = atof(cValue);
				break;
			case 4:
				variances[1] = atof(cValue);
				break;
			case 5:
				variances[2] = atof(cValue);
				break;
			case 6:
				variances[3] = atof(cValue);
				break;
			case 7:
				flip = atoi(cValue);
				break;
			case 8:
				clip = atoi(cValue);
				break;
			case 9:
				image_width = atoi(cValue);
				break;
			case 10:
				image_height = atoi(cValue);
				break;
			case 11:
				step_width = atof(cValue);
				break;
			case 12:
				step_height = atof(cValue);
				break; 
			case 13:
				offset = atof(cValue);
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
	int PriorBox::calculateFeaSize(std::vector<unsigned int> iIw, std::vector<unsigned int> iIh, std::vector<unsigned int> iIc)
	{
		LayerCommon.iInFeaW = iIw[0];
		LayerCommon.iInFeaH = iIh[0];
		LayerCommon.cInputChannel = iIc[0];

		LayerCommon.iOutFeaW = one;
		LayerCommon.iOutFeaH = one;
		//LayerCommon.cOutputChannel = iIh*iIw*iIc/(one*one);
		LayerCommon.cOutputChannel = 0;

		//input:int8
		LayerCommon.uiInputSize = iIw[0]*iIh[0]*iIc[0];

		//output:int32
		LayerCommon.uiOutputSize = LayerCommon.iOutFeaW * \
											LayerCommon.iOutFeaH * \
											LayerCommon.cOutputChannel * sizeof(unsigned int);
		return 0;
	}

	/*************************************************************************
	* Function Name : fillDDRAddress
	* Description   : fill the bin data ddr address
	* Parameters    : uiLastAddr -- last address
	* Returns       : 0 -- success
	**************************************************************************/
	unsigned long long PriorBox::fillDDRAddress(unsigned long long uiLastAddr,const char* num)
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
	int PriorBox::getInputScale(FILE *fileFp)
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
	void PriorBox::setQuantize(int iQuantize,float fIScale)
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
	void PriorBox::setFp32(int iFp32,unsigned long long uiBiasAddr,unsigned long long uiWScaleAddr,float fIScale)
	{

	}

	/*************************************************************************
	* Function Name : getNextInputOutputAddr
	* Description   : ge tNext Input Output Address
	* Parameters    : uiAddr -- input address
	* Returns       : uiNewAddr -- next layer input address
	**************************************************************************/
	unsigned long long PriorBox::getNextInputOutputAddr(unsigned long long uiAddr,unsigned int uiOneSeg,\
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
	unsigned long long PriorBox::setRegisterValue(unsigned long long uiLastAddr,unsigned int uiOneSeg,\
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
	int PriorBox::setRegisterValue(std::vector<unsigned long long> uiInputAddr, std::vector<unsigned long long> uiOutputAddr,const char* num)
	{
		return 0;
	}

	/*************************************************************************
	* Function Name : writeDDRInfoInputOutput
	* Description   : write input output ddr information to putput file
	* Parameters    : fileOutFp -- output file
	* Returns       : void
	**************************************************************************/
	void PriorBox::writeDDRInfoInputOutput(FILE *fileOutFp)
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
	int PriorBox::writeBinFile(FILE *fileInFp,FILE *fileOutFp,int iLayerIndex,const char* num)
	{
		return 0;
	}

	/*************************************************************************
	* Function Name : writeddrBinFile
	* Description   : write register value to bin file
	* Parameters    : fileRp -- out tmmodel bin file after the layer date		   
	* Returns       : 0 -- success
	**************************************************************************/
	int PriorBox::writeddrBinFile(FILE *fileRp)
	{
		return 0;
	}
}

