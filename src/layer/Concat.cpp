/*
 * ReLU.cpp
 *
 *  Created on: Jun 12, 2019
 *      Author: doyle
 */

#include "Concat.h"

namespace tmnet 
{
	/*************************************************************************
	* Function Name : ReLU
	* Description   : construct function ,initialize all the variable
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	Concat::Concat()
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
		cConcatFlag = 0;
		reluWriteBinFlag = false;
	}

	/*************************************************************************
	* Function Name : ~ReLU
	* Description   : deconstruct function
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	Concat::~Concat()
	{

	}

	/*************************************************************************
	* Function Name : loadParam
	* Description   : layer load param data
	* Parameters    : fileFp -- input param file
	* Returns       : 0 -- success
	**************************************************************************/
	int Concat::loadParam(FILE* fileFp, int output_num)
	{
		int id = 0;
		int value = 0;

		zero = output_num;
		while (fscanf(fileFp, "%d=%d", &id,&value) == 2)
		{
			switch (id)
			{
				case 0:
					zero = value;
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
	int Concat::calculateFeaSize(std::vector<unsigned int> iIw, std::vector<unsigned int> iIh, std::vector<unsigned int> iIc)
	{
		int dim = 3;
		int sum = 0;
		int InputNum = iIw.size();
		LayerCommon.iInFeaW = iIw[0];
		LayerCommon.iInFeaH = iIh[0];
		LayerCommon.cInputChannel = iIc[0];

		if (iIc[0] == 0)
		{
			dim = 2;
			if (iIh[0] == 0)
			{
				dim = 1;
			}
		}

		switch (dim)
		{
			case 1:
				for (int i = 0; i < InputNum; i++)
				{
					sum = sum + iIw[i];
				}
				LayerCommon.iOutFeaW = sum;
				LayerCommon.iOutFeaH = iIh[0];
				LayerCommon.cOutputChannel = iIc[0];
				break;
			case 2:
				for (int i = 0; i < InputNum; i++)
				{
					sum = sum + iIh[i];
				}
				LayerCommon.iOutFeaW = iIw[0];
				LayerCommon.iOutFeaH = sum;
				LayerCommon.cOutputChannel = iIc[0];
				break;
			case 3:
				for (int i = 0; i < InputNum; i++)
				{
					sum = sum + iIc[i];
				}
				LayerCommon.iOutFeaW = iIw[0];
				LayerCommon.iOutFeaH = iIh[0];
				LayerCommon.cOutputChannel = sum;
				break;
			default:
				break;
		}
		
		//input:int8
		LayerCommon.uiInputSize = iIw[0]*iIh[0]*8;

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
	unsigned long long Concat::fillDDRAddress(unsigned long long uiLastAddr,const char* num)
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
	int Concat::getInputScale(FILE *fileFp)
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
	void Concat::setQuantize(int iQuantize,float fIScale)
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
	void Concat::setFp32(int iFp32,unsigned long long uiBiasAddr,unsigned long long uiWScaleAddr,float fIScale)
	{

	}

	/*************************************************************************
	* Function Name : getNextInputOutputAddr
	* Description   : ge tNext Input Output Address
	* Parameters    : uiAddr -- input address
	* Returns       : uiNewAddr -- next layer input address
	**************************************************************************/
	unsigned long long Concat::getNextInputOutputAddr(unsigned long long uiAddr,unsigned int uiOneSeg,\
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
	unsigned long long Concat::setRegisterValue(unsigned long long uiLastAddr,unsigned int uiOneSeg,\
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
	int Concat::setRegisterValue(std::vector<unsigned long long> uiInputAddr, std::vector<unsigned long long> uiOutputAddr,const char* num)
	{
		return 0;
	}

	/*************************************************************************
	* Function Name : writeDDRInfoInputOutput
	* Description   : write input output ddr information to putput file
	* Parameters    : fileOutFp -- output file
	* Returns       : void
	**************************************************************************/
	void Concat::writeDDRInfoInputOutput(FILE *fileOutFp)
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
	int Concat::writeBinFile(FILE *fileInFp,FILE *fileOutFp,int iLayerIndex,const char* num)
	{
		return 0;
	}

	/*************************************************************************
	* Function Name : writeddrBinFile
	* Description   : write register value to bin file
	* Parameters    : fileRp -- out tmmodel bin file after the layer date		   
	* Returns       : 0 -- success
	**************************************************************************/
	int Concat::writeddrBinFile(FILE *fileRp)
	{
		return 0;
	}
}
