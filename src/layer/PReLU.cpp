/*
 * ReLU.cpp
 *
 *  Created on: Jun 12, 2019
 *      Author: doyle
 */

#include "PReLU.h"

namespace tmnet
{
	/*************************************************************************
	* Function Name : ReLU
	* Description   : construct function ,initialize all the variable
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	PReLU::PReLU()
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
	}

	/*************************************************************************
	* Function Name : ~ReLU
	* Description   : deconstruct function
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	PReLU::~PReLU()
	{

	}

	/*************************************************************************
	* Function Name : setLayerBinDataSize
	* Description   : set Layer Bin Data Size
	* Parameters    : NULL
	* Returns       : NULL
	**************************************************************************/
	void PReLU::setLayerBinDataSize(void)
	{
		uiBinDataSize =  num_output*sizeof(float);
	}

	/*************************************************************************
	* Function Name : loadParam
	* Description   : layer load param data
	* Parameters    : fileFp -- input param file
	* Returns       : 0 -- success
	**************************************************************************/
	int PReLU::loadParam(FILE* fileFp, int output_num)
	{
		int id = 0;
		int value = 0;
		num_output = output_num;
		while (fscanf(fileFp, "%d=%d", &id,&value) == 2)
		{
			switch (id)
			{
				case 0:
					num_output = value;
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
	int PReLU::calculateFeaSize(std::vector<unsigned int> iIw, std::vector<unsigned int> iIh, std::vector<unsigned int> iIc)
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
		LayerCommon.uiOutputSize = LayerCommon.iOutFeaW * \
								LayerCommon.iOutFeaH * \
								LayerCommon.cOutputChannel*sizeof(unsigned int);
		return 0;
	}

	/*************************************************************************
	* Function Name : getNewAddrFormDataSize
	* Description   : calculate the next bin data address (4K Algin)
	* Parameters    : NULL
	* Returns       : uiNewAddress -- nextt address
	**************************************************************************/
	uint64 PReLU::getNewAddrFormDataSize(unsigned int uiDataSize,uint64 uiAddress)
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
	uint64 PReLU::fillDDRAddress(uint64 uiLastAddr,const char* num)
	{
		uiLastAddr = uiLastAddr;
		return uiLastAddr;
	}

	/*************************************************************************
	* Function Name : getInputScale
	* Description   : get layer input scale value from bin file
	* Parameters    : fileFp -- input file
	* Returns       : 0 -- success
	**************************************************************************/
	int PReLU::getInputScale(FILE *fileFp)
	{
		int rc = 0;
		char *pcCharBuf = (char *)malloc(num_output*sizeof(float));
		//read prelu
		rc = fread(pcCharBuf, num_output*sizeof(float), 1, fileFp);
		free(pcCharBuf);
		return rc;
	}

	/*************************************************************************
	* Function Name : setQuantize
	* Description   : set quantize value
	* Parameters    : iQuantize -- quantize enable flag
	*                 fScale -- quantize scale
	* Returns       : uiLastAddr -- next address
	**************************************************************************/
	void PReLU::setQuantize(int iQuantize,float fIScale)
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
	void PReLU::setFp32(int iFp32,uint64 uiBiasAddr,uint64 uiWScaleAddr,float fIScale)
	{

	}

	/*************************************************************************
	* Function Name : getNextInputOutputAddr
	* Description   : ge tNext Input Output Address
	* Parameters    : uiAddr -- input address
	* Returns       : uiNewAddr -- next layer input address
	**************************************************************************/
	uint64 PReLU::getNextInputOutputAddr(uint64 uiAddr,unsigned int uiOneSeg,\
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
	uint64 PReLU::setRegisterValue(uint64 uiLastAddr,unsigned int uiOneSeg,\
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
	int PReLU::setRegisterValue(std::vector<unsigned long long> uiInputAddr, std::vector<unsigned long long> uiOutputAddr,const char* num)
	{
		return 0;
	}

	/*************************************************************************
	* Function Name : writeDDRInfoInputOutput
	* Description   : write input output ddr information to putput file
	* Parameters    : fileOutFp -- output file
	* Returns       : void
	**************************************************************************/
	void PReLU::writeDDRInfoInputOutput(FILE *fileOutFp)
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
	int PReLU::writeBinFile(FILE *fileInFp,FILE *fileOutFp,int iLayerIndex,const char* num)
	{
		char *pcCharBuf = (char *)malloc(num_output*sizeof(float));
		char cPrintBuf[PRINT_BUF_SIZE];
		int rc = 0;

		rc = fread(pcCharBuf, num_output * sizeof(float), 1, fileInFp);
		for (int j=0; j<num_output; j++)
		{
			memcpy(cPrintBuf, &pcCharBuf[sizeof(float)*j], sizeof(cPrintBuf));
			for (int k=0; k < (int)(sizeof(float)); k++)
			{
				fwrite(&cPrintBuf[3-k],sizeof(char),1,fileOutFp);
			}
		}
		free(pcCharBuf);
		return rc;
	}

	/*************************************************************************
	* Function Name : writeddrBinFile
	* Description   : write register value to bin file
	* Parameters    : fileRp -- out tmmodel bin file after the layer date		   
	* Returns       : 0 -- success
	**************************************************************************/
	int PReLU::writeddrBinFile(FILE *fileRp)
	{
		return 0;
	}
}

