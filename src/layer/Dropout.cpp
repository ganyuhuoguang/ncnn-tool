/*
 * Dropout.cpp
 *
 *  Created on: Jun 12, 2019
 *      Author: doyle
 */

#include "Dropout.h"

namespace tmnet
{
	/*************************************************************************
	* Function Name : Dropout
	* Description   : construct function ,initialize all the variable
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	Dropout::Dropout()
	{
		cRunFlag = 0;
		cConcatFlag = 0;
	}

	/*************************************************************************
	* Function Name : ~Dropout
	* Description   : deconstruct function
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	Dropout::~Dropout()
	{

	}

	/*************************************************************************
	* Function Name : loadParam
	* Description   : layer load param data
	* Parameters    : fileFp -- input param file
	* Returns       : 0 -- success
	**************************************************************************/
	int Dropout::loadParam(FILE* fileFp, int output_num)
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
	int Dropout::calculateFeaSize(std::vector<unsigned int> iIw, std::vector<unsigned int> iIh, std::vector<unsigned int> iIc)
	{
		LayerCommon.iInFeaW = iIw[0];
		LayerCommon.iInFeaH = iIh[0];
		LayerCommon.cInputChannel = iIc[0];

		LayerCommon.iOutFeaW = iIw[0];
		LayerCommon.iOutFeaH = iIh[0];
		LayerCommon.cOutputChannel = iIc[0];

		//input:fp32
		LayerCommon.uiInputSize = iIw[0]*iIh[0]*iIc[0]*sizeof(float);
		//output:fp32
		LayerCommon.uiOutputSize = iIw[0]*iIh[0]*iIc[0]*sizeof(float);
		return 0;
	}

	/*************************************************************************
	* Function Name : fillDDRAddress
	* Description   : fill the bin data ddr address
	* Parameters    : NULL
	* Returns       : 0 -- success
	**************************************************************************/
	unsigned long long Dropout::fillDDRAddress(unsigned long long uiLastAddr,const char* num)
	{
		return uiLastAddr;
	}

	/*************************************************************************
	* Function Name : getInputScale
	* Description   : get layer input scale value from bin file
	* Parameters    : fileFp -- input file
	* Returns       : 0 -- success
	**************************************************************************/
	int Dropout::getInputScale(FILE *fileFp)
	{
		return 0;
	}

	/*************************************************************************
	* Function Name : setRegisterValue
	* Description   : set layer register value
	* Parameters    : NULL
	* Returns       : 0 -- success
	**************************************************************************/
	unsigned long long Dropout::setRegisterValue(unsigned long long uiLastAddr,unsigned int uiOneSeg,\
													const unsigned int uiOriAddr,char cBufferNum)
	{
		return uiLastAddr;
	}

	/*************************************************************************
	* Function Name : setRegisterValue
	* Description   : set layer register value
	* Parameters    : uiInputAddr -- input data address
	* 				  uiOutputAddr -- output data address
	* Returns       : void
	**************************************************************************/
	int Dropout::setRegisterValue(std::vector<unsigned long long> uiInputAddr, std::vector<unsigned long long> uiOutputAddr,const char* num)
	{
		return 0;
	}

	/*************************************************************************
	* Function Name : writeRegToJson
	* Description   : write register to json struct
	* Parameters    : json -- json struct
	* Returns       : NULL
	**************************************************************************/
	void Dropout::writeRegToJson(cJSON *json)
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
	int Dropout::writeBinFile(int iDropCount,FILE *fileInFp,FILE *fileOutFp,int iLayerIndex,char cBit)
	{
		iDropCount++;
		return iDropCount;
	}

	/*************************************************************************
	* Function Name : setRegisterValue
	* Description   : set layer register value
	* Parameters    : uiInputAddr -- input data address
	* 				  uiOutputAddr -- output data address
	* Returns       : void
	**************************************************************************/
	void Dropout::writeDDRInfoInputOutput(FILE *fileOutFp)
	{

	}
}


