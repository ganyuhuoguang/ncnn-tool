/*
 * Input.cpp
 *
 *  Created on: Jun 12, 2019
 *      Author: doyle
 */

#include "Input.h"
#include <iostream>
#include <limits.h>

namespace tmnet
{
	/*************************************************************************
	* Function Name : Input
	* Description   : construct function ,initialize all the variable
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	Input::Input()
	{
		cRunFlag = 0;
		cConcatFlag = 0;
	}

	/*************************************************************************
	* Function Name : ~Input
	* Description   : deconstruct function
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	Input::~Input()
	{

	}

	/*************************************************************************
	* Function Name : loadParam
	* Description   : layer load param data
	* Parameters    : fileFp -- input param file
	* Returns       : 0 -- success
	**************************************************************************/
	int Input::loadParam(FILE* fileFp, int output_num)
	{
	    int id = 0;
	    int value = 0;
	    feature_w = output_num;
		while (fscanf(fileFp, "%d=%d", &id,&value) == 2)
		{
			switch (id)
			{
			case 0:
				feature_w = value;
				break;
			case 1:
				feature_h = value;
				break;
			case 2:
				feature_d = value;
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
	int Input::calculateFeaSize(std::vector<unsigned int> iIw, std::vector<unsigned int> iIh, std::vector<unsigned int> iIc)
	{
		LayerCommon.iInFeaW = 0;
		LayerCommon.iInFeaH = 0;
		LayerCommon.cInputChannel = 0;

		LayerCommon.iOutFeaW = feature_w;
		LayerCommon.iOutFeaH = feature_h;
		LayerCommon.cOutputChannel = feature_d;

		//input:null
		LayerCommon.uiInputSize = 0;
		//output:int8
		LayerCommon.uiOutputSize = LayerCommon.iOutFeaW*\
												LayerCommon.iOutFeaH * LayerCommon.cOutputChannel;
		return 0;
	}

	/*************************************************************************
	* Function Name : fillDDRAddress
	* Description   : fill the bin data ddr address
	* Parameters    : uiLastAddr -- last address
	* Returns       : 0 -- success
	**************************************************************************/
	unsigned long long Input::fillDDRAddress(unsigned long long uiLastAddr,const char* num)
	{
		return uiLastAddr;
	}

	/*************************************************************************
	* Function Name : getInputScale
	* Description   : get layer input scale value from bin file
	* Parameters    : fileFp -- input file
	* Returns       : 0 -- success
	**************************************************************************/
	int Input::getInputScale(FILE *fileFp)
	{
		return 0;
	}

	/*************************************************************************
	* Function Name : setRegisterValue
	* Description   : set layer register value
	* Parameters    : NULL
	* Returns       : uiLastAddr -- next address
	**************************************************************************/
	unsigned long long Input::setRegisterValue(unsigned long long uiLastAddr,unsigned int uiOneSeg,\
												const unsigned int uiOriAddr,char cBufferNum)
	{
		return uiLastAddr;
	}

	int Input::setRegisterValue(std::vector<unsigned long long> uiInputAddr, std::vector<unsigned long long> uiOutputAddr,const char* num)
	{
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
	int Input::writeBinFile(FILE *fileInFp,FILE *fileOutFp,int iLayerIndex,const char* num)
	{
		return 0;
	}

	/*************************************************************************
	* Function Name : writeddrBinFile
	* Description   : write register value to bin file
	* Parameters    : fileRp -- out tmmodel bin file after the layer date		   
	* Returns       : 0 -- success
	**************************************************************************/
	int Input::writeddrBinFile(FILE *fileRp)
	{
	   	return 0;
	}

	/*************************************************************************
	* Function Name : writeDDRInfoInputOutput
	* Description   : write input output data to out file
	* Parameters    : fileOutFp -- output file
	* Returns       : NULL
	**************************************************************************/
	void Input::writeDDRInfoInputOutput(FILE *fileOutFp)
	{
		
	}
}




