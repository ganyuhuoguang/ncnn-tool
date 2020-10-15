/*
 * ReLU.cpp
 *
 *  Created on: Jun 12, 2019
 *      Author: doyle
 */

#include "ReLU.h"

namespace tmnet
{
	/*************************************************************************
	* Function Name : ReLU
	* Description   : construct function ,initialize all the variable
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	ReLU::ReLU()
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
	ReLU::~ReLU()
	{

	}

	/*************************************************************************
	* Function Name : loadParam
	* Description   : layer load param data
	* Parameters    : fileFp -- input param file
	* Returns       : 0 -- success
	**************************************************************************/
	int ReLU::loadParam(FILE* fileFp, int output_num)
	{
		int id = 0;
		float fvalue = 0;
		slope = 0;
		reluWriteBinFlag = false; 
		while (fscanf(fileFp, "%d=%f", &id,&fvalue) == 2)
		{
			switch (id)
			{
			case 0:
				slope = fvalue;
				reluWriteBinFlag = true;
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
	int ReLU::calculateFeaSize(std::vector<unsigned int> iIw, std::vector<unsigned int> iIh, std::vector<unsigned int> iIc)
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
											LayerCommon.cOutputChannel * 2;
		return 0;
	}

	/*************************************************************************
	* Function Name : fillDDRAddress
	* Description   : fill the bin data ddr address
	* Parameters    : uiLastAddr -- last address
	* Returns       : 0 -- success
	**************************************************************************/
	unsigned long long ReLU::fillDDRAddress(unsigned long long uiLastAddr,const char* num)
	{
		return uiLastAddr;
	}

	/*************************************************************************
	* Function Name : getInputScale
	* Description   : get layer input scale value from bin file
	* Parameters    : fileFp -- input file
	* Returns       : 0 -- success
	**************************************************************************/
	int ReLU::getInputScale(FILE *fileFp)
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
	void ReLU::setQuantize(int iQuantize,float fIScale)
	{
		iQuantizeFlag = iQuantize;
		fBsQn = fIScale;
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
	void ReLU::setFp32(int iFp32,unsigned long long uiBiasAddr,unsigned long long uiWScaleAddr,float fIScale)
	{
		iFp32Flag = iFp32;
		uiBsAluSrc = uiBiasAddr;
		uiBsMulSrc = uiWScaleAddr;
		fOprand = fIScale;
	}

	/*************************************************************************
	* Function Name : getNextInputOutputAddr
	* Description   : ge tNext Input Output Address
	* Parameters    : uiAddr -- input address
	* Returns       : uiNewAddr -- next layer input address
	**************************************************************************/
	unsigned long long ReLU::getNextInputOutputAddr(unsigned long long uiAddr,unsigned int uiOneSeg,\
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
	unsigned long long ReLU::setRegisterValue(unsigned long long uiLastAddr,unsigned int uiOneSeg,\
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
	int ReLU::setRegisterValue(std::vector<unsigned long long> uiInputAddr, std::vector<unsigned long long> uiOutputAddr,const char* num)
	{
		return 0;
	}

	/*************************************************************************
	* Function Name : writeDDRInfoInputOutput
	* Description   : write input output ddr information to putput file
	* Parameters    : fileOutFp -- output file
	* Returns       : void
	**************************************************************************/
	void ReLU::writeDDRInfoInputOutput(FILE *fileOutFp)
	{
		char cPrintfbuf[100];
		sprintf(cPrintfbuf,"relu\n");
		fwrite(cPrintfbuf, strlen(cPrintfbuf), 1, fileOutFp);
		sprintf(cPrintfbuf, "input address %llx\n", uiRegMatSrcAddr);
		fwrite(cPrintfbuf, strlen(cPrintfbuf),1,fileOutFp);
		sprintf(cPrintfbuf,"output address %llx\n", uiRegMatDstAddr);
		fwrite(cPrintfbuf, strlen(cPrintfbuf), 1, fileOutFp);
	}

	/*************************************************************************
	* Function Name : writeRegToJson
	* Description   : write register to json struct
	* Parameters    : json -- json struct
	* Returns       : NULL
	**************************************************************************/
	void ReLU::writeRegToJson(cJSON *json)
	{
		char cName[200];
		memset(cName,0,sizeof(cName));
		sprintf(cName,"relu");
		cJSON *array = NULL;
		cJSON_AddItemToObject(json,cName,array=cJSON_CreateArray());
		memset(cName,0,sizeof(cName));
		//MAT_SRC_ADDR
		cJSON *obj = NULL;
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08llx",uiRegMatSrcAddr);
		cJSON_AddStringToObject(obj,"MAT_SRC_ADDR",cName);
		memset(cName,0,sizeof(cName));
		//MAT_DST_ADDR
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08llx",uiRegMatDstAddr);
		cJSON_AddStringToObject(obj,"MAT_DST_ADDR",cName);
		memset(cName,0,sizeof(cName));
		//CTRL
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08x",uiRegReluCtrl);
		cJSON_AddStringToObject(obj,"CTRL",cName);
		memset(cName,0,sizeof(cName));
		//CUBE_IN_WIDTH
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08x",uiRegCubeInWidth);
		cJSON_AddStringToObject(obj,"CUBE_IN_WIDTH",cName);
		memset(cName,0,sizeof(cName));
		//CUBE_IN_HEIGHT
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08x",uiRegCubeInHeight);
		cJSON_AddStringToObject(obj,"CUBE_IN_HEIGHT",cName);
		memset(cName,0,sizeof(cName));
		//CUBE_IN_CHANNEL
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08x",uiRegCubeInChannel);
		cJSON_AddStringToObject(obj,"CUBE_IN_CHANNEL",cName);
		memset(cName,0,sizeof(cName));
		//BS_BYPASS
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08x",uiRegBsBypass);
		cJSON_AddStringToObject(obj,"BS_BYPASS",cName);
		memset(cName,0,sizeof(cName));
		//BS_ALU_SRC
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08llx",uiRegBsAluSrc);
		cJSON_AddStringToObject(obj,"BS_ALU_SRC",cName);
		memset(cName,0,sizeof(cName));
		//BS_MUL_SRC
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08llx",uiRegBsMulSrc);
		cJSON_AddStringToObject(obj,"BS_MUL_SRC",cName);
		memset(cName,0,sizeof(cName));
		//BS_OPRAND
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08x",uiRegBsOprand);
		cJSON_AddStringToObject(obj,"BS_OPRAND",cName);
		memset(cName,0,sizeof(cName));
		//BS_QN
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08x",uiRegBsQn);
		cJSON_AddStringToObject(obj,"BS_QN",cName);
		memset(cName,0,sizeof(cName));
		//CLASSES_LENGTH
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08x",uiRegClassesLength);
		cJSON_AddStringToObject(obj,"CLASSES_LENGTH",cName);
		memset(cName,0,sizeof(cName));
		//CS_CFG
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08x",uiRegBsCfg);
		cJSON_AddStringToObject(obj,"CS_CFG",cName);
		memset(cName,0,sizeof(cName));
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
	int ReLU::writeBinFile(FILE *fileInFp,FILE *fileOutFp,int iLayerIndex,const char* num)
	{
		if (reluWriteBinFlag)
		{
			char *pcCharBuf = (char *)malloc(LayerCommon.cInputChannel*sizeof(float));
			char cPrintBuf[PRINT_BUF_SIZE];
			for (int j = 0; j < LayerCommon.cInputChannel; j++)
			{
				float relu = slope;
				memcpy(cPrintBuf, &relu, sizeof(cPrintBuf));
				for (int k=0; k < (int)(sizeof(float)); k++)
				{
					fwrite(&cPrintBuf[3-k],sizeof(char),1,fileOutFp);
				}
			}
			free(pcCharBuf);
		}
		return 0;
	}

	/*************************************************************************
	* Function Name : writeddrBinFile
	* Description   : write register value to bin file
	* Parameters    : fileRp -- out tmmodel bin file after the layer date		   
	* Returns       : 0 -- success
	**************************************************************************/
	int ReLU::writeddrBinFile(FILE *fileRp)
	{
		return 0;
	}
}

