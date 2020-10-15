/*
 * InnerProduct.cpp
 *
 *  Created on: Jun 12, 2019
 *      Author: doyle
 */

#include "InnerProduct.h"

namespace tmnet
{
	/*************************************************************************
	* Function Name : InnerProduct
	* Description   : construct function ,initialize all the variable
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	InnerProduct::InnerProduct()
	{
		cRunFlag = 1;
		cConcatFlag = 0;
	}

	/*************************************************************************
	* Function Name : ~InnerProduct
	* Description   : deconstruct function
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	InnerProduct::~InnerProduct()
	{

	}

	/*************************************************************************
	* Function Name : setLayerBinDataSize
	* Description   : set Layer Bin Data Size
	* Parameters    : NULL
	* Returns       : NULL
	**************************************************************************/
	void InnerProduct::setLayerBinDataSize(void)
	{
		//weight: float
		//bias: float
		uiBinDataSize = (data_size + num_output) * sizeof(float);
	}

	/*************************************************************************
	* Function Name : loadParam
	* Description   : layer load param data
	* Parameters    : fileFp -- input param file
	* Returns       : 0 -- success
	**************************************************************************/
	int InnerProduct::loadParam(FILE* fileFp, int output_num)
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
			case 1:
				bias_term = value;
				break;
			case 2:
				data_size = value;
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
int InnerProduct::calculateFeaSize(std::vector<unsigned int> iIw, std::vector<unsigned int> iIh, std::vector<unsigned int> iIc)
{
	LayerCommon.iInFeaW = iIw[0];
	LayerCommon.iInFeaH = iIh[0];
	LayerCommon.cInputChannel = iIc[0];

	LayerCommon.iOutFeaW = 1;
	LayerCommon.iOutFeaH = 1;
	LayerCommon.cOutputChannel = num_output;

	//input:float
	LayerCommon.uiInputSize = iIw[0]*iIh[0]*iIc[0]*sizeof(float);
	//output:float
	LayerCommon.uiOutputSize = num_output*sizeof(float);
	return 0;
}

	/*************************************************************************
	* Function Name : getNewAddrFormDataSize
	* Description   : calculate the next bin data address (4K Algin)
	* Parameters    : NULL
	* Returns       : uiNewAddress -- nextt address
	**************************************************************************/
	unsigned long long InnerProduct::getNewAddrFormDataSize(unsigned int uiDataSize,unsigned long long uiAddress)
	{
		const unsigned int uiOne= ALGN_SIZE;
		unsigned int uiNewAddress = 0;
		unsigned int ucStepCount = 0;

		if(uiDataSize%uiOne)
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
	unsigned long long InnerProduct::fillDDRAddress(unsigned long long uiLastAddr,const char* num)
	{
		unsigned int uiDataSize = 0;
		//write weight address and calculate the size of weight data    type:float
		uiWeightAddr = uiLastAddr;
		uiDataSize = data_size*sizeof(float);
		uiLastAddr = getNewAddrFormDataSize(uiDataSize,uiLastAddr);
		//write bias address and calculate the size of weight data    type:float
		uiBiasAddr = uiLastAddr;
		uiDataSize = num_output*sizeof(float);
		uiLastAddr = getNewAddrFormDataSize(uiDataSize,uiLastAddr);

		return uiLastAddr;
	}

	/*************************************************************************
	* Function Name : getInputScale
	* Description   : get layer input scale value from bin file
	* Parameters    : fileFp -- input file
	* Returns       : 0 -- success
	**************************************************************************/
	int InnerProduct::getInputScale(FILE *fileFp)
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
		unsigned char *charBuf = (unsigned char *)malloc(data_size* sizeof(float));
		float *fc_b = (float *)malloc(num_output* sizeof(float));

		//read tag
		rc = fread(&flag_struct, sizeof(flag_struct), 1,fileFp);
		assert(rc > 0);
		//read weights
		rc = fread(charBuf, data_size * sizeof(float), 1, fileFp);
		assert(rc > 0);
		//read bias
		rc = fread(fc_b, num_output * sizeof(float), 1, fileFp);
		assert(rc > 0);
		free(charBuf);
		free(fc_b);
		return rc;
	}

	/*************************************************************************
	* Function Name : getNextInputOutputAddr
	* Description   : ge tNext Input Output Address
	* Parameters    : uiAddr -- input address
	* Returns       : uiNewAddr -- next layer input address
	**************************************************************************/
	unsigned long long InnerProduct::getNextInputOutputAddr(unsigned long long uiAddr,unsigned int uiOneSeg,\
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
	unsigned long long InnerProduct::setRegisterValue(unsigned long long uiLastAddr,unsigned int uiOneSeg,\
														const unsigned int uiOriAddr,char cBufferNum)
	{
		//source data address
		uiRegInDataSrcAddr = uiLastAddr;
		uiLastAddr = getNextInputOutputAddr(uiLastAddr,uiOneSeg,uiOriAddr,cBufferNum);
		uiRegInDataClassesLength = LayerCommon.iInFeaW*LayerCommon.iInFeaH*LayerCommon.cInputChannel * sizeof(float);
		//output data address
		uiRegOutDataDstAddr = uiLastAddr;
		//weight data address
		uiRegWDataSrcAddr = uiWeightAddr;
		uiRegWDataClassesLength = data_size * sizeof(float);
		//start
		uiRegFcStart = 1;
		//bias address
		uiRegBiasSrcAddr = uiBiasAddr;
		//out number
		uiRegOutputNumber = LayerCommon.cOutputChannel;

		return uiLastAddr;
	}

	/*************************************************************************
	* Function Name : setRegisterValue
	* Description   : set layer register value
	* Parameters    : uiInputAddr -- input address
	* 				  uiOutputAddr -- output address
	* Returns       : 1-success
	**************************************************************************/
	int InnerProduct::setRegisterValue(std::vector<unsigned long long> uiInputAddr, std::vector<unsigned long long> uiOutputAddr,const char* num)
	{
		//source data address
		uiRegInDataSrcAddr = uiInputAddr[0];
		uiRegInDataClassesLength = LayerCommon.iInFeaW*LayerCommon.iInFeaH*LayerCommon.cInputChannel * sizeof(float);
		//output data address
		//uiRegOutDataDstAddr = uiOutputAddr;
		//weight data address
		uiRegWDataSrcAddr = uiWeightAddr;
		uiRegWDataClassesLength = data_size * sizeof(float);
		//start
		uiRegFcStart = 1;
		//bias address
		uiRegBiasSrcAddr = uiBiasAddr;
		//out number
		uiRegOutputNumber = LayerCommon.cOutputChannel;
		return 1;
	}

	/*************************************************************************
	* Function Name : writeDDRInfoWeight
	* Description   : write weight data information to output file
	* Parameters    : fileOutFp -- output file
	* Returns       : void
	**************************************************************************/
	void InnerProduct::writeDDRInfoWeight(FILE *fileOutFp)
	{
		char cPrintfbuf[100];

		sprintf(cPrintfbuf,"fc\n");
		fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);

		sprintf(cPrintfbuf,"weight address %llx\n",uiWeightAddr);
		fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);

		sprintf(cPrintfbuf,"bias address %llx\n",uiBiasAddr);
		fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
	}

	/*************************************************************************
	* Function Name : writeDDRInfoInputOutput
	* Description   : write input output data information to output file
	* Parameters    : fileOutFp -- output file
	* Returns       : void
	**************************************************************************/
	void InnerProduct::writeDDRInfoInputOutput(FILE *fileOutFp)
	{
		char cPrintfbuf[100];

		sprintf(cPrintfbuf,"fc\n");
		fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);

		sprintf(cPrintfbuf,"input address %llx\n",uiRegInDataSrcAddr);
		fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);

		sprintf(cPrintfbuf,"output address %llx\n",uiRegOutDataDstAddr);
		fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
	}

	/*************************************************************************
	* Function Name : writeRegToJson
	* Description   : write register to json struct
	* Parameters    : json -- json struct
	* Returns       : NULL
	**************************************************************************/
	void InnerProduct::writeRegToJson(cJSON *json)
	{
		char cName[200];
		memset(cName,0,sizeof(cName));

		sprintf(cName,"fc");
		cJSON *array = NULL;
		cJSON_AddItemToObject(json,cName,array=cJSON_CreateArray());
		memset(cName,0,sizeof(cName));
		//INDATA_SRC_ADDR
		cJSON *obj = NULL;
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08llx",uiRegInDataSrcAddr);
		cJSON_AddStringToObject(obj,"INDATA_SRC_ADDR",cName);
		memset(cName,0,sizeof(cName));
		//WDATA_SRC_ADDR
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08llx",uiRegWDataSrcAddr);
		cJSON_AddStringToObject(obj,"WDATA_SRC_ADDR",cName);
		memset(cName,0,sizeof(cName));
		//INDATA_CLASSES_LENGTH
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08x",uiRegInDataClassesLength);
		cJSON_AddStringToObject(obj,"INDATA_CLASSES_LENGTH",cName);
		memset(cName,0,sizeof(cName));
		//WDATA_CLASSES_LENGTH
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08x",uiRegWDataClassesLength);
		cJSON_AddStringToObject(obj,"WDATA_CLASSES_LENGTH",cName);
		memset(cName,0,sizeof(cName));
		//OUTDATA_DST_ADDR
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08llx",uiRegOutDataDstAddr);
		cJSON_AddStringToObject(obj,"OUTDATA_DST_ADDR",cName);
		memset(cName,0,sizeof(cName));
		//FC_START
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08x",uiRegFcStart);
		cJSON_AddStringToObject(obj,"FC_START",cName);
		memset(cName,0,sizeof(cName));
		//FC_FINISH
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08x",uiRegFcFinish);
		cJSON_AddStringToObject(obj,"FC_FINISH",cName);
		memset(cName,0,sizeof(cName));
		//OUT_DATA
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08x",uiRegOutData);
		cJSON_AddStringToObject(obj,"OUT_DATA",cName);
		memset(cName,0,sizeof(cName));
		//BIAS_SRC_ADDR
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08llx",uiRegBiasSrcAddr);
		cJSON_AddStringToObject(obj,"BIAS_SRC_ADDR",cName);
		memset(cName,0,sizeof(cName));
		//OUT_NUMBER
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08x",uiRegOutputNumber);
		cJSON_AddStringToObject(obj,"OUT_NUMBER",cName);
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
	int InnerProduct::writeBinFile(int iDropCount,FILE *fileInFp,FILE *fileOutFp,int iLayerIndex,char cBit)
	{
		int iBinLayIndex = 0;
		char cLayerHead[LAYER_HEADER_SIZE] = {(char)0xAA,(char)0xBB,(char)0xCC,(char)0xDD};
		char cWriteBuf = 0x00;
		unsigned int iInOutIndex = 0;
		unsigned long long ullAddrBuf = 0;
		unsigned int uiRegBuf = 0;
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
		/* fread file counts */
		int rc = 0;
		iBinLayIndex = iLayerIndex-iDropCount;
		//write header (1byte)
		fwrite(cLayerHead,sizeof(char),sizeof(cLayerHead),fileOutFp);
		//write layer index (4bytes)
		fwrite(&iBinLayIndex,sizeof(unsigned int),1,fileOutFp);
		//write type (1byte)
		cWriteBuf = 0x04;//LAYER_FC;
		fwrite(&cWriteBuf,sizeof(char),1,fileOutFp);
		//write input number (1byte) and index (4bytes for each)
		cWriteBuf = LayerCommon.viInput.size();
		fwrite(&cWriteBuf,sizeof(char),1,fileOutFp);
		for(int iWriteIn=0; iWriteIn<cWriteBuf; iWriteIn++)
		{
			iInOutIndex = LayerCommon.viInput[iWriteIn];
			iInOutIndex -= iDropCount;
			fwrite(&iInOutIndex,sizeof(unsigned int),1,fileOutFp);
		}
		//write out number (1byte) and index (4bytes for each)
		cWriteBuf = LayerCommon.viOutput.size();
		fwrite(&cWriteBuf,sizeof(char),1,fileOutFp);
		for(int iWriteOut=0; iWriteOut<cWriteBuf; iWriteOut++)
		{
			iInOutIndex = LayerCommon.viOutput[iWriteOut];
			iInOutIndex -= iDropCount;
			fwrite(&iInOutIndex,sizeof(unsigned int),1,fileOutFp);
		}

		//write register value
		if(cBit == ADD_64BIT)
		{
			//64bit address
			ullAddrBuf = uiRegInDataSrcAddr;
			fwrite(&ullAddrBuf,sizeof(unsigned long long),1,fileOutFp);
			ullAddrBuf = uiRegWDataSrcAddr;
			fwrite(&ullAddrBuf,sizeof(unsigned long long),1,fileOutFp);

		}
		else
		{
			//32bit address
			ullAddrBuf = uiRegInDataSrcAddr;
			fwrite(&ullAddrBuf,sizeof(unsigned int),1,fileOutFp);
			ullAddrBuf = uiRegWDataSrcAddr;
			fwrite(&ullAddrBuf,sizeof(unsigned int),1,fileOutFp);
		}

		uiRegBuf = uiRegInDataClassesLength;
		fwrite(&uiRegBuf,sizeof(unsigned int),1,fileOutFp);
		uiRegBuf = uiRegWDataClassesLength;
		fwrite(&uiRegBuf,sizeof(unsigned int),1,fileOutFp);
		uiRegBuf = uiRegOutDataDstAddr;
		fwrite(&uiRegBuf,sizeof(unsigned int),1,fileOutFp);
		uiRegBuf = uiRegFcStart;
		fwrite(&uiRegBuf,sizeof(unsigned int),1,fileOutFp);
		uiRegBuf = uiRegFcFinish;
		fwrite(&uiRegBuf,sizeof(unsigned int),1,fileOutFp);
		uiRegBuf = uiRegOutData;
		fwrite(&uiRegBuf,sizeof(unsigned int),1,fileOutFp);

		if(cBit == ADD_64BIT)
		{
			ullAddrBuf = uiRegBiasSrcAddr;
			fwrite(&ullAddrBuf,sizeof(unsigned long long),1,fileOutFp);

		}
		else
		{
			ullAddrBuf = uiRegBiasSrcAddr;
			fwrite(&ullAddrBuf,sizeof(unsigned int),1,fileOutFp);
		}

		uiRegBuf = uiRegOutputNumber;
		fwrite(&uiRegBuf,sizeof(unsigned int),1,fileOutFp);

		//write bin file data
		unsigned char *charBuf = (unsigned char *)malloc(data_size* sizeof(float));
		float *fc_b = (float *)malloc(num_output* sizeof(float));
		char *pcDataBuf = NULL;
		int iFeatureSize = LayerCommon.iInFeaW*LayerCommon.iInFeaW;
		char cFCEightChannel[DATA_CHANNEL_NUM*iFeatureSize*sizeof(float)];
		char cPrintBuf[PRINT_BUF_SIZE];
		if(charBuf == NULL || fc_b == NULL)
		{
			printf("fc malloc wrong %d\n",iLayerIndex);
			return -1;
		}

		rc = fread(&flag_struct, sizeof(flag_struct), 1,fileInFp);
		assert(rc > 0);
		//weight
		rc = fread(charBuf, data_size * sizeof(float), 1, fileInFp);
		assert(rc > 0);
		pcDataBuf = (char *)charBuf;
		//only the first layer weights need be te rearranged

		if(cFirstFlag)
		{
			for(size_t j=0; j<num_output; j++)
			{
				for(size_t m=0; m<(data_size/(num_output*iFeatureSize*8)) ;m++)
				{
					memcpy(cFCEightChannel,pcDataBuf,sizeof(cFCEightChannel));
					for(int n=0; n<iFeatureSize; n++)
					{
						for(int p=0; p<DATA_CHANNEL_NUM; p++)
						{
							memcpy(cPrintBuf,&cFCEightChannel[(iFeatureSize*p+n)*sizeof(float)],sizeof(cPrintBuf));
							for(size_t k=0; k<sizeof(cPrintBuf); k++)
							{
								fwrite(&cPrintBuf[k],sizeof(char), 1, fileOutFp);
							}
						}
					}
					pcDataBuf += sizeof(cFCEightChannel);
				}
			}
		}
		else
		{
			for (size_t j=0; j<data_size; j++)
			{
				memcpy(cPrintBuf,&charBuf[sizeof(cPrintBuf)*j],sizeof(cPrintBuf));
				for(size_t k=0; k<sizeof(cPrintBuf); k++)
				{
					fwrite(&cPrintBuf[k],sizeof(char), 1, fileOutFp);
				}
			}
		}

		//bias
		rc = fread(fc_b, num_output * sizeof(float), 1, fileInFp);
		if(rc <= 0){
			return -1;
		}
	
		for (size_t j=0; j<num_output; j++)
		{
			memcpy(cPrintBuf,&fc_b[j],sizeof(cPrintBuf));
			for(size_t k=0; k<sizeof(cPrintBuf); k++)
			{
				fwrite(&cPrintBuf[k],sizeof(char), 1, fileOutFp);
			}
		}

		free(charBuf);
		free(fc_b);

		return iDropCount;
	}
}


