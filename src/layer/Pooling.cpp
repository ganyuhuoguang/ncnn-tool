/*
 * Pooling.cpp
 *
 *  Created on: Jun 12, 2019
 *      Author: doyle
 */

#include "Pooling.h"
#include <limits.h>

namespace tmnet
{
	/*************************************************************************
	* Function Name : Pooling
	* Description   : construct function ,initialize all the variable
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	Pooling::Pooling()
	{
		iQuantizeFlag = 0;
		fArithScale = 0;
		cRunFlag = 0;
		cConcatFlag = 0;
	}

	/*************************************************************************
	* Function Name : ~Pooling
	* Description   : deconstruct function
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	Pooling::~Pooling()
	{

	}

	/*************************************************************************
	* Function Name : loadParam
	* Description   : layer load param data
	* Parameters    : fileFp -- input param file
	* Returns       : 0 -- success
	**************************************************************************/
	int Pooling::loadParam(FILE* fileFp, int output_num)
	{
	    int id = 0;
	    int value = 0;
		kernel_h = UCHAR_MAX;
		stride_h = UCHAR_MAX;

		pad_top = UCHAR_MAX;
		pad_right = UCHAR_MAX;
		pad_left = UCHAR_MAX;
		pad_bottom = UCHAR_MAX;
		pad_mode = 0;  //if there is no pooling, the pad_mode defaults to 0 
	    pooltype = output_num;
		while (fscanf(fileFp, "%d=%d", &id,&value) == 2)
		{
			switch (id)
			{
				case 0:
					pooltype = value;
					break;
				case 1:
					kernel_w = value;
					PoolingConfig.pooling_size = kernel_w;
					break;
				case 2:
					stride_w = value;
					PoolingConfig.pooling_stride = stride_w;
					break;
				case 3:
					pad_left = value;
					PoolingConfig.pooling_padding = pad_left;
					break;
				case 4:
					global_pooling = value;
					break;
				case 5:
					pad_mode = value;
					break;
				case 11:
					kernel_h = value;
					break;
				case 12:
					stride_h = value;
					break;
				case 13:
					pad_top = value;
					break;
				case 14:
					pad_right = value;
					break;
				case 15:
					pad_bottom = value;
					break;
				default:
					break;
			}

			if (kernel_h == UCHAR_MAX)
			{
				kernel_h = kernel_w;
			}
			if(stride_h == UCHAR_MAX)	
			{
				stride_h = stride_w;
			}
			if(pad_top == UCHAR_MAX)	
			{
				pad_top = pad_left;	
			}
			if(pad_right == UCHAR_MAX)
			{
				pad_right = pad_left;
			}
			if(pad_bottom == UCHAR_MAX)
		    {
				pad_bottom = pad_top;
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
	int Pooling::calculateFeaSize(std::vector<unsigned int> iIw, std::vector<unsigned int> iIh, std::vector<unsigned int> iIc)
	{
		LayerCommon.iInFeaW = iIw[0];
		LayerCommon.iInFeaH = iIh[0];
		LayerCommon.cInputChannel = iIc[0];
	    int wtailpad = 0;

	    if (pad_mode == POOLING_PADING_MODE_FULL) // full padding
	    {
			int wtail = (LayerCommon.iInFeaW + pad_left + pad_right - kernel_w) % stride_w;

	        if (wtail != 0)
	            wtailpad = stride_w - wtail;
		
			int left = pad_left;
			int right = pad_right + wtailpad;

			LayerCommon.iOutFeaW = (LayerCommon.iInFeaW + left + right - kernel_w) / stride_w + 1;
		}
	    else if (pad_mode == POOLING_PADING_MODE_VALID) // valid padding
	    {
			LayerCommon.iOutFeaW = (LayerCommon.iInFeaW + pad_left + pad_right - kernel_w) / stride_w + 1;
		}
	    else if (pad_mode == POOLING_PADING_MODE_SAME) // tensorflow padding=SAME
	    {
			int wpad = kernel_w + (LayerCommon.iInFeaW - 1) / stride_w * stride_w - LayerCommon.iInFeaW;
	        int hpad = kernel_h + (LayerCommon.iInFeaH - 1) / stride_h * stride_h - LayerCommon.iInFeaH;
			if (wpad > 0 || hpad > 0)
	        {
				int left = pad_left;
				int right = pad_right + wtailpad;

				LayerCommon.iOutFeaW = (LayerCommon.iInFeaW + left + right - kernel_w) / stride_w + 1;
			}
			else
			{
				LayerCommon.iOutFeaW = (LayerCommon.iInFeaW + pad_left + pad_right - kernel_w) / stride_w + 1;
			}
		}

		LayerCommon.iOutFeaH = LayerCommon.iOutFeaW;
		LayerCommon.cOutputChannel = iIc[0];

		//input:float
		LayerCommon.uiInputSize = iIw[0]*iIh[0]*iIc[0]*sizeof(float);
		//output:float or int8
		if(iQuantizeFlag)
		{
			//int8
			LayerCommon.uiOutputSize = LayerCommon.iOutFeaW*LayerCommon.iOutFeaH*\
												LayerCommon.cOutputChannel;
		}
		else
		{
			//float
			LayerCommon.uiOutputSize = LayerCommon.iOutFeaW*LayerCommon.iOutFeaH*\
														LayerCommon.cOutputChannel*sizeof(float);
		}
		return 0;
	}

	/*************************************************************************
	* Function Name : fillDDRAddress
	* Description   : fill the bin data ddr address
	* Parameters    : uiLastAddr -- last address
	* Returns       : 0 -- success
	**************************************************************************/
	unsigned long long Pooling::fillDDRAddress(unsigned long long uiLastAddr,const char* num)
	{
		return uiLastAddr;
	}

	/*************************************************************************
	* Function Name : getInputScale
	* Description   : get layer input scale value from bin file
	* Parameters    : fileFp -- input file
	* Returns       : 0 -- success
	**************************************************************************/
	int Pooling::getInputScale(FILE *fileFp)
	{
		return 0;
	}

	/*************************************************************************
	* Function Name : getNextInputOutputAddr
	* Description   : ge tNext Input Output Address
	* Parameters    : uiAddr -- input address
	* Returns       : uiNewAddr -- next layer input address
	**************************************************************************/
	unsigned long long Pooling::getNextInputOutputAddr(unsigned long long uiAddr,unsigned int uiOneSeg,\
															unsigned int uiOriAddr,char cBufferNum)
	{
		unsigned int uiNewAddr = 0;

		if (uiAddr >= ((cBufferNum-1)*uiOneSeg + uiOriAddr))
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
	unsigned long long Pooling::setRegisterValue(unsigned long long uiLastAddr,unsigned int uiOneSeg,\
												const unsigned int uiOriAddr,char cBufferNum)
	{
		unsigned int fBuf = 0;
		//pooling layer matrix source address
		uiRegMatSrcAddr = uiLastAddr;
		uiLastAddr = getNextInputOutputAddr(uiLastAddr,uiOneSeg,uiOriAddr,cBufferNum);
		//pooling layer matrix dst address
		uiRegMatDstAddr = uiLastAddr;
		//input width
		uiRegMatRowIn = LayerCommon.iInFeaW;
		//input height
		uiRegMatColIn = LayerCommon.iInFeaH;
		//output width
		uiRegMatRowOut = LayerCommon.iOutFeaW;
		//output height
		uiRegMatColOut = LayerCommon.iOutFeaH;
		//matrix channel
		uiRegMatChannel = LayerCommon.cInputChannel;
		//arith_a input matrix total size
		uiRegAirthA = LayerCommon.iInFeaW*LayerCommon.iInFeaH*LayerCommon.cInputChannel;
		//arith_b output matrix total size
		uiRegAirthB = LayerCommon.iOutFeaW*LayerCommon.iOutFeaH*LayerCommon.cOutputChannel;
		//pooling control register
		if(iQuantizeFlag)
		{
			memcpy(&fBuf,&fArithScale,sizeof(float));
			uiRegAirthScale = fBuf;
			uiRegCtrl =poolRegCtrl(1);
		}
		else
		{
			uiRegCtrl =poolRegCtrl(0);
		}
		return uiLastAddr;
	}

	/*************************************************************************
	* Function Name : setRegisterValue
	* Description   : set layer register value
	* Parameters    : uiInputAddr -- input address
	*                 uiOutputAddr -- output address
	* Returns       : success
	**************************************************************************/
	int Pooling::setRegisterValue(std::vector<unsigned long long> uiInputAddr, std::vector<unsigned long long> uiOutputAddr,const char* num)
	{
		unsigned int fBuf = 0;
		//pooling layer matrix source address
		uiRegMatSrcAddr = uiInputAddr[0];
		//pooling layer matrix dst address
		//  uiRegMatDstAddr = uiOutputAddr[0];
		//input width
		uiRegMatRowIn = LayerCommon.iInFeaW;
		//input height
		uiRegMatColIn = LayerCommon.iInFeaH;
		//output width
		uiRegMatRowOut = LayerCommon.iOutFeaW;
		//output height
		uiRegMatColOut = LayerCommon.iOutFeaH;
		//matrix channel
		uiRegMatChannel = LayerCommon.cInputChannel;
		//arith_a input matrix total size
		uiRegAirthA = LayerCommon.iInFeaW*LayerCommon.iInFeaH*LayerCommon.cInputChannel;
		//arith_b output matrix total size
		uiRegAirthB = LayerCommon.iOutFeaW*LayerCommon.iOutFeaH*LayerCommon.cOutputChannel;
		//pooling control register
		if(iQuantizeFlag)
		{
			memcpy(&fBuf,&fArithScale,sizeof(float));uiRegAirthScale = fBuf;
			uiRegCtrl =poolRegCtrl(1);
		}
		else
		{
			uiRegCtrl =poolRegCtrl(0);
		}
		return 0;
	}

	/*************************************************************************
	* Function Name : writeDDRInfoInputOutput
	* Description   : write input output ddr information to putput file
	* Parameters    : fileOutFp -- output file
	* Returns       : void
	**************************************************************************/
	void Pooling::writeDDRInfoInputOutput(FILE *fileOutFp)
	{
		char cPrintfbuf[100];

		sprintf(cPrintfbuf,"pool\n");
		fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);

		sprintf(cPrintfbuf,"input address %llx\n",uiRegMatSrcAddr);
		fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);

		sprintf(cPrintfbuf,"output address %llx\n",uiRegMatDstAddr);
		fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
	}

	/*************************************************************************
	* Function Name : poolRegCtrl
	* Description   : get ctrl register
	* Parameters    : ucQuanEnable -- quantize enable flag
	* Returns       : uiLastAddr -- next address
	**************************************************************************/
	unsigned int Pooling::poolRegCtrl(unsigned char ucQuanEnable)
	{
		unsigned int uiRegValue = 0;
		unsigned int uiValueBuf = 0;

		uiValueBuf = ucQuanEnable;
		uiRegValue |= uiValueBuf<<REG_POOL_QUAN_BIT;

		uiValueBuf = pooltype;
		uiRegValue |= uiValueBuf<<REG_POOL_MODE_BIT;

		uiValueBuf = pad_left;
		uiRegValue |= uiValueBuf<<REG_POOL_PAD_BIT;

		uiValueBuf = stride_w;
		uiRegValue |= uiValueBuf<<REG_POOL_STRIDE_BIT;

		uiValueBuf = kernel_w;
		uiRegValue |= uiValueBuf<<REG_POOL_SZ_BIT;

		uiRegValue += 1;
 
		return uiRegValue;
	}

	/*************************************************************************
	* Function Name : setQuantize
	* Description   : set quantize value
	* Parameters    : iQuantize -- quantize enable flag
	*                 fScale -- quantize scale
	* Returns       : uiLastAddr -- next address
	**************************************************************************/
	void Pooling::setQuantize(int iQuantize,float fScale)
	{
		iQuantizeFlag = iQuantize;
		fArithScale = fScale;
	}

	/*************************************************************************
	* Function Name : writeRegToJson
	* Description   : write register to json struct
	* Parameters    : json -- json struct
	* Returns       : NULL
	**************************************************************************/
	void Pooling::writeRegToJson(cJSON *json)
	{
		char cName[200];

		memset(cName,0,sizeof(cName));
		sprintf(cName,"pool");
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
		//MAT_ROW_IN
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08x",uiRegMatRowIn);
		cJSON_AddStringToObject(obj,"MAT_ROW_IN",cName);
		memset(cName,0,sizeof(cName));
		//MAT_COL_IN
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08x",uiRegMatColIn);
		cJSON_AddStringToObject(obj,"MAT_COL_IN",cName);
		memset(cName,0,sizeof(cName));
		//MAT_ROW_OUT
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08x",uiRegMatRowOut);
		cJSON_AddStringToObject(obj,"MAT_ROW_OUT",cName);
		memset(cName,0,sizeof(cName));
		//MAT_COL_OUT
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08x",uiRegMatColOut);
		cJSON_AddStringToObject(obj,"MAT_COL_OUT",cName);
		memset(cName,0,sizeof(cName));
		//MAT_CHANNEL
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08x",uiRegMatChannel);
		cJSON_AddStringToObject(obj,"MAT_CHANNEL",cName);
		memset(cName,0,sizeof(cName));
		//ARITH_A
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08x",uiRegAirthA);
		cJSON_AddStringToObject(obj,"ARITH_A",cName);
		memset(cName,0,sizeof(cName));
		//ARITH_B
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08x",uiRegAirthB);
		cJSON_AddStringToObject(obj,"ARITH_B",cName);
		memset(cName,0,sizeof(cName));
		//ARITH_SCALE
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08x",uiRegAirthScale);
		cJSON_AddStringToObject(obj,"ARITH_SCALE",cName);
		memset(cName,0,sizeof(cName));
		//CTRL
		cJSON_AddItemToArray(array,obj=cJSON_CreateObject());
		sprintf(cName,"0x%08x",uiRegCtrl);
		cJSON_AddStringToObject(obj,"CTRL",cName);
		memset(cName,0,sizeof(cName));
	}

	/*************************************************************************
	* Function Name : writeddrBinFile
	* Description   : write register value to bin file
	* Parameters    : fileRp -- out tmmodel bin file after the layer date		   
	* Returns       : 0 -- success
	**************************************************************************/
	int Pooling::writeddrBinFile(FILE *fileRp)
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
	int Pooling::writeBinFile(FILE *fileInFp,FILE *fileOutFp,int iLayerIndex,const char* num)
	{
		return 0;
	}
}






