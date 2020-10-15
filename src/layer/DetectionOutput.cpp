/*
 * ReLU.cpp
 *
 *  Created on: Jun 12, 2019
 *      Author: doyle
 */

#include "DetectionOutput.h"
#include <sys/stat.h>

namespace tmnet
{
	/*************************************************************************
	* Function Name : ReLU
	* Description   : construct function ,initialize all the variable
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	DetectionOutput::DetectionOutput()
	{
		//quantize value initialize
		iQuantizeFlag = 0;
		fBsQn = 0;
		//int32->fp32 value initialize
		iFp32Flag = 0;
		uiBsAluSrc = 0;
		uiBsMulSrc = 0;
		fOprand = 0;
		cRunFlag = 1;
		cConcatFlag = 0;
	}

	/*************************************************************************
	* Function Name : ~ReLU
	* Description   : deconstruct function
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	DetectionOutput::~DetectionOutput()
	{

	}

	/*************************************************************************
	* Function Name : loadParam
	* Description   : layer load param data
	* Parameters    : fileFp -- input param file
	* Returns       : 0 -- success
	**************************************************************************/
	int DetectionOutput::loadParam(FILE* fileFp, int output_num)
	{
		int id = 0;
		float value = 0.0f;

		num_class = output_num;//todo
		while (fscanf(fileFp, "%d=%f", &id,&value) == 2)
		{
			switch (id)
			{
			case 0:
				num_class = value;
				break;
			case 1:
				nms_threshold = value;
				break;
			case 2:
				nms_top_k = value;
				break;
			case 3:
				keep_top_k = value;
				break;
			case 4:
				confidence_threshold = value;
				break;
			case 5:
				variances[0] = value;
				break;
			case 6:
				variances[1] = value;
				break;
			case 7:
				variances[2] = value;
				break;
			case 8:
				variances[3] = value;
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
	int DetectionOutput::calculateFeaSize(std::vector<unsigned int> iIw, std::vector<unsigned int> iIh, std::vector<unsigned int> iIc)
	{
		LayerCommon.iInFeaW = iIw[0];
		LayerCommon.iInFeaH = iIh[0];
		LayerCommon.cInputChannel = iIc[0];

		LayerCommon.iOutFeaW = iIw[0];
		LayerCommon.iOutFeaH = iIh[0];
		LayerCommon.cOutputChannel = iIc[0];

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
	unsigned long long DetectionOutput::fillDDRAddress(unsigned long long uiLastAddr, const char* num)
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
	int DetectionOutput::getInputScale(FILE *fileFp)
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
	void DetectionOutput::setQuantize(int iQuantize,float fIScale)
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
	void DetectionOutput::setFp32(int iFp32,unsigned long long uiBiasAddr,unsigned long long uiWScaleAddr,float fIScale)
	{

	}

	/*************************************************************************
	* Function Name : getNextInputOutputAddr
	* Description   : ge tNext Input Output Address
	* Parameters    : uiAddr -- input address
	* Returns       : uiNewAddr -- next layer input address
	**************************************************************************/
	unsigned long long DetectionOutput::getNextInputOutputAddr(unsigned long long uiAddr,unsigned int uiOneSeg,\
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
	unsigned long long DetectionOutput::setRegisterValue(unsigned long long uiLastAddr,unsigned int uiOneSeg,\
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
	int DetectionOutput::setRegisterValue(std::vector<unsigned long long> uiInputAddr, std::vector<unsigned long long> uiOutputAddr,const char* num)
	{
		DetectionOutputhead.mboxLocAddr = uiInputAddr[0];
		DetectionOutputhead.mboxConfAddr = uiInputAddr[1];
		return 0;
	}

	/*************************************************************************
	* Function Name : writeDDRInfoInputOutput
	* Description   : write input output ddr information to putput file
	* Parameters    : fileOutFp -- output file
	* Returns       : void
	**************************************************************************/
	void DetectionOutput::writeDDRInfoInputOutput(FILE *fileOutFp)
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
	int DetectionOutput::writeBinFile(FILE *fileInFp,FILE *fileOutFp,int iLayerIndex,const char* num)
	{
		char HEAD[sizeof(DetectionOutputhead)];
		int channelnumber = atoi(num);

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

		char cPrintBuf[PRINT_BUF_SIZE];
		unsigned int k = 0, rc = 0;

		//get weight
		FILE *Infile;
		char *path = "./models/ncnn2tm/concat.txt";
		if ( (Infile = fopen(path, "rt")) == NULL )
		{
			tmtool_log(LOG_ERROR, "Fail to open concat file!");
        	return -1;
		}
		struct stat statbuf;
    	stat(path,&statbuf);

    	int size = statbuf.st_size;
		char *pcCharBuf = (char *)malloc(size);
		float weight[size];
		int data_size = 0;
		
		while( fgets(pcCharBuf, size, Infile) != NULL ) {
			static int i = 0;
			weight[i] = atof(pcCharBuf);
			i++;
			data_size = i;
		}
		fclose(Infile);
		
		//read the tag
		rc = fread(&flag_struct, sizeof(flag_struct), 1,fileInFp);
		assert(rc > 0);
		//write weight
		// rc = fread(pcCharBuf, data_size, 1, fileInFp);
		// assert(rc > 0);

		int output_num = num_output;

		//add layerhead, size, address
		char cLayerHead[LAYER_HEADER_SIZE]={(char)0xDD, (char)0xCC, (char)0xBB, (char)0xAA, 05};
		DetectionOutputhead.InFeaW = LayerCommon.iInFeaW;//7668
		DetectionOutputhead.InFeaH = LayerCommon.iInFeaH;//2
		DetectionOutputhead.InputChannel = LayerCommon.cInputChannel;
		DetectionOutputhead.OutFeaW = LayerCommon.iOutFeaW;//random
		DetectionOutputhead.OutFeaH = LayerCommon.iOutFeaH;
		DetectionOutputhead.OutputChannel = LayerCommon.cOutputChannel;
		DetectionOutputhead.num_class = num_class;
        DetectionOutputhead.nms_threshold = nms_threshold;
        DetectionOutputhead.nms_top_k = nms_top_k;
        DetectionOutputhead.keep_top_k = keep_top_k;
        DetectionOutputhead.confidence_threshold = confidence_threshold;
        DetectionOutputhead.variances[0] = variances[0];
		DetectionOutputhead.variances[1] = variances[1];
		DetectionOutputhead.variances[2] = variances[2];
		DetectionOutputhead.variances[3] = variances[3];

		// DetectionOutputhead.FeaInAddr = convdwregisters.uiDWRegFeatureSrcAdd;
		// DetectionOutputhead.FeaOutAddr = convdwregisters.uiDWRegFeatureDstAdd;
		//DetectionOutputhead.LScale = Qn_a;
		DetectionOutputhead.PSPLFlag = WeightAddrFlag;
		DetectionOutputhead.DataSize = data_size;
		//convdwhead.DataAddr = convdwregisters.uiDWRegWeightSrcAdd;
		fwrite(cLayerHead,sizeof(char),sizeof(cLayerHead),fileOutFp);
		memcpy(HEAD,&DetectionOutputhead,sizeof(DetectionOutputhead));
		int DetectionheadLength = sizeof(DetectionOutputhead)/4;
		for (int i = 0; i < DetectionheadLength; i++)
		{
			for(int k=0; k< 4; k++)
			{
				fwrite(&HEAD[i*4+3-k],sizeof(char),1,fileOutFp);
			}
		}

		//write weight
		for (int j=0; j<data_size; j++)
		{
			memcpy(cPrintBuf,&weight[j],sizeof(cPrintBuf));
			for(int k=0; k< 4 ;k++)
			{
				fwrite(&cPrintBuf[3-k],sizeof(char),1,fileOutFp);
			}
		}
		free(pcCharBuf);
		return 0;
	}

	/*************************************************************************
	* Function Name : writeddrBinFile
	* Description   : write register value to bin file
	* Parameters    : fileRp -- out tmmodel bin file after the layer date		   
	* Returns       : 0 -- success
	**************************************************************************/
	int DetectionOutput::writeddrBinFile(FILE *fileRp)
	{
		return 0;
	}
}

