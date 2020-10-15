/*
 * layercommon.h
 *
 *  Created on: Jun 11, 2019
 *      Author: doyle
 */

#ifndef LAYERCOMMON_H_
#define LAYERCOMMON_H_

#include <stdio.h>
#include <string>
#include <vector>
#include <stdlib.h>
#include <string.h>

#define NDEBUG 1
#include <assert.h>

#include "cJSON.h"
#include "log.h"

#define CONFIG_JSON_FILE "reg_config.json"
#define TMMODEL_BIN	"tmmodel.bin"
#define TMDDR_BIN	"tmDDR.bin"
#define CONSTRUCT_TXT	"../models/caffe2tm/construct.txt"
#define DDR_TXT		"../models/caffe2tm/DDR_info.txt"

#define LAYER_INPUT	  0x00
#define INPUT_NAME    "Input"
#define LAYER_CONV    0x01
#define CONV_NAME	  "Convolution"
#define LAYER_CONVDW  0x02
#define CONVDW_NAME	  "ConvolutionDepthWise"
#define LAYER_PRELU   0x03
#define PRELU_NAME	  "PReLU"
#define LAYER_SPLIT   0x04
#define SPLIT_NAME	  "Split"
#define LAYER_ELTWISE 0x05
#define ELTWISE_NAME "Eltwise"
#define LAYER_BINARYOP 0x06
#define BINARYOP_NAME "BinaryOp"
#define LAYER_RELU 0x07
#define RELU_NAME "ReLU"
#define LAYER_POOLING 0x08
#define POOLING_NAME "Pooling"
#define LAYER_BATCHNORM 0x09
#define BATCHNORM_NAME "BatchNorm"
#define LAYER_REGION 0x0a
#define REGION_NAME "Region"
#define LAYER_REORG 0x0b
#define REORG_NAME "Reorg"
#define LAYER_CONCAT 0x0c
#define CONCAT_NAME "Concat"
#define LAYER_SCALE 0x0d	
#define SCALE_NAME "Scale"
#define LAYER_SOFTMAX 0x0e	
#define SOFTMAX_NAME "Softmax"
#define LAYER_RESHAPE 0x0f	
#define RESHAPE_NAME "Reshape"
#define LAYER_POOLING_CONVDW 0x10
#define POOLING_CONVDW_NAME "Pooling_ConvolutionDepthWise"
#define LAYER_PRIORBOX 0x11
#define PRIORBOX_NAME "PriorBox"
#define LAYER_DETECTIONOUTPUT 0x12
#define DETECTIONOUTPUT_NAME "DetectionOutput"
#define LAYER_PERMUTE 0x13
#define PERMUTE_NAME "Permute"
#define LAYER_FLATTEN 0x14
#define FLATTEN_NAME "Flatten"
#define LAYER_UPSAMPLE 0x15
#define UPSAMPLE_NAME "Upsample"

#define ENABLE	1
#define DISABLE	!ENABLE

#define ADD_32BIT	32
#define ADD_64BIT	64
#define FRAMSIZE	4096
#define WRAMSIZE	8192
#define CDPRAMSIZE  4096
#define LAYER_HEADER_SIZE	5
#define PRINT_BUF_SIZE	4
#define DATA_CHANNEL_NUM	8

#define ALGN_SIZE	0x1000
#define CONV_BUFF_SIZE	16384
#define CONV_WEIGHT_BUFF	1024
#define CONV_PADW_PADH_NO_DEFAULT (-233)
#define PRINT_BUF_SIZE	4

#define REG_KER_H_BIT	16
#define REG_FEA_H_BIT	16
#define REG_FEA_CO_BIT	16
#define REG_PAD_SZ_BIT	1
#define REG_PAD_ENABLE	1

#define RGB_CHANEL_NUM_SIZE             3 
#define DPU_INPUT_CHANEL_NUM_SIZE       4
#define CONV_OUT_BIAS_SCALE_IN_BIN      1
#define CONV_OUT_A_SCALE_IN_BIN         1

#define LAYER_OUT_PUT_BIN_ASCALE_BIAS_PRELU   3
#define LAYER_OUT_PUT_BIN_ASCALE_BIAS         2
#define FirstLayerChannel 4 

#define CONV_BUFF_SIZE	16384
#define CONV_WEIGHT_BUFF	1024
#define PRINT_BUF_SIZE	4

#define REG_KER_H_BIT	16
#define REG_FEA_H_BIT	16
#define REG_FEA_CO_BIT	16
#define REG_PAD_SZ_BIT	1
#define REG_PAD_ENABLE	1

#define DDR_4K	0x1000
#define DDR_1K	1024
#define MB_TO_BYTE	DDR_1K*DDR_1K
#define PL_NO_USE	16

#define STORED_PS	0
#define STORED_PL	1
#define STORED_PS_LAST_HALF	2

#define MAX_ADDRESS 0xffffffffffffffff
#define INPUT_LAYER_INNUM	3
#define PARAM_MAGIC	7767517
#define PRASE_BUFF	257

#define ENABLE			1
#define DISABLE			!ENABLE
/*EW block get max data from ddr*/
#define MAX_VALUE_ONCE	16384	 
#define SEG_NUM_PRAM	8

#define REG_POOL_SZ_BIT			1
#define REG_POOL_STRIDE_BIT		5
#define REG_POOL_PAD_BIT		9
#define REG_POOL_MODE_BIT		13
#define REG_POOL_QUAN_BIT		14

#define POOLING_PADING_MODE_FULL 0
#define POOLING_PADING_MODE_VALID 1
#define POOLING_PADING_MODE_SAME 2

#define SOFTMAX_OUTPUT_LENGTH 1000
#define OFFSETADD 0x100000

/*reset mode*/
#define CONCAT 0
#define REORG 1
#define PERMUTE 2
#define SPLIT 3
#define TRUE 1
#define FALSE 0
#define SOFTMAXSEGLENGTH 21 

namespace tmnet
{
	typedef struct
	{
		char cLayerType;
		std::vector<unsigned int> viInput;
		std::vector<unsigned int> viOutput;
		std::vector<std::string> bottomNames;
		std::vector<std::string> topNames;
		unsigned int iInFeaW;
		unsigned int iInFeaH;
		unsigned int cInputChannel;
		//input size in bytes
		unsigned int uiInputSize;

		unsigned int iOutFeaW;
		unsigned int iOutFeaH;
		unsigned int cOutputChannel;
		//output size in bytes
		unsigned int uiOutputSize;
	} LAYER_COMMON;

	typedef struct 
	{
		unsigned char pooling_size;
		unsigned char pooling_stride;
		unsigned char pooling_padding;
	}POOLING;

	typedef unsigned long long   uint64;
	typedef signed long		     sint32;

	class LayerCom
	{
	public:
		//construct function
		LayerCom();
		//deconstruct function
		~LayerCom();
		//layer input and output information
		LAYER_COMMON LayerCommon;
		POOLING PoolingConfig; 
		//load param file parameters
		virtual int loadParam(FILE* fileFp, int output_num) = 0;
		//calculate feature size
		virtual int calculateFeaSize(std::vector<unsigned int> iIw, std::vector<unsigned int> iIh, std::vector<unsigned int> iIc) = 0;
		//bin data ddr address
		virtual unsigned long long fillDDRAddress(unsigned long long uiLastAddr,const char* num) = 0;
		//get input scale
		virtual int getInputScale(FILE *fileFp) = 0;
		//set registers value
		virtual unsigned long long setRegisterValue(unsigned long long uiLastAddr,unsigned int uiOneSeg,\
													const unsigned int uiOriAddr,char cBufferNum) = 0;
		virtual int setRegisterValue(std::vector<unsigned long long> uiInputAddr, std::vector<unsigned long long> uiOutputAddr,const char* num) = 0;
		//set qunaitze value
		virtual void setQuantize(int iQuantize,float fIScale);
		//set fp32 value
		virtual void setFp32(int iFp32,unsigned long long uiBiasAddr,unsigned long long uiWScaleAddr,float fIScale);
		//get fp32 translate information
		virtual int getFp32Infor(unsigned long long *uiBias,unsigned long long *uiWScale,float *fIScale);
		//get fp32 translate information
		virtual int getQuantizeInfor(float *fIScale);
		//write bin file
		// virtual int writeBinFile(int iDropCount,FILE *fileInFp,FILE *fileOutFp,int iLayerIndex,char cBit) = 0;
		virtual int writeBinFile(FILE *fileInFp,FILE *fileOutFp,int iLayerIndex,const char* num) = 0;
		//write weight data to DDR_info.txt
		virtual void writeDDRInfoWeight(FILE *fileOutFp);
		
		virtual int writeddrBinFile(FILE *fileRp) = 0;
		//write input output data to DDR_info.txt
		virtual void writeDDRInfoInputOutput(FILE *fileOutFp) = 0;
		// layer type name
		int type;
		// layer name
		std::string name;

		//first flag, setting only once
		bool cFirstFlag;

		bool outputFlag;

		bool cCloseQnFlag;

		bool fp16_enable;

		bool cUpsampleFlag;

		bool cPoolFlag;

		bool reluWriteBinFlag;

		bool cPreluFlag;

		bool cReLUFlag;
		
		bool cEltwiseReLUFlag;

		bool cPoolingFlag;

		bool cInstructionFlag;

		bool getDataFromSplit;

		unsigned char poolingsize;
		
		unsigned char poolingstride;
		
		unsigned char poolingpadding;

		//run on cpu flag, unused flag
		bool cCpuFlag;

		//data size in bin file in bytes
		unsigned int uiBinDataSize;

		//need to be run
		char cRunFlag;

		char cConcatFlag;

		unsigned long long concatLayerAddrBuf;
		
		float Qn_a;

		float inputScale;
		
		float lscale;

		float ew1BnA;

		float ew2BnA;

		// float Qn_a;
		int GLB_NUM;

		int WeightAddrFlag;
	    bool cSoftMaxFlag;
		std::vector<int> vInOutLayer;

		std::vector<int> splitOutputToEW;

		std::vector<int> splitOutputToConv;

        unsigned long long uiLayerOutputAddr;
	};
}
#endif /* LAYERCOMMON_H_ */
