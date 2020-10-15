/*
 * network.h
 *
 *  Created on: Jun 12, 2019
 *      Author: doyle
 */

#ifndef NETWORK_H_
#define NETWORK_H_

#include "layercommon.h"
#include "Convolution.h"
#include "ConvolutionDepthWise.h"
#include "Input.h"
#include "PReLU.h"
#include "Split.h"
#include "Eltwise.h"
#include "BinaryOp.h"
#include "Pooling.h"
#include "ReLU.h"
#include "Concat.h"
#include "BatchNorm.h"
#include "Region.h"
#include "Reorg.h"
#include "Scale.h"
#include "Softmax.h"
#include "Reshape.h"
#include "pConvolutionDepthWise.h"
#include "PriorBox.h"
#include "DetectionOutput.h"
#include "Permute.h"
#include "Flatten.h"
#include "Upsample.h"
#include "boardconfig.h"

namespace tmnet
{
	typedef struct
	{ 
		std::vector<std::string> blobName;
		std::vector<std::string> topBlobName;
		unsigned int	fromLayer;
		std::vector<unsigned int> forLayer;
	} BLOBS;

	class Network : public BoardConfig
	{
	public:
		Network();
		~Network();
		//layer param
		std::vector<LayerCom*> vLayers;
		//blobs
		std::vector<BLOBS> blobs;
		//load param file
		int	loadParamFile(const char* protopath);
		//load bin file
		int	loadBinFile(const char* protopath, const char* num);
		//create layer by name, if pooling = 0 as pooling layer, else pooling = 1, as ConvolutionDepthWise layer
		LayerCom *createLayer(const char* type, int pooling); 
		//write blobs
		int writeOneBlobsFromLayer(std::vector<std::string> blobName,int fromLayerIndex);
		int writeOneBlobsForLayer(std::vector<std::string> blobName,int currentLayer);

	    //merge Middle Layer
	    void mergeMiddleLayer(void);

		//get structure
		void fillLayerStructInOut(void);
		int setUpsampleFlag(int layerNum);
		int getConcatInputFromInstructionLayer(int layerNum);
		int getConcatInputLayer(int layerNum);
		//get feature size
		void calculateAllFeatureSize(void);
		void calculateLayerRelationship(void);
		//bin data address buff
		int fillAllDDRAddress(const char* num);
		//load all input scale
		void loadInstructionParam(FILE *fileFp);
		//set quantize and int32->fp32 value
		int loadConvInputScale(int layerNum);
		int setFp16EnableFlag(int layerNum);
		void setQuanFp32Value(void);
		//set all the register
		void setAllRegister(const char* num);
		//write whole config file
		void writeWholeCjson(void);
		//write the whole bin file
		void writeWholeBinFile(FILE *fileFp,const char* num);
		//set first flag
		void setEnableFlagForInstructionLayers(void);
		void setOutputFlag(void);
		void classifySplitOutputLayers(void);
		//print net construt and feature size
		void getConcatInputAddr(int outAddrCount);
		void printConstruct(void);
		// recursize match instruction layer from bottom to top
		int recursiveMatchLayer();

		int dropCount;

		//get PS and PL DDR address by config file
		int getBoardInformation(const char* protopath,const char* inBoardName);

		//get the biggest feature size
		void getMaxFeatureSize(void);

		//printf all ddr address information
		void printAllDDRInfo(void);

		unsigned int maxFeature;
		//one seg of input output data size
		unsigned int oneSegInOut;

		//board type index
		int boardTypeIndex;
		//bin file data address
		unsigned long long  binDataAddr;
		//input output data address
		unsigned long long inOutDataAddr;
		//ps max address(half of the ps ddr)
		unsigned long long  psMaxAddr;
		//pl max address (will not use the last 16M of the PL ddr)
		unsigned long long plMaxAddr;

		//get all the inout address
		void getAllBufferAddress(unsigned long long startAddress,char curPlace);
		std::vector<unsigned long long> inOutAddr;

		//ps overflow flag
		bool psOverflowFlag;
		//pl overflow flag
		bool plOverflowFlag;
		std::vector<int> concatFeatureSize;
		std::vector<int> concatInstructionLayer;
		std::vector<float> inputScaleCache;
		std::vector<int>  outputGiveToCpu;
	};
}
#endif /* NETWORK_H_ */

