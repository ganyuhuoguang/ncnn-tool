/*
 * network.cpp
 *
 *  Created on: Jun 12, 2019
 *      Author: doyle
 */

#include "network.h"

#include "boardconfig_data.h"
#include "cJSON.h"
#define NDEBUG 1
#include <assert.h>

#include "platform.h"
#include <unistd.h>
#include <iostream>
#include <vector> 
#include <algorithm>
#include <limits.h>

namespace tmnet
{
	using std::max;
	using std::string;
	
	const char* pcLayerName[] = {INPUT_NAME,
								CONV_NAME,
								CONVDW_NAME,
								PRELU_NAME,
								SPLIT_NAME,
								ELTWISE_NAME,
								BINARYOP_NAME,
								RELU_NAME,
								POOLING_NAME,
								BATCHNORM_NAME,
								REGION_NAME,
								REORG_NAME,
								CONCAT_NAME,
								SCALE_NAME,
								SOFTMAX_NAME,
								RESHAPE_NAME,
								POOLING_CONVDW_NAME,
								PRIORBOX_NAME,
								DETECTIONOUTPUT_NAME,
								PERMUTE_NAME,
								FLATTEN_NAME,
								UPSAMPLE_NAME,
	};
							
	static const int supportLayer = sizeof(pcLayerName) / sizeof(char *);

	/*************************************************************************
	* Function Name : Network
	* Description   : construct function ,initialize all the variable
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	Network::Network()
	{
		binDataAddr = 0;
		inOutDataAddr = 0;
		dropCount = 0;
		boardTypeIndex = -1;
		maxFeature = 0;
		oneSegInOut = 0;
		//ps overflow flag
		psOverflowFlag = false;
		//pl overflow flag
		plOverflowFlag = false;

		plMaxAddr = 0;
		psMaxAddr = 0;
	}

	/*************************************************************************
	* Function Name : ~Network
	* Description   : deconstruct function
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	Network::~Network()
	{

	}

	/*************************************************************************
	* Function Name : fillAllDDRAddress
	* Description   : fill All bin data DDR Address
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	int Network::fillAllDDRAddress(const char* num)
	{	
		if (boardTypeIndex < 0)
		{
			tmtool_log(LOG_ERROR, "Board Information Error!");
			return -1;
		}
		int iResult = 0;
		int layer_count = vLayers.size();
		LayerCom* Layer = NULL;
		unsigned long long ullMaxAddrBuff = 0;
		BoardData* pBoard = NULL;
		pBoard = vBoardData[boardTypeIndex];

		ullMaxAddrBuff = (vBoardData[boardTypeIndex]->cWeightPlace?plMaxAddr:psMaxAddr);

		//check if the PL size is 0
		if ((vBoardData[boardTypeIndex]->cWeightPlace == STORED_PL) && (pBoard->uiPlSize == 0))
		{
			ullMaxAddrBuff = psMaxAddr;
		}
		else if ((vBoardData[boardTypeIndex]->cWeightPlace == STORED_PS) && (pBoard->uiPsSize == 0))
		{
			ullMaxAddrBuff = plMaxAddr;
		}

		for(int i=0; i<layer_count; i++)
		{
			Layer = vLayers[i];
			if(!(plOverflowFlag && psOverflowFlag))
			{
				//check if this layer data is out range
				if((binDataAddr + Layer->uiBinDataSize) >= ullMaxAddrBuff)
				{
					if(vBoardData[boardTypeIndex]->cWeightPlace == STORED_PL)
					{
						plOverflowFlag = true;
						if(pBoard->uiPsSize == 0)
						{
							//not enough space ERROR
							iResult = -1;
							break;
						}
						else
						{
							binDataAddr = vBoardData[boardTypeIndex]->ullPsStart;
							ullMaxAddrBuff = psMaxAddr;
						}
					}
					else
					{
						psOverflowFlag = true;
						if(pBoard->uiPlSize == 0)
						{
							//no PL DDR
							plOverflowFlag = true;
						}
						else
						{
							binDataAddr = vBoardData[boardTypeIndex]->ullPlStart;
							ullMaxAddrBuff = plMaxAddr;
						}
					}
				}

				if(plOverflowFlag && psOverflowFlag)
				{
					//write to PS the other half of the ddr
					binDataAddr = psMaxAddr;
					tmtool_log(LOG_COMMON, "using over half the ps ddr!");
				}
			}
			binDataAddr = Layer->fillDDRAddress(binDataAddr,num);
		}
		if(iResult < 0)
		{
			tmtool_log(LOG_ERROR, "Not Enough Space Error!");
		}
		return iResult;
	}

	/*************************************************************************
	* Function Name : fillLayerStructInOut
	* Description   : fill All Struct In Out
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	void Network::fillLayerStructInOut(void)
	{
		int iBlobCount = blobs.size();
		BLOBS *pBlob = NULL;
		for(int i=0; i<iBlobCount; i++)
		{
			pBlob = &blobs[i];
	        vLayers[pBlob->fromLayer]->LayerCommon.viInput.push_back(pBlob->blobName.size());
			vLayers[pBlob->fromLayer]->LayerCommon.viOutput.push_back(pBlob->topBlobName.size());
			int iBlobNameCount = pBlob->blobName.size(); 
		    for(int j = 0; j < iBlobNameCount; j++)
			{
				std::string tmpBlobName = pBlob->blobName[j];
				vLayers[pBlob->fromLayer]->LayerCommon.bottomNames.push_back(tmpBlobName);
				std::string bottomBlobNames = vLayers[pBlob->fromLayer]->LayerCommon.bottomNames[j];
			}
			int iTopBlobNameCount = pBlob->topBlobName.size(); 
			for(int j = 0; j < iTopBlobNameCount; j++)
			{
				std::string tmpTopBlobName = pBlob->topBlobName[j];
				vLayers[pBlob->fromLayer]->LayerCommon.topNames.push_back(tmpTopBlobName);
				std::string topBlobName = vLayers[pBlob->fromLayer]->LayerCommon.topNames[j];
			}
		}
	}

	/*************************************************************************
	* Function Name : writeOneBlobsFromLayer
	* Description   : write One Blobs FromLayer
	* Parameters    : blobName -- blob name
	* 				  fromLayerIndex -- the index of this blob in param file
	* Returns       : 0 --success
	**************************************************************************/
	int Network::writeOneBlobsFromLayer(std::vector<std::string> blobName, int fromLayerIndex)
	{
		int iBlobNameCount = blobName.size(); 
		BLOBS newBlob;
	    for (int i = 0; i < iBlobNameCount; i++)
		{	
			blobs[fromLayerIndex].topBlobName.push_back(blobName[i]);
		}

		blobs[fromLayerIndex].fromLayer = fromLayerIndex;
		return 0;
	}

	/*************************************************************************
	* Function Name : writeOneBlobsForLayer
	* Description   : write One Blobs for Layer
	* Parameters    : blobName -- blob name
	* 				  fromLayerIndex -- the index of this blob in param file
	* Returns       : 0 --success
	**************************************************************************/
	int Network::writeOneBlobsForLayer(std::vector<std::string> blobName,int currentLayer)
	{
		int iBlobNameCount = blobName.size(); 
		u_int32_t LayerIndex = 0xFFFFFFFF;
   
	    for (int i = 0; i < iBlobNameCount; i++)
		{
			blobs[currentLayer].blobName.push_back(blobName[i]);
		}
	    LayerIndex = currentLayer;
		//blob not exist
		if (LayerIndex == 0xFFFFFFFF)
		{
			return -1;
		}
		//the input blobname already exist
		blobs[LayerIndex].forLayer.push_back(currentLayer);  
		return 0;
	}

	/*************************************************************************
	* Function Name : calculateAllFeatureSize
	* Description   : calculate All Feature Size
	* Parameters    : NULL
	* Returns       : NULL
	**************************************************************************/
	void Network::calculateAllFeatureSize(void)
	{
		int layer_count = vLayers.size();
		LayerCom* Layer = NULL;
		LayerCom* LastLayer = NULL;
		LayerCom* PrintLayer = NULL;
		bool compare_ok = false;
		std::vector<unsigned int> iW;
		std::vector<unsigned int> iH;
		std::vector<unsigned int> iC;	
		for (int i = 0; i<layer_count; i++)
		{
			Layer = vLayers[i];
			compare_ok = false;
			int iLayerInputCount = Layer->LayerCommon.viInput[0];

			iW.clear();
			iH.clear();
			iC.clear();

			if (0 == i)
			{
				iW.push_back(0);
				iH.push_back(0);
				iC.push_back(INPUT_LAYER_INNUM);
				Layer->calculateFeaSize(iW,iH,iC);
			}
			else
			{

				for(int t = 0; t < iLayerInputCount; t++)
				{	
					for(int j = 0; j < i; j++)
					{
						LastLayer = vLayers[i - j - 1];
						int iLayerOutputCount = LastLayer->LayerCommon.viOutput.size();
						if (iLayerOutputCount > 0)
						{
							int iLastLayerOutputCount = LastLayer->LayerCommon.viOutput[0];
		
							for(int k = 0; k < iLastLayerOutputCount; k++)
							{
								if(!Layer->LayerCommon.bottomNames[t].compare(LastLayer->LayerCommon.topNames[k]))
								{
									iW.push_back(LastLayer->LayerCommon.iOutFeaW);
									iH.push_back(LastLayer->LayerCommon.iOutFeaH);
									iC.push_back(LastLayer->LayerCommon.cOutputChannel);

									compare_ok = true;
									break;
								}
							}
							if(compare_ok)
							{
								compare_ok = false;
								break;
							}
						}
					}
				}
				Layer->calculateFeaSize(iW,iH,iC);
			}
		}
		getMaxFeatureSize();
	}

	/*************************************************************************
	* Function Name : calculateLayerRelationship
	* Description   : Get input layers id of each layer
	* Parameters    : void
	* Returns       : void
	**************************************************************************/

	void Network::calculateLayerRelationship(void)
	{
		int layer_count = vLayers.size();   // layers store all layers parameter
		LayerCom* Layer = NULL;      // current layer
		LayerCom* preLayer = NULL;   // last layer
		bool compare_ok = false;
		for(int i = 0; i < layer_count; i++)
		{
			Layer = vLayers[i];
			compare_ok = false;
			int iLayerInputCount = Layer->LayerCommon.viInput[0];
			int iLayerOutputCount = Layer->LayerCommon.viOutput[0];
	        if (i == 0)
			{
				int inLayerNum = i;
				int outLayerNum = i;
				Layer->vInOutLayer.push_back(inLayerNum);	//ignore first layer
				Layer->vInOutLayer.push_back(outLayerNum);
			}
			else 
			{
				if (iLayerInputCount >= 1 && iLayerOutputCount >= 1)
				{
					for(int l = 0; l < iLayerInputCount; l++)
					{
						for(int j = 0; j < i; j++)
						{
							preLayer = vLayers[i - j - 1];
							int iLastLayerOutputSize = preLayer->LayerCommon.viOutput.size();
							if (iLastLayerOutputSize > 0)
							{
								int iLastLayerOutputCount = preLayer->LayerCommon.viOutput[0];
								for(int k = 0; k < iLastLayerOutputCount; k++)
								{
									if(!Layer->LayerCommon.bottomNames[l].compare(preLayer->LayerCommon.topNames[k]))
									{
										compare_ok = true;
										break;
									}
								}
								if (compare_ok)
								{
									int inLayerNum = i - j - 1;
									Layer->vInOutLayer.push_back(inLayerNum);
									compare_ok = false;
									break;
								}
							}
						}
					}
					int iLayerOutputCount = Layer->LayerCommon.viOutput[0];
					for (int l = 0; l < iLayerOutputCount; l++)
					{
		                int outLayerNum = i;
						Layer->vInOutLayer.push_back(outLayerNum);
					}
				}
			}
		}
	}


	/*************************************************************************
	* Function Name : loadInstructionParam
	* Description   : Get what parameters instruction layers need from other layers 
	* Parameters    : fileFp -- ncnn bin file
	* Returns       : NULL
	**************************************************************************/
	void Network::loadInstructionParam(FILE *fileFp)
	{
		int convLayerNum = 0;
		int ewInputNum = 0;
		int ewInputLayer = 0;
		int num = 0;
		int pooling_num = 0;
		int size_num = 0;
		int glb_num = 0;
		int layer_count = vLayers.size();
		LayerCom* Layer = NULL;
		assert(layer_count > 0);
		inputScaleCache.clear();
		for(int i = 0; i < layer_count; i++)
		{
			Layer = vLayers[i];
			Layer->fp16_enable = false;
			if ((Layer->type == LAYER_CONV) || (Layer->type == LAYER_CONVDW) || (Layer->type == LAYER_POOLING_CONVDW))
			{
				num++;
			}

			if (Layer->type == LAYER_POOLING)
			{
				pooling_num++;
			}
		
			if ((Layer->type == LAYER_CONV) || (Layer->type == LAYER_CONVDW) || (Layer->type == LAYER_ELTWISE) || (Layer->type == LAYER_PERMUTE))
			{
				glb_num++;
			}

			if (Layer->type == LAYER_SPLIT && Layer->cRunFlag == 1)
			{
				glb_num++;
			}
		}

		float qn[layer_count];
		int PoolingSize   [pooling_num];
		int PoolingStride [pooling_num];
		int PoolingPad    [pooling_num];
	    memset(&qn, 0, sizeof(qn));
		for(int i = 0; i < layer_count; i++) 
		{
			Layer = vLayers[i];
			Layer->getInputScale(fileFp);
		
			if((Layer->type == LAYER_CONV) || (Layer->type == LAYER_CONVDW) || (Layer->type == LAYER_POOLING_CONVDW))
			{
				qn[i] = Layer->Qn_a;
			}
			else
			{
				qn[i] = 0.0;
				Layer->Qn_a = 0.0;
			}
        
			Layer->GLB_NUM = glb_num;

			if (Layer->type == LAYER_POOLING)
			{
				PoolingSize[size_num] = Layer->PoolingConfig.pooling_size;
				PoolingStride[size_num] = Layer->PoolingConfig.pooling_stride;
				PoolingPad[size_num] = Layer->PoolingConfig.pooling_padding;
				size_num++;
			}
		}

		size_num = 0;
		for(int i = 0; i < layer_count; i++)
		{ 
			Layer = vLayers[i];
			Layer->poolingsize = 0;
			Layer->poolingstride = 0;
			Layer->poolingpadding = 0;
			if (Layer->cPoolingFlag == 1)
			{
				Layer->poolingsize = PoolingSize[size_num];
				Layer->poolingstride = PoolingStride[size_num];
				Layer->poolingpadding = PoolingPad[size_num];
				size_num++;
			}
			else
			{
				Layer->poolingsize = 0;
				Layer->poolingstride = 1;
				Layer->poolingpadding = 0;
			}
            
			
			if (Layer->cCloseQnFlag == false)
			{
				if (Layer->type == LAYER_CONV || Layer->type == LAYER_CONVDW || Layer->type == LAYER_ELTWISE)
				{
					int convLayer = i;
					loadConvInputScale(convLayer);
					Layer->Qn_a = inputScaleCache[convLayerNum];
					convLayerNum++;
				}
			}
			else
			{
				Layer->Qn_a = 0;
			}
		}

		inputScaleCache.clear();
		for(int i = 0; i < layer_count; i++)
		{
			Layer = vLayers[i];
			if (Layer->type == LAYER_ELTWISE)
			{
				int iLastLayerInputCount = Layer->LayerCommon.viInput[0];
				for (int j = 0; j < iLastLayerInputCount; j++)
				{
					int ewInputLayer = Layer->vInOutLayer[j];
					loadConvInputScale(ewInputLayer);
					setFp16EnableFlag(ewInputLayer);
				}
				Layer->ew1BnA = inputScaleCache[ewInputNum++];
				Layer->ew2BnA = inputScaleCache[ewInputNum++];
			}
		}
	}




	/*************************************************************************
	* Function Name : setFp16EnableFlag
	* Description   : mark the instruction layer which need to make Fp16 enable 
	* Parameters    : layerNum -- input layers id of current layer  
	* Returns       : 0 --success
	**************************************************************************/
	int Network::setFp16EnableFlag(int layerNum)
	{
		LayerCom* Layer = NULL;
		LayerCom* lastLayer = NULL;
		int layerIndex = layerNum;
		Layer = vLayers[layerIndex];
		if (Layer->cInstructionFlag == true)
		{
			Layer->fp16_enable = true;
			Layer->cCloseQnFlag = true;
			return 0;
		}
		else
		{
			for (int i = layerIndex - 1; i >= 0; i--)
			{
				lastLayer = vLayers[i];
				int iLastLayerOutputSize = lastLayer->LayerCommon.viOutput.size();
				if (iLastLayerOutputSize > 0)
					{
						int iLastLayerInputCount = lastLayer->LayerCommon.viOutput[0];
						for(int k = 0; k < iLastLayerInputCount; k++)
						{
							/*current layer output compare to following input to find next input layer*/
							if(!Layer->LayerCommon.bottomNames[0].compare(lastLayer->LayerCommon.topNames[k])) 
							{
								layerIndex = i;							
								if (lastLayer->cInstructionFlag == true)
								{
									lastLayer->fp16_enable = true;
									return 0;
								}
								break;
							}
						}
					}
			}
		}
		/*if reach the condition end the loop*/
		setFp16EnableFlag(layerIndex);
		return -1;
	}


	/*************************************************************************
	* Function Name : loadConvInputScale
	* Description   : store Qn_a paremeter of Convolution in order
	* Parameters    : layerNum -- input layers id of current layer  
	* Returns       : 0 --success
	**************************************************************************/

	int Network::loadConvInputScale(int layerNum)
	{
		int layerIndex = layerNum;	
		int layerCount = vLayers.size();
		bool compare_ok = false;
		LayerCom* Layer = NULL;
		LayerCom* nextLayer = NULL;
		Layer = vLayers[layerIndex];
		if (Layer->type == LAYER_SPLIT)
		{
			int LayerOutputSize = Layer->LayerCommon.viOutput[0];
			for (int n = 0; n < LayerOutputSize; n++)
			{
				for (int j = layerIndex + 1; j < layerCount; j++)
				{
					nextLayer = vLayers[j];
					int iLastLayerOutputSize = nextLayer->LayerCommon.viInput.size();
					if (iLastLayerOutputSize > 0)
					{
						int iLastLayerInputCount = nextLayer->LayerCommon.viInput[0];
						for(int k = 0; k < iLastLayerInputCount; k++)
						{
							if(!Layer->LayerCommon.topNames[n].compare(nextLayer->LayerCommon.bottomNames[k]))
							{
								if(nextLayer->type == LAYER_CONV || nextLayer->type == LAYER_CONVDW)
								{
									compare_ok = true;
									break;
								}
							}
						}
						if (compare_ok)
						{
							layerIndex = j;
							compare_ok = false;
							break;
						}
					}
				}
			}
		}
		else
		{
			for(int j = layerIndex + 1; j < layerCount; j++)
        	{
				nextLayer = vLayers[j];
				int i = nextLayer->LayerCommon.viOutput.size();
				if (i > 0)
				{
					int iNextLayerInputCount = nextLayer->LayerCommon.viInput[0];
					for(int k = 0; k < iNextLayerInputCount; k++)
					{
						/*current layer output compare to following input to find next input layer*/
						if(!Layer->LayerCommon.topNames[0].compare(nextLayer->LayerCommon.bottomNames[k])) 
						{
							compare_ok = true;
							break;
						}
					}
					if (compare_ok)
					{
						layerIndex = j;
						compare_ok = false;
						break;
					}
				}
        	}
		}

		Layer = vLayers[layerIndex];
		if (Layer->type == LAYER_CONV || Layer->type ==LAYER_CONVDW)
		{
			inputScaleCache.push_back(Layer->inputScale);
			return 0;
		}
		/*if reach the condition end the loop*/
		loadConvInputScale(layerIndex);
		return -1;
	}

	/*************************************************************************
	* Function Name : setQuanFp32Value
	* Description   : set all the Quantize and Fp32 Value
	* Parameters    : NULL
	* Returns       : NULL
	**************************************************************************/
	void Network::setQuanFp32Value(void)
	{
	// 	int layer_count = layers.size();
	// 	LayerCom* Layer = NULL;
	// 	LayerCom* NearLayer = NULL;

	// 	for(int i=1; i<layer_count; i++)
	// 	{
	// 		Layer = layers[i];
	// 		NearLayer = layers[i-1];
	// 		//int32 -> fp32
	// 		if(Layer->type == LAYER_RELU && NearLayer->type == LAYER_CONV)
	// 		{
	// 			unsigned long long uiBiasAdd = 0;
	// 			unsigned long long uiWScaleAdd = 0;
	// 			float fScale = 0;
	// 			NearLayer->getFp32Infor(&uiBiasAdd,&uiWScaleAdd,&fScale);
	// //			printf("%d fp32 badd %llx wscale %llx iscale %f\n",i,uiBiasAdd,uiWScaleAdd,fScale);
	// 			Layer->setFp32(1,uiBiasAdd,uiWScaleAdd,fScale);
	// 		}

	// 		//set quantize
	// 		NearLayer = layers[i+1];
	// 		if((Layer->type == LAYER_RELU || Layer->type == LAYER_POOL) && NearLayer->type == LAYER_CONV)
	// 		{
	// 			float fScale = 0;
	// 			NearLayer->getQuantizeInfor(&fScale);
	// //			printf("%d quantize iscale %f\n",i,fScale);
	// 			Layer->setQuantize(1,fScale);
	// 		}
	// 	}
	}

	/*************************************************************************
	* Function Name : getAllBufferAddress
	* Description   : calculate all the buffer address
	* Parameters    : startAddress -- start address
	*                 curPlace -- write at PS or PL
	* Returns       : void
	**************************************************************************/
	void Network::getAllBufferAddress(unsigned long long startAddress,char curPlace)
	{
		unsigned long long ullAddrBuf = startAddress;
		char cPlaceBuf = curPlace;

		for(int i=0; i<vBoardData[boardTypeIndex]->cBufferNum; i++)
		{
			if (cPlaceBuf == STORED_PL)
			{
				if ((ullAddrBuf + oneSegInOut) > plMaxAddr)
				{
					if (vBoardData[boardTypeIndex]->uiPsSize == 0)
					{
						//no PS then return
						tmtool_log(LOG_ERROR, "not enough buffer space!");
						return;
					}
					plOverflowFlag = true;
					if (psOverflowFlag)
					{
						cPlaceBuf = STORED_PS_LAST_HALF;
						ullAddrBuf = psMaxAddr;
						inOutAddr.push_back(ullAddrBuf);
						ullAddrBuf += oneSegInOut;
					}
					else
					{
						if (vBoardData[boardTypeIndex]->cWeightPlace == STORED_PS)
						{
							//weight is write in PS
							ullAddrBuf = binDataAddr+4*ALGN_SIZE;
							cPlaceBuf = STORED_PS;
							inOutAddr.push_back(ullAddrBuf);
							ullAddrBuf += oneSegInOut;
						}
						else
						{
							//weight is write in PL
							ullAddrBuf = vBoardData[boardTypeIndex]->ullPsStart;
							cPlaceBuf = STORED_PS;
							inOutAddr.push_back(ullAddrBuf);
							ullAddrBuf += oneSegInOut;
						}
					}
				}
				else
				{
					inOutAddr.push_back(ullAddrBuf);
					ullAddrBuf += oneSegInOut;
				}
			}
			else if (cPlaceBuf == STORED_PS)
			{
				if ((ullAddrBuf + oneSegInOut) > psMaxAddr)
				{
					psOverflowFlag = true;
					if (plOverflowFlag || vBoardData[boardTypeIndex]->uiPlSize == 0)
					{
						if (vBoardData[boardTypeIndex]->uiPlSize == 0)
						{
							plOverflowFlag = true;
						}
						cPlaceBuf = STORED_PS_LAST_HALF;
						ullAddrBuf = psMaxAddr;
						inOutAddr.push_back(ullAddrBuf);
						ullAddrBuf += oneSegInOut;
					}
					else
					{
						if (vBoardData[boardTypeIndex]->cWeightPlace == STORED_PS)
						{
							//weight is write in PS
							ullAddrBuf = binDataAddr+4*ALGN_SIZE;
							cPlaceBuf = STORED_PL;
							inOutAddr.push_back(ullAddrBuf);
							ullAddrBuf += oneSegInOut;
						}
						else
						{
							//weight is write in PL
							ullAddrBuf = vBoardData[boardTypeIndex]->ullPlStart;
							cPlaceBuf = STORED_PL;  
							inOutAddr.push_back(ullAddrBuf);
							ullAddrBuf += oneSegInOut;
						}
					}
				}
				else
				{
					inOutAddr.push_back(ullAddrBuf);
					ullAddrBuf += oneSegInOut;
				}
			}
			else if (cPlaceBuf == STORED_PS_LAST_HALF)
			{
				inOutAddr.push_back(ullAddrBuf);
				ullAddrBuf += oneSegInOut;
			}
		}
	}

	/*************************************************************************
	* Function Name : setAllRegister
	* Description   : set all register value
	* Parameters    : NULL
	* Returns       : NULL
	**************************************************************************/
	void Network::setAllRegister(const char* num)
	{
		int layer_count = vLayers.size();
		LayerCom* Layer = NULL;
		unsigned long long ullMaxAddrBuff = 0;
		//0-write to ps 1-write to pl 2-last half of ps
		char curPlace = 0; 
		int iIndex = 0;
		std::vector<unsigned long long> ullInAddr;
		std::vector<int> vInLayerCount;
		std::vector<unsigned long long> ullOutAddr;

		for(int i=0; i<layer_count; i++)
		{
			Layer = vLayers[i];	
			Layer->WeightAddrFlag = vBoardData[boardTypeIndex]->cWeightPlace;
		}

		if (vBoardData[boardTypeIndex]->cDataPlace == vBoardData[boardTypeIndex]->cWeightPlace)
		{
			inOutDataAddr = binDataAddr+4*ALGN_SIZE;
			if(plOverflowFlag && vBoardData[boardTypeIndex]->cWeightPlace)
			{
				//weight is overflow at PL DDR
				if(!psOverflowFlag)
				{
					//PS is not overflow
					ullMaxAddrBuff = psMaxAddr;
					curPlace = STORED_PS;
				}
				else
				{
					curPlace = STORED_PS_LAST_HALF;
				}
			}
			else if(psOverflowFlag && (vBoardData[boardTypeIndex]->cWeightPlace == STORED_PS))
			{
				//weight is overflow at PS DDR
				if((!plOverflowFlag) && (vBoardData[boardTypeIndex]->uiPlSize != 0))
				{
					//PL is not overflow
					if(vBoardData[boardTypeIndex]->uiPlSize != 0)
					{
						ullMaxAddrBuff = plMaxAddr;
						curPlace = STORED_PL;
					}
					else
					{
						//there is not PL DDR then write to PS second half
						ullMaxAddrBuff = MAX_ADDRESS;
						curPlace = STORED_PS_LAST_HALF;
					}
				}
				else
				{
					curPlace = STORED_PS_LAST_HALF;
				}
			}
			else
			{
				//weight is not overflow
				ullMaxAddrBuff = (vBoardData[boardTypeIndex]->cDataPlace?plMaxAddr:psMaxAddr);
				if(vBoardData[boardTypeIndex]->uiPsSize != 0)
				{
					curPlace = vBoardData[boardTypeIndex]->cWeightPlace;
				}
				else
				{
					curPlace = STORED_PL;
				}
			}
		}
		else
		{
			if (!(psOverflowFlag && plOverflowFlag))
			{
				//weight on the different side and weight did not overflow the other side
				if ((vBoardData[boardTypeIndex]->uiPlSize != 0) && (vBoardData[boardTypeIndex]->uiPsSize != 0))
				{
					inOutDataAddr = (vBoardData[boardTypeIndex]->cDataPlace?vBoardData[boardTypeIndex]->ullPlStart:vBoardData[boardTypeIndex]->ullPsStart);
					ullMaxAddrBuff = (vBoardData[boardTypeIndex]->cDataPlace?plMaxAddr:psMaxAddr);
					curPlace = vBoardData[boardTypeIndex]->cDataPlace;
				}
				else
				{
					inOutDataAddr = binDataAddr+4*ALGN_SIZE;
					ullMaxAddrBuff = (vBoardData[boardTypeIndex]->cWeightPlace?plMaxAddr:psMaxAddr);
					curPlace  = vBoardData[boardTypeIndex]->cWeightPlace;
				}
			}
			else
			{
				//weight is overflow on the other side
				if (psOverflowFlag && plOverflowFlag)
				{
					//weight use all ddr usable space
					curPlace = STORED_PS_LAST_HALF;
				}
				else if (psOverflowFlag)
				{
					//weight is overflow at PS DDR and PL is still available
					if (vBoardData[boardTypeIndex]->uiPlSize != 0)
					{
						ullMaxAddrBuff = plMaxAddr;
						curPlace = STORED_PL;
					}
					else
					{
						curPlace = STORED_PS_LAST_HALF;
					}
				}
				else if (plOverflowFlag)
				{
					//weight is overflow at PL DDR and PS is still available

					if (vBoardData[boardTypeIndex]->uiPsSize != 0)
					{
						ullMaxAddrBuff = psMaxAddr;
						curPlace = STORED_PS;
					}
					else
					{
						tmtool_log(LOG_ERROR, "Not Enough Space When Writing Data Address!");
						return;
					}
				}
				inOutDataAddr = binDataAddr+4*ALGN_SIZE;
			}
		}

		//save original information
		getAllBufferAddress(inOutDataAddr, curPlace);
		ullOutAddr.push_back(inOutAddr[iIndex]);
		int outAddrCount = inOutAddr.size();
		getConcatInputAddr(outAddrCount);
		for(int i=0; i<layer_count; i++)
		{
			Layer = vLayers[i];
		    ullInAddr.clear();
			ullOutAddr.clear();
			vInLayerCount.clear();
			if(i == 0)
			{
				Layer->uiLayerOutputAddr = ullOutAddr[0];
			}
			if (1 == Layer->cRunFlag)
			{
				if (Layer->LayerCommon.viInput[0] > 1)
				{
					int iLayerInputCount = Layer->LayerCommon.viInput[0];
				    for (int j = 0; j < iLayerInputCount; j++)
					{
                        vInLayerCount.push_back(Layer->vInOutLayer[j]);
					}
                    
					iLayerInputCount = vInLayerCount.size();
				    for (int j = 0; j < iLayerInputCount; j++)
					{	
                        ullInAddr.push_back(vLayers[vInLayerCount[j]]->uiLayerOutputAddr);
					}
					iIndex++;

					if (iIndex >= (outAddrCount - 1)) 
					{
						iIndex = 0;
					}
					if (Layer->cConcatFlag == 1)
					{
						ullOutAddr.push_back(Layer->concatLayerAddrBuf);
					}
					else
					{
						ullOutAddr.push_back(inOutAddr[iIndex]);
					}
					Layer->uiLayerOutputAddr = ullOutAddr[0];
				}
				else if (Layer->LayerCommon.viOutput[0] > 1)
				{
					ullInAddr.push_back(vLayers[Layer->vInOutLayer[0]]->uiLayerOutputAddr);
					iIndex++;
					if (iIndex >= (outAddrCount -1)) 
					{
						iIndex = 0;
					}
					ullOutAddr.push_back(inOutAddr[iIndex]);
					//ew inputaddress
					Layer->uiLayerOutputAddr = ullOutAddr[0];
					iIndex++;
					if (iIndex >= (outAddrCount -1)) 
					{
						iIndex = 0;
					}
					ullOutAddr.push_back(inOutAddr[iIndex]);
					for (int split_conv = 0; split_conv < Layer->splitOutputToConv.size(); split_conv++)
					{
						vLayers[Layer->splitOutputToConv[split_conv]]->getDataFromSplit = true;
						vLayers[Layer->splitOutputToConv[split_conv]]->concatLayerAddrBuf = inOutAddr[iIndex];
					}
				}
				else 
				{
					if(Layer->getDataFromSplit == true)
					{
						ullInAddr.push_back(Layer->concatLayerAddrBuf);
					}
					else
					{
						ullInAddr.push_back(vLayers[Layer->vInOutLayer[0]]->uiLayerOutputAddr);
					}
					iIndex++;
					if (iIndex >= (outAddrCount -1)) 
					{
						iIndex = 0;
					}
					
					if(Layer->cConcatFlag == 1)
					{
						ullOutAddr.push_back(Layer->concatLayerAddrBuf);
					}
					else
					{
						ullOutAddr.push_back(inOutAddr[iIndex]);
						if (Layer->outputFlag == true)
						{
							outputGiveToCpu.push_back(inOutAddr[iIndex]);
							outputGiveToCpu.push_back(Layer->LayerCommon.iOutFeaW);
							outputGiveToCpu.push_back(Layer->LayerCommon.iOutFeaH);
							outputGiveToCpu.push_back(Layer->LayerCommon.cOutputChannel);
						}
					}
					Layer->uiLayerOutputAddr = ullOutAddr[0];
				}
			}
			else if(0 == Layer->cRunFlag)
			{
				if (Layer->type == LAYER_CONCAT)
				{
					 Layer->uiLayerOutputAddr = Layer->concatLayerAddrBuf;
				}
				else
				{
					Layer->uiLayerOutputAddr = vLayers[Layer->vInOutLayer[0]]->uiLayerOutputAddr;
				}
				
				ullInAddr.push_back(0);
			}
			Layer->setRegisterValue(ullInAddr, ullOutAddr, num);
		}
	}


	/*************************************************************************
	* Function Name : getConcatInputAddr
	* Description   : allocate the input layers of concat layers address in advance
	* Parameters    : outAddrCount -- bufferAddress numbers
	* Returns       : 0 --success
	**************************************************************************/
	void Network::getConcatInputAddr(int outAddrCount)
	{
		unsigned int featureSize;
		unsigned int concatBuff = 0;
		unsigned int concatInputIndex = 0;
		bool concatFirstInputFlag;
		unsigned int concatIndex = 0;
		int layer_count = vLayers.size();
		LayerCom* Layer = NULL;
		LayerCom* concatInputLayer = NULL;
		LayerCom* concatInputInstructionLayer = NULL;
		concatFeatureSize.clear();
		concatFeatureSize.push_back(0);

		for (int i = 0; i < layer_count ; i++)
		{
			Layer = vLayers[i];
			if(Layer->type == LAYER_CONCAT) 
			{
				for(int j = i + 1;j < layer_count; j++)
				{
					if(!Layer->LayerCommon.topNames[0].compare(vLayers[j]->LayerCommon.bottomNames[0]))
					{
						if (vLayers[j]->type == LAYER_CONV)
						{
							Layer->cCloseQnFlag = false;
						}
						else
						{
							Layer->cCloseQnFlag = true;
						}
					 	break;
					}

				}				
			}

			if ( Layer->outputFlag == 1)
			{
				Layer->cCloseQnFlag = true;
			}
		}


		for (int i = 0; i<layer_count; i++)
		{
			Layer = vLayers[i];
			Layer->concatLayerAddrBuf = 0;
			if (Layer->type == LAYER_CONCAT)
			{
				int iLayerInputCount = Layer->LayerCommon.viInput[0];
				for (int t = 0; t < iLayerInputCount; t++)
				{
					int concatLayer = Layer->vInOutLayer[t];
					concatInputLayer = vLayers[concatLayer];
					if (concatInputLayer->LayerCommon.cOutputChannel == 0)
					{	
						featureSize = concatInputLayer->LayerCommon.iOutFeaW * concatInputLayer->LayerCommon.iOutFeaH;
						if (concatInputLayer->LayerCommon.iOutFeaH == 0)
						{
							featureSize = concatInputLayer->LayerCommon.iOutFeaW;
						}
						
					}
					else
					{
						featureSize = concatInputLayer->LayerCommon.iOutFeaH * concatInputLayer->LayerCommon.iOutFeaH * concatInputLayer->LayerCommon.cOutputChannel;

					}
							
					if (Layer->cCloseQnFlag == false)
					{
						concatFeatureSize.push_back(featureSize);
					}
					else
					{
						concatFeatureSize.push_back(featureSize * 4);
					}
				}		
			}
		}

		for (unsigned int j = 0; j < concatInstructionLayer.size(); j++)
		{
			concatInputInstructionLayer = vLayers[concatInstructionLayer[j]];
			concatInputInstructionLayer->cConcatFlag = 1;
			concatBuff = concatBuff + concatFeatureSize[concatIndex];
		    concatInputInstructionLayer->concatLayerAddrBuf = inOutAddr[outAddrCount - 1] + 2 * oneSegInOut + concatBuff;
			concatIndex++;
		}

		for (int i = 0; i < layer_count; i++)
		{
			Layer = vLayers[i];
			concatFirstInputFlag = false;
			if (Layer->type == LAYER_CONCAT)
			{
				int iLayerInputCount = Layer->LayerCommon.viInput[0];
				for (int t = 0; t < iLayerInputCount; t++)
				{ 
					if (concatFirstInputFlag == false)
					{
						int InstructionLayer = concatInstructionLayer[concatInputIndex];
						concatInputInstructionLayer = vLayers[InstructionLayer];
						Layer->concatLayerAddrBuf = concatInputInstructionLayer->concatLayerAddrBuf;
						concatFirstInputFlag = true;
					}
					concatInputIndex++;
				}		
			}
			
		}
	}
	/*************************************************************************
	* Function Name : writeWholeBinFile
	* Description   : write a complete tmmodel bin file
	* Parameters    : fileFp -- input ncnn bin file
	* Returns       : NULL
	**************************************************************************/
	void Network::writeWholeBinFile(FILE *fileFp, const char* num)
	{
		int layerLength = vLayers.size();
		char cPrinBuf[4];
		int channelnumber = atoi(num);
		unsigned int reluLayerId = UINT_MAX;
		LayerCom* Layer = NULL;
		FILE *fileOutFp = fopen(TMMODEL_BIN, "wb");
		assert(fileOutFp != NULL);
		char channelNumHead[LAYER_HEADER_SIZE] = {(char)0xAB, (char)0xBC, (char)0xCD, (char)0xDE, (char)channelnumber};
		fwrite(channelNumHead, sizeof(char), sizeof(channelNumHead), fileOutFp);
		for (int j = 0; j < layerLength; j++)
		{
			Layer = vLayers[j];
			if (Layer->cFirstFlag == true)
			{
				float firstInputScale = Layer->inputScale;
				memcpy(cPrinBuf, &firstInputScale, sizeof(firstInputScale));
				for(int k=0; k< 4; k++)
				{
					fwrite(&cPrinBuf[3-k],sizeof(char),1,fileOutFp);
				}
			}
			
		}
		for (int k = 0; k < outputGiveToCpu.size(); k++)
		{
			int outputMessage = outputGiveToCpu[k];
			memcpy(cPrinBuf, &outputMessage, sizeof(outputMessage));
			for(int k=0; k< 4; k++)
				{
					fwrite(&cPrinBuf[3-k],sizeof(char),1,fileOutFp);
				}
		}  

		for(int i=0; i<layerLength; i++)
		{
			//change write bin file order when conv Continuous appearance
			if((i > 1) && (vLayers[i-1]->type == LAYER_CONV) && (vLayers[i]->type == LAYER_CONV))											
			{
				for(unsigned int j = i + 1;j < layerLength; j++)
				{
					if(!vLayers[i-1]->LayerCommon.topNames[0].compare(vLayers[j]->LayerCommon.bottomNames[0]))
					{
						if(vLayers[j]->type == LAYER_RELU)
						{
							Layer = vLayers[j];
							dropCount = Layer->writeBinFile(fileFp, fileOutFp, j, num);
							reluLayerId = j;
						}
						break;
					}
				}
				Layer = vLayers[i];
				dropCount = Layer->writeBinFile(fileFp, fileOutFp, i, num);
			}
			else if(i != reluLayerId)
			{
				Layer = vLayers[i];
				dropCount = Layer->writeBinFile(fileFp, fileOutFp, i, num);
			}
		}

		fclose(fileOutFp);
		dropCount = 0;
		FILE *fileRp = fopen(TMMODEL_BIN, "ab+");
		for(int i=0; i<layerLength; i++)
		{
			Layer = vLayers[i];
			Layer-> writeddrBinFile(fileRp);
		}
		fclose(fileRp);
	}

	/*************************************************************************
	* Function Name : setEnableFlagForInstructionLayers
	* Description   : mark the instruction layer which need to make various enable 
	* Parameters    : void  
	* Returns       : void
	**************************************************************************/
	void Network::setEnableFlagForInstructionLayers(void)
	{
		int layerLength = vLayers.size();
		LayerCom* Layer = NULL;
		LayerCom* Layer_next = NULL;
		LayerCom* Layer_before = NULL;
		bool cFConv = false;
		int qn=0;
		int convLayerNum = 0;
		concatInstructionLayer.clear();
	
		for(int i=0; i<layerLength; i++)
		{
			Layer = vLayers[i];
			Layer->cPoolingFlag = false;
			Layer->cUpsampleFlag = false;
			Layer->cInstructionFlag = false;
			Layer->getDataFromSplit = false;
			if(Layer->type == LAYER_CONV && !cFConv) //cFConv = 
			{
				Layer->cFirstFlag = 1;
				cFConv = true;
			}
        
			if (i>0)
			{
				Layer_next = vLayers[i-1];
			
				if(Layer_next->type == LAYER_CONVDW && Layer->type == LAYER_PRELU)
				{
					Layer_next->cPreluFlag = true;
				}
				else if(Layer_next->type == LAYER_CONV && Layer->type == LAYER_PRELU)
				{
					Layer_next->cPreluFlag = true;
				}
				else
				{
					Layer_next->cPreluFlag = false;
				}

				if(Layer->type == LAYER_CONVDW || Layer->type == LAYER_CONV || Layer->type == LAYER_PERMUTE || Layer->type == LAYER_ELTWISE || Layer->type == LAYER_PRIORBOX)
				{
					Layer->cInstructionFlag = true;
				}

				if(Layer_next->type == LAYER_CONV && Layer->type == LAYER_SOFTMAX)
				{
					Layer_next->cSoftMaxFlag = true;
				}
				else
				{
					Layer_next->cSoftMaxFlag = false;
				}

				if (Layer_next->type == LAYER_CONV && Layer->type == LAYER_SOFTMAX)
				{
					Layer_next->cCloseQnFlag = true;
				}
				else if (Layer_next->type == LAYER_CONV && Layer->type == LAYER_PERMUTE)
				{
					Layer_next->cCloseQnFlag = true;
				}
				else if (Layer_next->type == LAYER_CONVDW && Layer->type == LAYER_SOFTMAX)
				{
					Layer_next->cCloseQnFlag = true;
				}
				else if (Layer_next->type == LAYER_CONVDW && Layer->type == LAYER_PERMUTE)
				{
					Layer_next->cCloseQnFlag = true;
				}
				else
				{
					Layer_next->cCloseQnFlag = false;
				}
				

				if(Layer->type == LAYER_CONV || Layer->type == LAYER_CONVDW)
				{
					qn++;
				}

				if(Layer->type == LAYER_CONV)
				{
					convLayerNum = i;
				}

				if (Layer->type == LAYER_UPSAMPLE)
				{
					int upsampleInput = Layer->vInOutLayer[0];
					setUpsampleFlag(upsampleInput);  
				}
				
				if (Layer->type == LAYER_CONCAT)
				{
					int iLayerInputCount = Layer->LayerCommon.viInput[0];
					for (int t = 0; t < iLayerInputCount; t++)
					{
						int concatLayer = Layer->vInOutLayer[t];
						getConcatInputFromInstructionLayer(concatLayer);
					}
				}
			}

			if(i>1)
			{
				Layer_before = vLayers[i-2];
			
				if(Layer_before->type == LAYER_CONV && Layer_next->type == LAYER_RELU && Layer->type == LAYER_POOLING)
				{
					Layer_before->cPoolingFlag = true;
				}
				else
				{
					Layer_before->cPoolingFlag = false;
				}
				if(Layer_before->type == LAYER_CONVDW && Layer_next->type == LAYER_RELU && Layer->type == LAYER_POOLING)
				{
					Layer_before->cPoolingFlag = true;
				}
				
				if(Layer_before->type == LAYER_CONV && Layer_next->type == LAYER_RELU && Layer->type == LAYER_ELTWISE)
				{
					Layer_before->cCloseQnFlag = true;
				}
				
				if ((Layer_before->type == LAYER_CONVDW || Layer_before->type == LAYER_CONV) && Layer_next->type == LAYER_ELTWISE && Layer->type == LAYER_RELU)
				{
					Layer->cEltwiseReLUFlag = true;
				}

				if (Layer_before->type == LAYER_ELTWISE && Layer_before->type == LAYER_RELU && Layer_before->type == LAYER_POOLING)
				{
					Layer->cPoolFlag = true;
				}

				if(Layer_before->type == LAYER_PERMUTE && Layer_next->type == LAYER_FLATTEN && Layer->type == LAYER_PRIORBOX)
				{
					Layer_before->cSoftMaxFlag = true;
				}
		    }
		}

		Layer=vLayers[convLayerNum];
		Layer->cCloseQnFlag = true;

		for (int i = 0; i < layerLength; i++)
		{
			Layer = vLayers[i];
			Layer->cReLUFlag = false;
			if(Layer->type == LAYER_RELU && Layer->reluWriteBinFlag == false) 
			{
				for(int j = i-1;j >= 0;j--)
				{
					if(!Layer->LayerCommon.bottomNames[0].compare(vLayers[j]->LayerCommon.topNames[0]))
					{
						vLayers[j]->cReLUFlag = true;
					 	break;
					}

				}				
			}
			if(Layer->type == LAYER_RELU && Layer->reluWriteBinFlag == true) 
			{
				for(int j = i-1;j >= 0;j--)
				{
					if(!Layer->LayerCommon.bottomNames[0].compare(vLayers[j]->LayerCommon.topNames[0]))
					{
						vLayers[j]->cPreluFlag = true;
					 	break;
					}
				}				
			}
		}
		classifySplitOutputLayers();
		setOutputFlag();
	}


	/*************************************************************************
	* Function Name : setOutputFlag
	* Description   : mark the instruction layer which need to output
	* Parameters    : void  
	* Returns       : void
	**************************************************************************/
	void Network::setOutputFlag(void)
	{
		int layer_count = vLayers.size();
		LayerCom* Layer = NULL;
		LayerCom* nextLayer = NULL;
		for(int i = 0; i < layer_count; i++)
		{
			Layer = vLayers[i];
			Layer->outputFlag = false;
			if (Layer->type == LAYER_CONV)
			{
				Layer->outputFlag = true;
				int LayerOutputSize = Layer->LayerCommon.viOutput[0];
				for (int n = 0; n < LayerOutputSize; n++)
				{
					for (int j = i + 1; j < layer_count; j++)
					{
						nextLayer = vLayers[j];
						int iLastLayerInputCount = nextLayer->LayerCommon.viInput[0];
						for(int k = 0; k < iLastLayerInputCount; k++)
						{
							if(!Layer->LayerCommon.topNames[n].compare(nextLayer->LayerCommon.bottomNames[k]))
							{
								Layer->outputFlag = false;
								break;
							}
						}
					}
				}
			}
		}
	}




	/*************************************************************************
	* Function Name : classifySplitOutputLayers
	* Description   : classify output layers of Split into Eltwise and Convolution
	* Parameters    : void  
	* Returns       : void
	**************************************************************************/
	void Network::classifySplitOutputLayers(void)
	{
		int layer_count = vLayers.size();
		LayerCom* Layer = NULL;
		LayerCom* nextLayer = NULL;
		for(int i = 0; i < layer_count; i++)
		{
			Layer = vLayers[i];
			Layer->splitOutputToConv.clear();
			Layer->splitOutputToEW.clear();
			if (Layer->type == LAYER_SPLIT)
			{
				Layer->cRunFlag = 0;
				int LayerOutputSize = Layer->LayerCommon.viOutput[0];
				for (int n = 0; n < LayerOutputSize; n++)
				{
					for (int j = i + 1; j < layer_count; j++)
					{
						nextLayer = vLayers[j];
						int iLastLayerInputCount = nextLayer->LayerCommon.viInput[0];
						for(int k = 0; k < iLastLayerInputCount; k++)
						{
							if(!Layer->LayerCommon.topNames[n].compare(nextLayer->LayerCommon.bottomNames[k]))
							{
								if(nextLayer->type == LAYER_CONV)
								{
									Layer->splitOutputToConv.push_back(j);
									break;
								}
									
								if(nextLayer->type == LAYER_ELTWISE)
								{
									Layer->splitOutputToEW.push_back(j);
									Layer->cRunFlag = 1;
									break;
								}
							}
						}
					}
				}
			}
		}
	}




	/*************************************************************************
	* Function Name : setUpsampleFlag
	* Description   : mark the instruction layer which need to make Upsample enable 
	* Parameters    : layerNum -- input layers id of current layer  
	* Returns       : 0 --success
	**************************************************************************/
	int Network::setUpsampleFlag(int layerNum)
	{
		LayerCom* Layer = NULL;
		LayerCom* lastLayer = NULL;
		bool compareFinish = false;
		int layerIndex = layerNum;
		Layer = vLayers[layerIndex];
		for (int i = layerIndex - 1; i >= 0; i--)
		{
			lastLayer = vLayers[i];
			int iLastLayerOutputSize = lastLayer->LayerCommon.viOutput.size();
			if (iLastLayerOutputSize > 0)
				{
					int iLastLayerInputCount = lastLayer->LayerCommon.viOutput[0];
					for(int k = 0; k < iLastLayerInputCount; k++)
					{
						/*current layer output compare to following input to find next input layer*/
						if(!Layer->LayerCommon.bottomNames[0].compare(lastLayer->LayerCommon.topNames[k])) 
						{
							compareFinish = true;
							if (lastLayer-> type == LAYER_CONV || lastLayer-> type == LAYER_CONVDW)
							{
								lastLayer->cUpsampleFlag = true;
								return 0;
							}
							break;
						}
					}
					if (compareFinish)
					{
						layerIndex = i;
						compareFinish = false;
						break;
					}
				}
		}
		/*if reach the condition end the loop*/
		setUpsampleFlag(layerIndex);
		return -1;
	}




	/*************************************************************************
	* Function Name : getConcatInputFromInstructionLayer
	* Description   : get Concat input layer from which instruction layer
	* Parameters    : layerNum -- input layers id of current layer  
	* Returns       : 0 --success
	**************************************************************************/
	int Network::getConcatInputFromInstructionLayer(int layerNum)
	{
		LayerCom* Layer = NULL;
		LayerCom* lastLayer = NULL;
		int layerIndex = layerNum;
		Layer = vLayers[layerIndex];
		if (Layer->cInstructionFlag == true)
		{
			concatInstructionLayer.push_back(layerIndex);
			return 0;
		}
		else
		{
			for (int i = layerIndex - 1; i >= 0; i--)
			{
				lastLayer = vLayers[i];
				int iLastLayerOutputSize = lastLayer->LayerCommon.viOutput.size();
				if (iLastLayerOutputSize > 0)
					{
						int iLastLayerInputCount = lastLayer->LayerCommon.viOutput[0];
						for(int k = 0; k < iLastLayerInputCount; k++)
						{
							/*current layer output compare to following input to find next input layer*/
							if(!Layer->LayerCommon.bottomNames[0].compare(lastLayer->LayerCommon.topNames[k])) 
							{
								layerIndex = i;							
								if (lastLayer->cInstructionFlag == true)
								{
									concatInstructionLayer.push_back(i);
									return 0;
								}
								break;
							}
						}
					}
			}
		}
		/*if reach the condition end the loop*/
		getConcatInputFromInstructionLayer(layerIndex);
		return -1;
	}

	/*************************************************************************
	* Function Name : printConstruct
	* Description   : print Construct.txt
	* Parameters    : NULL
	* Returns       : NULL
	**************************************************************************/
	void Network::printConstruct(void)
	{
	#if PRINT_INFO
		int layer_count = layers.size();
		LayerCom* Layer = NULL;
		FILE *fileOutFp = fopen(CONSTRUCT_TXT,"w");
		char cPrintfbuf[100];
		int dropCount = 0;
		memset(cPrintfbuf,0,sizeof(cPrintfbuf));

		for(int i=1; i<layer_count; i++)
		{
			Layer = layers[i];
			if(Layer->type == LAYER_INPUT || Layer->type == LAYER_DROP)
			{
				dropCount++;
				continue;
			}
			//type
			sprintf(cPrintfbuf,"type: ");
			fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
			fwrite(pcLayerName[Layer->type],strlen(pcLayerName[Layer->type]),1,fileOutFp);
			memset(cPrintfbuf,0,sizeof(cPrintfbuf));
			sprintf(cPrintfbuf," layer=%d\n",i-dropCount);
			fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
			memset(cPrintfbuf,0,sizeof(cPrintfbuf));

			//input
			sprintf(cPrintfbuf,"input layer:%d ",Layer->LayerCommon.viInput.size());
			fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
			memset(cPrintfbuf,0,sizeof(cPrintfbuf));

			for(int i=0; i<Layer->LayerCommon.viInput.size(); i++)
			{
				sprintf(cPrintfbuf," %d",Layer->LayerCommon.viInput[i]-dropCount);
				fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
				memset(cPrintfbuf,0,sizeof(cPrintfbuf));
			}

			//ouput
			sprintf(cPrintfbuf," out layer:%d ",Layer->LayerCommon.viOutput.size());
			fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
			memset(cPrintfbuf,0,sizeof(cPrintfbuf));

			for(int i=0; i<Layer->LayerCommon.viOutput.size(); i++)
			{
				sprintf(cPrintfbuf," %d",Layer->LayerCommon.viOutput[i]-dropCount);
				fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
				memset(cPrintfbuf,0,sizeof(cPrintfbuf));
			}
			sprintf(cPrintfbuf,"\n");
			fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
			memset(cPrintfbuf,0,sizeof(cPrintfbuf));
			//feature
			sprintf(cPrintfbuf,"input size w=%d h=%d c=%d\n",Layer->LayerCommon.iInFeaW,Layer->LayerCommon.iInFeaH,Layer->LayerCommon.cInputChannel);
			fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
			memset(cPrintfbuf,0,sizeof(cPrintfbuf));
			sprintf(cPrintfbuf,"output size w=%d h=%d c=%d\n",Layer->LayerCommon.iOutFeaW,Layer->LayerCommon.iOutFeaH,Layer->LayerCommon.cOutputChannel);
			fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
			memset(cPrintfbuf,0,sizeof(cPrintfbuf));

			sprintf(cPrintfbuf,"\n");
			fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
			memset(cPrintfbuf,0,sizeof(cPrintfbuf));
		}
	#endif
	}

	/*************************************************************************
	* Function Name : loadParamFile
	* Description   : load param file
	* Parameters    : protopath -- input ncnn param
	* Returns       : 0 success
	**************************************************************************/
	int Network::loadParamFile(const char* protopath)
	{
		std::string layer_type;
		std::string layer_name;
		std::vector<std::string> top_blob;
		std::vector<std::string> bottom_blob;
		std::string blob;

		int magic = 0;
		int layer_count = 0;
		int blob_count = 0;
		int nscan = 0;
		int bottom_count = 0;
		int top_count = 0;
		int id = 0;
		int value = 0;
		int output_num = 0;
		FILE* fp = fopen(protopath, "rb");
		if (!fp)
		{
			tmtool_log(LOG_ERROR, "fopen %s failed\n", protopath);
			return -1;
		}
   
		int nbr = fscanf(fp, "%d", &magic);

		if (nbr != 1)
		{
			tmtool_log(LOG_ERROR, "issue with param file");
			return -1;
		}

		if (magic != PARAM_MAGIC)
		{
			tmtool_log(LOG_ERROR, "param is too old, please regenerate");
			return -1;
		}

		nbr = fscanf(fp, "%d %d", &layer_count, &blob_count);
		if (nbr != 2 || layer_count <= 0 || blob_count <= 0)
		{
			tmtool_log(LOG_ERROR, "issue with param file");
			return -1;
		}

		vLayers.resize(layer_count);
		blobs.resize(layer_count);
		//blobs.clear();

		for (int i=0; i<layer_count; i++)
		{
			bottom_blob.clear();
			top_blob.clear();
			layer_type.clear();
			layer_name.clear();
			blob.clear();
			layer_type.resize(256);
			layer_name.resize(256);
			nscan = fscanf(fp, "%256s %256s %d %d ", &layer_type[0], &layer_name[0], &bottom_count, &top_count);
		
			if (nscan != 4)
			{
				continue;
			}

			if (bottom_count != 0 && top_count != 0)
			{   
				if (bottom_count > 1 || top_count > 1) 
				{
					for (int j = 0; j < bottom_count; j++)
					{
						blob.clear();
						blob.resize(256);
						nscan = fscanf(fp, "%256s ", &blob[0]);
						bottom_blob.push_back(blob);
					}
					for (int j = 0; j < top_count; j ++)
					{
						blob.clear();
						blob.resize(256);
						nscan = fscanf(fp, "%256s ", &blob[0]);
						top_blob.push_back(blob);
					}
				} else {
					blob.resize(256);
				    nscan = fscanf(fp, "%256s ", &blob[0]);
					bottom_blob.push_back(blob);
					blob.clear();
					blob.resize(256);
				    nscan = fscanf(fp, "%256s ", &blob[0]);
					top_blob.push_back(blob);
				}
				//write output
				writeOneBlobsFromLayer(top_blob,i);  
				//write input
				writeOneBlobsForLayer(bottom_blob,i);  
			}
			else if (bottom_count == 0 && top_count != 0)
			{
				for (int j = 0; j < top_count; j++)
				{
				    blob.resize(256);
				    nscan = fscanf(fp, "%256s ", &blob[0]);
					top_blob.push_back(blob);
				}
	            blob.clear();
				blob.resize(256);
				char cstr[1] = ""; //for run
				blob.assign(cstr);
				bottom_blob.push_back(blob);
				writeOneBlobsFromLayer(top_blob,i);
				writeOneBlobsForLayer(bottom_blob,i);  
			}

			if((NULL != strstr(layer_type.c_str(), PRIORBOX_NAME)) || (NULL != strstr(layer_type.c_str(), RELU_NAME)))
			{
				char fvalue[50];
				double fvalue1;
				int index = 0;
				// if (fscanf(fp, "%d=%d,%f", &id, &index, &fvalue1) == 2)
				// {
				// 	output_num = 0;
				// }
			}
			else
			{
				if (fscanf(fp, "%d=%d", &id, &value) == 2)
				{
					output_num = value;
				}
				else
				{
					output_num = 0;
				}
			}

			LayerCom* Layer = createLayer(layer_type.c_str(), output_num);
			Layer->name = layer_name;
			Layer->loadParam(fp, output_num);
			vLayers[i] = Layer;
		}

		//network structure
		fillLayerStructInOut();

		//calculate feature size
		calculateAllFeatureSize();

		//fill bin file data ddr address
		calculateLayerRelationship();
		setEnableFlagForInstructionLayers();  

		
		fclose(fp);
		return 0;
	}

	/*************************************************************************
	* Function Name : loadBinFile
	* Description   : load bin file
	* Parameters    : protopath -- input ncnn param
	* Returns       : NULL
	**************************************************************************/
	int Network::loadBinFile(const char* protopath, const char* num)
	{
		fillAllDDRAddress(num);	
		FILE* fp = fopen(protopath, "rb");
 
		if(fp == NULL)
		{
			tmtool_log(LOG_ERROR, "loadBinFile openfile fail");
			return -1;
		}

		// //load all the input scale value from bin file
		loadInstructionParam(fp);  

		// //set all the layer that need to set quantize register and fp32 translate register
		// setQuanFp32Value();
		// //set all the register balue
	
		// //write a json file
		// writeWholeCjson();

		/* reset to origin position */
		fseek(fp, 0, SEEK_SET);

		setAllRegister(num);

		writeWholeBinFile(fp, num);
		fclose(fp);
		return 0;
	}

	/*************************************************************************
	* Function Name : createLayer
	* Description   : create layer based on layer name
	* Parameters    : type -- layer name
	* Returns       : NULL
	**************************************************************************/
	LayerCom *Network::createLayer(const char* type, int pooling)
	{	
		LayerCom *newLayer = NULL;
		int iLayerIndex = 0;

		for(int i=0; i<supportLayer; i++)
		{
			if (strcmp(type,pcLayerName[i])==0)
			{
				iLayerIndex = i;
			}
		}

		if (iLayerIndex == LAYER_POOLING && pooling == 1)
		{
			iLayerIndex = LAYER_POOLING_CONVDW;
		}

		switch(iLayerIndex)
		{
			case LAYER_INPUT:
				newLayer = new Input;
				break;
			case LAYER_BINARYOP:
				newLayer = new BinaryOp;
				break;
			case LAYER_CONV:
				newLayer = new Convolution;
				break;
			case LAYER_CONVDW:
				newLayer = new ConvolutionDepthWise;
				break;
			case LAYER_PRELU:
				newLayer = new PReLU;
				break;
			case LAYER_SPLIT:
				newLayer = new Split;
				break;
			case LAYER_ELTWISE:
				newLayer = new Eltwise;
				break;
			case LAYER_RELU:
				newLayer = new ReLU;
				break;
			case LAYER_POOLING:
				newLayer = new Pooling;
				break;
			case LAYER_BATCHNORM:
				newLayer = new BatchNorm;
				break;
			case LAYER_REGION:
				newLayer = new Region;
				break;
			case LAYER_REORG:
				newLayer = new Reorg;
				break;
			case LAYER_CONCAT:
				newLayer = new Concat;
				break;
			case LAYER_SCALE:
				newLayer = new Scale;
				break;
			case LAYER_SOFTMAX:
				newLayer = new Softmax;
				break;			
			case LAYER_RESHAPE:
				newLayer = new Reshape;
				break;
			case LAYER_POOLING_CONVDW:
	            newLayer = new PConvolutionDepthWise;
				break;
			case LAYER_PRIORBOX:
	            newLayer = new PriorBox;
				break;
			case LAYER_DETECTIONOUTPUT:
	            newLayer = new DetectionOutput;
				break;
			case LAYER_PERMUTE:
				newLayer = new Permute;
				break;
			case LAYER_FLATTEN:
				newLayer = new Flatten;
				break;	
			case LAYER_UPSAMPLE:
	            newLayer = new Upsample;
				break;
			default: 
			break;
		}
		newLayer->type = iLayerIndex;
		return newLayer;
	}

	/*************************************************************************
	* Function Name : getMaxFeatureSize
	* Description   : calculate all buffer size and get the biggest one
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	void Network::getMaxFeatureSize(void)
	{
		int layer_count = vLayers.size();
		LayerCom* Layer = NULL;
		unsigned int uiMax = 0;

		for(int i=0; i<layer_count; i++)
		{
			Layer = vLayers[i];
			if(!i)
			{
				uiMax = Layer->LayerCommon.uiInputSize;
				uiMax = max(uiMax,Layer->LayerCommon.uiOutputSize);
			}
			else
			{
				uiMax = max(uiMax,Layer->LayerCommon.uiInputSize);
				uiMax = max(uiMax,Layer->LayerCommon.uiOutputSize);
			}
		}

		maxFeature = uiMax;
		if(maxFeature%DDR_4K)
		{
			oneSegInOut = (maxFeature/DDR_4K + 1)*DDR_4K;
		}
		else
		{
			oneSegInOut = maxFeature;
		}
	}

	/*************************************************************************
	* Function Name : getBoardInformation
	* Description   : set board information from the board setting file and input board name
	* Parameters    : protopath -- board setting file
	* 				  inBoardName -- input board name
	* Returns       : -1 -- board name no found
	*                 other -- index of input board name information
	**************************************************************************/
	int Network::getBoardInformation(const char* protopath,const char* inBoardName)
	{ 
		readDDRConfigFile(protopath);  
		int iTotalBoard = vBoardData.size(); 
		BoardData* pBoard = NULL;

		for(int i=0; i<iTotalBoard; i++)
		{
			pBoard = vBoardData[i];
			//check if the config file has input board name
			if (!pBoard->strName.compare(inBoardName))  
			{
				boardTypeIndex = i;
				if (pBoard->cWeightPlace)
				{
					//put bin file data to PL DDR
					if (pBoard->uiPlSize != 0)
					{
						binDataAddr = pBoard->ullPlStart;
					}
					else
					{
						//PL DDR is 0mb
						binDataAddr = pBoard->ullPsStart;
					}

				}
				else
				{
					//put  bin file data to PS DDR
					if (pBoard->uiPsSize != 0)
					{
						binDataAddr = pBoard->ullPsStart;
					}
					else
					{
						//PS DDR is 0mb
						binDataAddr = pBoard->ullPlStart;
					}
				}
				break;
			}
		}
		if (boardTypeIndex >= 0)
		{
			pBoard = vBoardData[boardTypeIndex];
			if (pBoard->uiPlSize != 0)
			{
				plMaxAddr = pBoard->ullPlStart + (pBoard->uiPlSize - PL_NO_USE)*MB_TO_BYTE;
			}

			if (pBoard->uiPsSize != 0)
			{
				psMaxAddr = pBoard->ullPsStart + (pBoard->uiPsSize-100)*MB_TO_BYTE;  
			}

			if (pBoard->uiPlSize == 0 && pBoard->uiPsSize == 0)
			{
				boardTypeIndex = -1;
			}
		}
		return boardTypeIndex;
	}

	/*************************************************************************
	* Function Name : printAllDDRInfo
	* Description   : printf all ddr information to DDR_TXT
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	void Network::printAllDDRInfo(void)
	{
	#if PRINT_INFO
		FILE *fileOutFp = fopen(DDR_TXT,"w");
		char cPrintfbuf[100];
		int layer_count = layers.size();
		LayerCom* Layer = NULL;

		//write input output information
		sprintf(cPrintfbuf,"BOARD_NAME: ");
		fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
		fwrite(vBoardData[boardTypeIndex]->strName.data(),strlen(cPrintfbuf),1,fileOutFp);

		sprintf(cPrintfbuf,"\naddress %d bits\n",vBoardData[boardTypeIndex]->cAddrBits);
		fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
		sprintf(cPrintfbuf,"PS size %d Mb\n",vBoardData[boardTypeIndex]->uiPsSize);
		fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
		sprintf(cPrintfbuf,"PL size %d Mb\n",vBoardData[boardTypeIndex]->uiPlSize);
		fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
		sprintf(cPrintfbuf,"PS Start Addr 0x%llx\n",vBoardData[boardTypeIndex]->ullPsStart);
		fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
		sprintf(cPrintfbuf,"PL Start Addr 0x%llx\n",vBoardData[boardTypeIndex]->ullPlStart);
		fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
		sprintf(cPrintfbuf,"Weight data put to [%d]\n",vBoardData[boardTypeIndex]->cWeightPlace);
		fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
		sprintf(cPrintfbuf,"Input Output data put to [%d]\n",vBoardData[boardTypeIndex]->cDataPlace);
		fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
		sprintf(cPrintfbuf,"Input Output data buffer number %d\n",inOutAddr.size());
		fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
		sprintf(cPrintfbuf,"Input Output data one seg 0x%x\n",oneSegInOut);
		fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
		sprintf(cPrintfbuf,"The biggest feature in bytes is %d\n",maxFeature);
		fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
		sprintf(cPrintfbuf,"PS MAX address is 0x%llx\n",psMaxAddr);
		fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
		sprintf(cPrintfbuf,"PL MAX address is 0x%llx\n",plMaxAddr);
		fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);


		if(plOverflowFlag && psOverflowFlag)
		{
			//using the second half of the PS DDR!
			sprintf(cPrintfbuf,"**********using the second half of the PS DDR!************\n");
			printf("**********using the second half of the PS DDR!************\n");
			fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
		}

		if(plOverflowFlag)
		{
			//PL is overflow
			sprintf(cPrintfbuf,"##########PL is overflow!##########\n");
			printf("##########PL is overflow!##########\n");
			fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
		}

		if(psOverflowFlag)
		{
			sprintf(cPrintfbuf,"##########PS is overflow!##########\n");
			printf("##########PS is overflow!##########\n");
			fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
		}

		sprintf(cPrintfbuf,"**********weight data************\n");
		fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
		//write weight address
		for(int i=0; i<layer_count; i++)
		{
			Layer = layers[i];
			if(Layer->type == LAYER_CONV || Layer->type == LAYER_FC)
			{
				Layer->writeDDRInfoWeight(fileOutFp);
			}
		}

		sprintf(cPrintfbuf,"**********input putput data************\n");
		fwrite(cPrintfbuf,strlen(cPrintfbuf),1,fileOutFp);
		//write input output address
		for(int i=0; i<layer_count; i++)
		{
			Layer = layers[i];
			Layer->writeDDRInfoInputOutput(fileOutFp);
		}


		fclose(fileOutFp);

	#endif
	}

}

