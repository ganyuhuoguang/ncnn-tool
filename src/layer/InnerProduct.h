/*
 * InnerProduct.h
 *
 *  Created on: Jun 12, 2019
 *      Author: doyle
 */

#ifndef INNERPRODUCT_H_
#define INNERPRODUCT_H_

#include "layercommon.h"
#include "InnerProduct_param.h"

namespace tmnet
{
	class InnerProduct : public LayerCom , public FcParam
	{
	public:
		InnerProduct();
		~InnerProduct();

		//load param data
		int loadParam(FILE* fileFp, int output_num);
		//calculate feature size
		int calculateFeaSize(std::vector<unsigned int> iIw, std::vector<unsigned int> iIh, std::vector<unsigned int> iIc);
		//bin data ddr address
		unsigned long long fillDDRAddress(unsigned long long uiLastAddr,const char* num);
		//get input scale
		int getInputScale(FILE *fileFp);
		//set registers value
		unsigned long long setRegisterValue(unsigned long long uiLastAddr,unsigned int uiOneSeg,\
											const unsigned int uiOriAddr,char cBufferNum);

		int setRegisterValue(std::vector<unsigned long long> uiInputAddr, std::vector<unsigned long long> uiOutputAddr,const char* num);

		unsigned long long getNextInputOutputAddr(unsigned long long uiAddr,unsigned int uiOneSeg,\
													unsigned int uiOriAddr,char cBufferNum);
		//write config file
		void writeRegToJson(cJSON *json);
		//write bin file
		int writeBinFile(int iDropCount,FILE *fileInFp,FILE *fileOutFp,int iLayerIndex,char cBit);
		//write weight data to DDR_info.txt
		void writeDDRInfoWeight(FILE *fileOutFp);
		//write input output data to DDR_info.txt
		void writeDDRInfoInputOutput(FILE *fileOutFp);

	private:
		//calculate ddr address
		unsigned long long getNewAddrFormDataSize(unsigned int uiDataSize,unsigned long long uiAddress);
		//set bin data size
		void setLayerBinDataSize(void);
	};
}
#endif /* INNERPRODUCT_H_ */
