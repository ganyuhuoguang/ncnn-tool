/*
 * Flatten.h
 *
 *  Created on: Jun 12, 2019
 *      Author: doyle
 */

#ifndef FLATTEN_H_
#define FLATTEN_H_

#include "layercommon.h"
#include "flatten_param.h"

namespace tmnet
{
	class Flatten : public LayerCom , public FlattenParam
	{
	public:
		Flatten();
		~Flatten();

		//quantize flag
		int iQuantizeFlag;
		float fBsQn;
		//fp32 flag
		int iFp32Flag;
		unsigned long long uiBsAluSrc;
		unsigned long long uiBsMulSrc;
		float fOprand;

		//load param data
		int loadParam(FILE* fileFp, int output_num);
		//calculate feature size
		int calculateFeaSize(std::vector<unsigned int> iIw, std::vector<unsigned int> iIh, std::vector<unsigned int> iIc);
		//set qunaitze value
		void setQuantize(int iQuantize,float fIScale);
		//set fp32 value
		void setFp32(int iFp32,unsigned long long uiBiasAddr,unsigned long long uiWScaleAddr,float fIScale);
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
		int writeBinFile(FILE *fileInFp, FILE *fileOutFp, int iLayerIndex, const char* num);
		int writeddrBinFile(FILE *fileRp);
		//write input output data to DDR_info.txt
		void writeDDRInfoInputOutput(FILE *fileOutFp);
	};
}



#endif /* Flatten_H_ */
