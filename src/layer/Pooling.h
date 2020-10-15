/*
 * Pooling.h
 *
 *  Created on: Jun 12, 2019
 *      Author: doyle
 */

#ifndef POOLING_H_
#define POOLING_H_

#include "layercommon.h"
#include "Pool_param.h"

namespace tmnet
{
	class Pooling : public LayerCom , public PoolParam
	{
	public:
		Pooling();
		~Pooling();

		//quantize flag
		int iQuantizeFlag;
		float fArithScale;

		//load param data
		int loadParam(FILE* fileFp, int output_num);
		//calculate feature size
		int calculateFeaSize(std::vector<unsigned int> iIw, std::vector<unsigned int> iIh, std::vector<unsigned int> iIc);
		//bin data ddr address
		unsigned long long fillDDRAddress(unsigned long long uiLastAddr,const char* num);
		//get input scale
		int getInputScale(FILE *fileFp);
		//set qunaitze value
		void setQuantize(int iQuantize,float fScale);
		//set registers value
		unsigned long long setRegisterValue(unsigned long long uiLastAddr,unsigned int uiOneSeg,\
											const unsigned int uiOriAddr,char cBufferNum);

		int setRegisterValue(std::vector<unsigned long long> uiInputAddr, std::vector<unsigned long long> uiOutputAddr,const char* num);

		unsigned long long getNextInputOutputAddr(unsigned long long uiAddr,unsigned int uiOneSeg,\
													unsigned int uiOriAddr,char cBufferNum);
		//write config file
		void writeRegToJson(cJSON *json);
		//write bin file
		int writeBinFile(FILE *fileInFp,FILE *fileOutFp,int iLayerIndex,const char* num);
		int writeddrBinFile(FILE *fileRp);
		//write input output data to DDR_info.txt
		void writeDDRInfoInputOutput(FILE *fileOutFp);

	private:
		unsigned int poolRegCtrl(unsigned char ucQuanEnable);
	};
}
#endif /* POOLING_H_ */
