/*
 * Dropout.h
 *
 *  Created on: Jun 12, 2019
 *      Author: doyle
 */

#ifndef DROPOUT_H_
#define DROPOUT_H_

#include "layercommon.h"

namespace tmnet
{
	class Dropout : public LayerCom
	{
	public:
		Dropout();
		~Dropout();

		//load param file data
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
		//write config file
		void writeRegToJson(cJSON *json);
		//write bin file
		int writeBinFile(int iDropCount,FILE *fileInFp,FILE *fileOutFp,int iLayerIndex,char cBit);
		//write input output data to DDR_info.txt
		void writeDDRInfoInputOutput(FILE *fileOutFp);
	};
}



#endif /* DROPOUT_H_ */
