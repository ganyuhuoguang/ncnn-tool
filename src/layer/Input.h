/*
 * Input.h
 *
 *  Created on: Jun 12, 2019
 *      Author: doyle
 */

#ifndef INPUT_H_
#define INPUT_H_

#include "layercommon.h"
#include "Input_param.h"

namespace tmnet 
{
	class Input : public LayerCom , public Inputparam
	{
	public:
		Input();
		~Input();
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
		int writeddrBinFile(FILE *fileRp);
		//write bin file
		int writeBinFile(FILE *fileInFp,FILE *fileOutFp,int iLayerIndex,const char* num);
		//write input output data to DDR_info.txt
		void writeDDRInfoInputOutput(FILE *fileOutFp);
	};
}

#endif /* INPUT_H_ */
