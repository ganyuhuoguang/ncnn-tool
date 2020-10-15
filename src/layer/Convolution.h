/*
 * Convolution.h
 *
 *  Created on: Jun 11, 2019
 *      Author: doyle
 */

#ifndef CONVOLUTION_H_
#define CONVOLUTION_H_

#include "layercommon.h"
#include "conv_param.h"

namespace tmnet
{
	class Convolution : public LayerCom , public ConvParam
	{
	public:
		//construct
		Convolution();
		~Convolution();
		//load param data
		int loadParam(FILE* fileFp, int output_num);
		//calculate feature size
		int calculateFeaSize(std::vector<unsigned int> iIw, std::vector<unsigned int> iIh, std::vector<unsigned int> iIc);
		//bin data ddr address
		uint64 fillDDRAddress(uint64 uiLastAddr,const char* num);
		//get input scale
		int getInputScale(FILE *fileFp);
		//set registers value
		uint64 setRegisterValue(uint64 uiLastAddr,unsigned int uiOneSeg,\
											const unsigned int uiOriAddr,char cBufferNum);
		int setRegisterValue(std::vector<unsigned long long> uiInputAddr, std::vector<unsigned long long> uiOutputAddr,const char* num);

		uint64 getNextInputOutputAddr(uint64 uiAddr,unsigned int uiOneSeg,\
													unsigned int uiOriAddr,char cBufferNum);
		//get fp32 translate information
		int getFp32Infor(uint64 *uiBias,uint64 *uiWScale,float *fIScale);
		//get fp32 translate information
		int getQuantizeInfor(float *fIScale);

		//write bin file
		int writeBinFile(FILE *fileInFp,FILE *fileOutFp,int iLayerIndex,const char* num);

		int writeddrBinFile(FILE *fileRp);
		//write weight data to DDR_info.txt
		void writeDDRInfoWeight(FILE *fileOutFp);
		//write input output data to DDR_info.txt
		void writeDDRInfoInputOutput(FILE *fileOutFp);

	private:
		unsigned int uiIfmSplTim;
		unsigned int uiRowPerLd;

		//register value setting
		unsigned int convRegKernelSize(void);
		unsigned int convRegFeatureSize(void);
		unsigned int convRegFeatureChannel(void);
		unsigned int convRegPadCtrl(void);
		unsigned int convRegIFM_SPL_TIM(void);
		unsigned int convRegROW_PER_LD(void);
		unsigned int convRegROW_LST_LD(void);

		//calculate ddr address
		uint64 getNewAddrFormDataSize(unsigned int uiDataSize,uint64 uiAddress);
		//set bin data size
		void setLayerBinDataSize(void);
	};
}
#endif /* CONVOLUTION_H_ */
