/*
 * ConvolutionDepthWise.h
 *
 *  Created on: Jun 11, 2019
 *      Author: doyle
 */

#ifndef PCONVOLUTIONDEPTHWISE_H_
#define PCONVOLUTIONDEPTHWISE_H_

#include "layercommon.h"
#include "pconvdw_param.h"

namespace tmnet
{

	#define CONV_BUFF_SIZE	16384

	#define CONV_WEIGHT_BUFF	1024

	#define PRINT_BUF_SIZE	4

	#define REG_KER_H_BIT	16
	#define REG_FEA_H_BIT	16
	#define REG_FEA_CO_BIT	16
	#define REG_PAD_SZ_BIT	1
	#define REG_PAD_ENABLE	1

	#define CONV_WEIGHT_VAL	0x01010101

	class PConvolutionDepthWise : public LayerCom , public PConvDwParam
	{
	public:
		//construct
		PConvolutionDepthWise();
		~PConvolutionDepthWise();

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
		//get fp32 translate information
		int getFp32Infor(unsigned long long *uiBias,unsigned long long *uiWScale,float *fIScale);
		//get fp32 translate information
		int getQuantizeInfor(float *fIScale);
		//write bin file
		int writeBinFile(FILE *fileInFp,FILE *fileOutFp,int iLayerIndex,const char* num);
		int writeBinFile(int iDropCount,FILE *fileInFp,FILE *fileOutFp,int iLayerIndex,char cBit);
		//write weight data to DDR_info.txt
		int writeddrBinFile(FILE *fileRp);
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
		unsigned long long getNewAddrFormDataSize(unsigned int uiDataSize,unsigned long long uiAddress);
		//set bin data size
		void setLayerBinDataSize(void);
	};

}


#endif /* CONVOLUTION_H_ */
