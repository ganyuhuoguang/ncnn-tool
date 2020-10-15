/*
 * layercommon.cpp
 *
 *  Created on: Jun 11, 2019
 *      Author: doyle
 */

#include "layercommon.h"

namespace tmnet 
{
	/*************************************************************************
	* Function Name : LayerCom
	* Description   : construct function ,initialize all the variable
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	LayerCom::LayerCom()
	{
		memset(&LayerCommon,0,sizeof(LAYER_COMMON));
		cFirstFlag = 0;
		uiBinDataSize = 0;
		cCloseQnFlag = 0;               
		cPoolFlag = 0;
		cPreluFlag = 0;
		cCpuFlag = 0;
	}

	/*************************************************************************
	* Function Name : ~LayerCom
	* Description   : deconstruct function
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	LayerCom::~LayerCom()
	{

	}

	/*************************************************************************
	* Function Name : setQuantize
	* Description   : set layer quantize value
	* Parameters    : iQuantize -- quantizer enable flag
	*   			  fIScale -- scale value
	* Returns       : NULL
	**************************************************************************/
	void LayerCom::setQuantize(int iQuantize,float fIScale)
	{

	}

	/*************************************************************************
	* Function Name : setFp32
	* Description   : set fp32 translate value
	* Parameters    : iFp32 -- fp32 enable flag
	* 				  uiBiasAddr -- bias address
	* 				  uiWScaleAddr -- weight scale address
	* 				  fIScale -- input scale
	* Returns       : NULL
	**************************************************************************/
	void LayerCom::setFp32(int iFp32,unsigned long long uiBiasAddr,unsigned long long uiWScaleAddr,float fIScale)
	{
		return;
	}

	/*************************************************************************
	* Function Name : getFp32Infor
	* Description   : get fp32 information
	* Parameters    : uiBias -- output bias address
	* 				  uiWScale -- output weight scale address
	* 				  fIScale -- output scale
	* Returns       : 0 -- success
	**************************************************************************/
	int LayerCom::getFp32Infor(unsigned long long *uiBias,unsigned long long *uiWScale,float *fIScale)
	{
		return 0;
	}

	/*************************************************************************
	* Function Name : getQuantizeInfor
	* Description   : get quantize information
	* Parameters    : fIScale -- output scale
	* Returns       : 0 -- success
	**************************************************************************/
	int LayerCom::getQuantizeInfor(float *fIScale)
	{
		return 0;
	}

	/*************************************************************************
	* Function Name : writeDDRInfoWeight
	* Description   : write DDR Inforamation to file
	* Parameters    : fileOutFp -- output file
	* Returns       : NULL
	**************************************************************************/
	void LayerCom::writeDDRInfoWeight(FILE *fileOutFp)
	{

	}
}
