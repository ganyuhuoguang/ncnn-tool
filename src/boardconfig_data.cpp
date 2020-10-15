/*
 * boardconfig_data.cpp
 *
 *  Created on: Jun 14, 2019
 *      Author: doyle
 */

#include "boardconfig_data.h"

namespace tmnet
{
	/*************************************************************************
	* Function Name : BoardData
	* Description   : construct function ,initialize all the variable
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	BoardData::BoardData()
	{
		//address bits
		cAddrBits = 32;
		//ps ddr size
		uiPsSize = 0;
		//pl ddr size
		uiPlSize = 0;
		//ps ddr start address
		ullPsStart = 0;
		//pl ddr start address
		ullPlStart = 0;

		//0-ps 1-pl
		//weight stored place
		cWeightPlace = 1;
		//data stored place
		cDataPlace = 1;

		//input output data buffer number
		cBufferNum = 0;
	}

	/*************************************************************************
	* Function Name : ~BoardData
	* Description   : deconstruct function
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	BoardData::~BoardData()
	{

	}
}


