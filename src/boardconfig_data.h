/*
 * boardconfig_data.h
 *
 *  Created on: Jun 14, 2019
 *      Author: doyle
 */

#ifndef BOARDCONFIG_DATA_H_
#define BOARDCONFIG_DATA_H_

#include <stdio.h>
#include <string>
#include <vector>
#include <stdlib.h>
#include <string.h>

#define DDR_4K	0x1000
#define DDR_1K	1024
#define MB_TO_BYTE	DDR_1K*DDR_1K
#define PL_NO_USE	16
#define READ_BUF_SIZE 256

namespace tmnet
{
	class BoardData
	{
	public:
		BoardData();
		~BoardData();
		//board name
		std::string strName;

		//address bits
		char cAddrBits;

		//ps ddr size
		unsigned int uiPsSize;
		//pl ddr size
		unsigned int uiPlSize;
		//ps ddr start address
		unsigned long long ullPsStart;
		//pl ddr start address
		unsigned long long ullPlStart;

		//0-ps 1-pl
		//weight stored place
		char cWeightPlace;
		//data stored place
		char cDataPlace;

		//input output data buffer number
		char cBufferNum;
	};
}
#endif /* BOARDCONFIG_DATA_H_ */
