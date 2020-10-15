/*
 * boardconfig.h
 *
 *  Created on: Jun 13, 2019
 *      Author: doyle
 */

#ifndef BOARDCONFIG_H_
#define BOARDCONFIG_H_

#include <stdio.h>
#include <string>
#include <vector>
#include <stdlib.h>
#include <string.h>
#include "boardconfig_data.h"

namespace tmnet
{
	class BoardConfig
	{
	public:
		BoardConfig();
		~BoardConfig();

		//board data
		std::vector<BoardData*> vBoardData;

		//read config file
		void readDDRConfigFile(const char* cFilePath);
		//print all config data
		void printAllConfigData(void);
	};
}
#endif /* BOARDCONFIG_H_ */
