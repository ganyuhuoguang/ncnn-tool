/*
 * boardconfig.cpp
 *
 *  Created on: Jun 13, 2019
 *      Author: doyle
 */

#include "boardconfig.h"
#include <log.h>

namespace tmnet
{
	/*************************************************************************
	* Function Name : BoardConfig
	* Description   : construct function ,initialize all the variable
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	BoardConfig::BoardConfig()
	{

	}

	/*************************************************************************
	* Function Name : ~BoardConfig
	* Description   : deconstruct function
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	BoardConfig::~BoardConfig()
	{

	}

	/*************************************************************************
	* Function Name : readDDRConfigFile
	* Description   : deconstruct function
	* Parameters    : cFilePath --
	* Returns       : void
	**************************************************************************/
	void BoardConfig::readDDRConfigFile(const char* cFilePath)
	{
		char cReadName[READ_BUF_SIZE];
		int uiReadValue = 0;
		long long int ullAddress = 0;
		int cPlace = 0;
		int iTotalNumber = 0;
		int iScanNum = 0;
		FILE *fileBoard = NULL;
	
		fileBoard = fopen(cFilePath,"rb");

		if(fileBoard == NULL)
		{
			tmtool_log(LOG_ERROR, "open boardconfig file failed!");
			return;
		}

		while(1)
		{
			BoardData *pBoard = new BoardData();
			//board name
			iScanNum = fscanf(fileBoard, "BOARD=%256s ",cReadName);
			if(iScanNum != 1)
			{
				break;
			}
			pBoard->strName = cReadName;

			//PS ddr size
			iScanNum = fscanf(fileBoard, "ADD=%d ",&cPlace);
			if(iScanNum != 1)
			{
				break;
			}
			pBoard->cAddrBits = cPlace;

			//PS ddr size
			iScanNum = fscanf(fileBoard, "PS=%d ",&uiReadValue);
			if(iScanNum != 1)
			{
				break;
			}
			pBoard->uiPsSize = uiReadValue;

			//PL ddr size
			iScanNum = fscanf(fileBoard, "PL=%d ",&uiReadValue);
			if(iScanNum != 1)
			{
				break;
			}
			pBoard->uiPlSize = uiReadValue;

			//Ps ddr start address
			iScanNum = fscanf(fileBoard, "SA=%x ",&ullAddress);
			if(iScanNum != 1)
			{
				break;
			}
			pBoard->ullPsStart = ullAddress;

			//PL ddr start address
			iScanNum = fscanf(fileBoard, "LA=%x ",&ullAddress);
			if(iScanNum != 1)
			{
				break;
			}
			pBoard->ullPlStart = ullAddress;

			//weight data store place
			iScanNum = fscanf(fileBoard, "W=%d ",&cPlace);
			if(iScanNum != 1)
			{
				break;
			}
			pBoard->cWeightPlace = cPlace;

			//inout data store place
			iScanNum = fscanf(fileBoard, "D=%d ",&cPlace);
			if(iScanNum != 1)
			{
				break;
			}
			pBoard->cDataPlace = cPlace;

			//input output data buffer number
			iScanNum = fscanf(fileBoard, "BUF=%d ",&cPlace);
			if(iScanNum != 1)
			{
				break;
			}
			pBoard->cBufferNum = cPlace;

			vBoardData.push_back(pBoard);
			iTotalNumber++;
		}
	}

	/*************************************************************************
	* Function Name : printAllConfigData
	* Description   : deconstruct function
	* Parameters    : void
	* Returns       : void
	**************************************************************************/
	void BoardConfig::printAllConfigData(void)
	{
		int iBoardDataNum = 0;
		iBoardDataNum = vBoardData.size();
		BoardData* pBoard = NULL;

		printf("there are %d board data\n",iBoardDataNum);

		for(int i=0; i<iBoardDataNum; i++)
		{
			pBoard = vBoardData[i];
			printf("name=%s \
					PSsize=%d \
					PLsize=%d \
					PSaddr=%llx \
					PLaddr=%llx \
					Weight=%d \
					Data=%d\n", pBoard->strName.c_str(),
								pBoard->uiPsSize,
								pBoard->uiPlSize,
								pBoard->ullPsStart,
								pBoard->ullPlStart,
								pBoard->cWeightPlace,
								pBoard->cDataPlace);
		}
	}
}
