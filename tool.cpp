#include <string>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <sys/stat.h> 
#include <unistd.h>
#include <getopt.h>
#include <stdarg.h>
#include "network.h"
#include "caffe2ncnn.h"
#include "tmtool.h"
#include "benchncnn.h"
#include "mxnet2ncnn.h"
#include "darknet2ncnn.h"
#include "convert_verify.h"
#include <log.h>

#define X86 "X86"
#define RK "RK"
#define ZYNQ "ZYNQ"

static void print_usage(bool help)
{
    FILE* stream = (help) ? stdout : stderr;

    fprintf(stream, "Usage: tmtool [--op ] [--proto ] [--model ] [--param ] [--bin ] [--mtcnn ] [--channel ] [--config ] [--platform ] [--target ] [--table ] [--image1 ] [--image2 ] [--threshod ].\n");
    if (!help) {
        return;
    }

    fprintf(stream, "\t--op       : Options configure.\n");
	fprintf(stream, "\t  caffe2tmcnn  : Transform caffe model into ncnn model.\n");
	fprintf(stream, "\t  mxnet2tmcnn  : Transform mxnet model into ncnn model.\n");
	fprintf(stream, "\t  darknet2tmcnn: Transform darknet model into ncnn model.\n");
	fprintf(stream, "\t  tmcmd        : Transform ncnn model into tm model.\n");
	fprintf(stream, "\t--proto    : Input model param name.\n");
	fprintf(stream, "\t--model    : Input model name.\n");
	fprintf(stream, "\t--param    : Input ncnn model param name.\n");
	fprintf(stream, "\t--bin      : Input ncnn model name.\n");
	fprintf(stream, "\t--mtcnn    : Input mtcnn model path.\n");
    fprintf(stream, "\t--channel  : Channel number.\n");
    fprintf(stream, "\t--config   : The setting file of different hardware platforms.\n");
    fprintf(stream, "\t--platform : Device name.\n");
    fprintf(stream, "\t--target   : Target platform type,eg:ZC706、ZCU102、ZC7100、ZC7020、AXZ7100\n");
	fprintf(stream, "\t--table    : Input model table with qn reference.\n");
	fprintf(stream, "\t--threshod : Input quantitation threshold.\n");
	fprintf(stream, "\t--iamge1   : Input image name.\n");
	fprintf(stream, "\t--image2   : Input image name.\n");
	fprintf(stream, "\t--help     : Show help usage.\n\n");

    fprintf(stream, "caffe2tmcnn example: ./tmtool --op caffe2tmcnn --proto caffe.prototxt --model caffe.caffemodel --param ncnn.param --bin ncnn.bin --table ncnn.table --threshod 256\n\n");
	fprintf(stream, "mxnet2tmcnn example: ./tmtool --op mxnet2tmcnn --proto mxnet.json --model mxnet.params --param ncnn.param --bin ncnn.bin\n\n");
	fprintf(stream, "darknet2tmcnn example: ./tmtool --op darknet2tmcnn --proto darknet.prototxt --model darknet.caffemodel --param ncnn.param --bin ncnn.bin\n\n");
	fprintf(stream, "ncnn2tmcnn example: ./tmtool --op tmcmd --param ncnn.param --bin ncnn.bin --channel 8 --config board.ini --platform ZC706 --target ZYNQ\n\n");
	fprintf(stream, "run example: ./tmtool --op run --param ncnn.param --bin ncnn.bin --mtcnn mtcnn image1 image2\n\n");
    return;
}

/**
 * show truemicro copyright info
 */
void printCopyright(){
	fprintf(stderr, "\n");
	fprintf(stderr, "\t****TmTool, a model and command transformation tool****\n");
	fprintf(stderr, "\t\tAll rights to interpret by TrueMicro.\n");
	fprintf(stderr, "\n");
}

int main(int iArgc, char **cArgv)
{
	const char *pcProto = NULL;
	const char *pcModel = NULL;
	const char *pcNcnnParam = NULL;
	const char *pcNcnnBin = NULL;
	const char *pcChannelNum = NULL;  //ChannelNum
	const char *pcBoardConfig = NULL; //board.ini
	const char *pcBoardName = NULL;   //ZC706,AXZ7100,ZCU102
    const char *pcArchType = NULL;	//X86,RK,ZYNQ
	const char *pcTable = NULL;
	const char *pcOption = NULL;
	const char *pcMtcnn = NULL;
	const char *pcImage1 = NULL;
	const char *pcImage2 = NULL;

	int iThreshod;
	int iRes = -1;
	int status = 0;
	int option;
	int caffe2tmcnnflag = 0;
	int mxnet2tmcnnflag = 0;
	int darknet2tmcnnflag = 0;
	int tmcmdflag = 0;
	int runflag = 0;
	int c;

	tmnet::Network nNetWork;

    struct option longopts[]={
        {"op",      1,NULL,'o'},
        {"config",  1,NULL,'c'},
        {"platform",1,NULL,'p'},
        {"target",  1,NULL,'d'},
		{"channel", 1,NULL,'n'},
		{"proto",   1,NULL,'f'},
		{"model",   1,NULL,'m'},
		{"param",   1,NULL,'g'},
		{"bin",     1,NULL,'b'},
		{"mtcnn",   1,NULL,'q'},
		{"table",   1,NULL,'a'},
		{"threshod",1,NULL,'r'},
		{"image1",  1,NULL,'i'},
		{"image2",  1,NULL,'k'},
		{"tmcnn",   0,NULL,'e'},
		{"tmcmd",   0,NULL,'t'},
		{"help",    0,NULL,'h'},
        {0,0,0,0},
	};

	/* show license to terminate */
	printCopyright();

	if(iArgc < 2){
		print_usage(true);
		exit(0);
	}

	while((option = getopt_long(iArgc, cArgv, "o:c:p:d:n:f:m:g:b:q:a:r:i:k:eth", longopts, NULL)) != -1){
        switch (option)
        {
            case 'o':  //op
                if (optarg != NULL) {
					pcOption = optarg;
					if(strcmp(pcOption,"caffe2tmcnn") == 0)
					{
						caffe2tmcnnflag = 1;
					}
					else if(strcmp(pcOption,"mxnet2tmcnn") == 0)
					{
						mxnet2tmcnnflag = 1;
					}
					else if(strcmp(pcOption,"darknet2tmcnn") == 0)
					{
						darknet2tmcnnflag = 1;
					}
					else if(strcmp(pcOption,"tmcmd") == 0)
					{
						tmcmdflag = 1;
					}
					else if(strcmp(pcOption,"run") == 0)
					{
						runflag = 1;
					}
				}
                break;
            case 'c':  //config
                if (optarg != NULL) {
					pcBoardConfig = optarg;
				}
                break;
            case 'p':  //platform
                if (optarg != NULL) {
					pcBoardName = optarg;
				}
                break;
			case 'd':  //target
                if (optarg != NULL) {
					pcArchType = optarg;
				}
                break;
			case 'n':  //channel
                if (optarg != NULL) {
					pcChannelNum = optarg;
				}
                break;
			case 'b':  //bin
                if (optarg != NULL) {
					pcNcnnBin = optarg;
				}
                break;
			case 'g':  //param
                if (optarg != NULL) {
					pcNcnnParam = optarg;
				}
                break;
			case 'f':  //prototxt
                if (optarg != NULL) {
					pcProto = optarg;
				}
                break;
			case 'm':  //model
                if (optarg != NULL) {
					pcModel = optarg;
				}
                break;
			case 'a':  //table
                if (optarg != NULL) {
					pcTable = optarg;
				}
                break;
			case 'r':  //threshod
                if (optarg != NULL) {
					iThreshod = atoi(optarg);
				}
                break;
			case 'q':  //mtcnn
                if (optarg != NULL) {
					pcMtcnn = optarg;
				}
                break;
			case 'i':  //image1
                if (optarg != NULL) {
					pcImage1 = optarg;
				}
                break;
			case 'k':  //image2
                if (optarg != NULL) {
					pcImage2 = optarg;
				}
                break;
			case 'e':  //tmcnn
                if (optarg != NULL) {
					caffe2tmcnnflag = 1;
				}
                break;
			case 't':  //tmcmd
                if (optarg != NULL) {
					tmcmdflag = 1;
				}
                break;
            case 'h':
                print_usage(true);
                exit(0);
            case '?':
				tool_log(LOG_COMMON, "unknow option: %c", optopt);
				break;
            default:
                print_usage(true);
                return -EINVAL;
        }
    }

	if(caffe2tmcnnflag == 1)
	{
		iRes = caffe2ncnn(pcProto, pcModel, pcNcnnParam, pcNcnnBin, pcTable, iThreshod);
	}

	if(mxnet2tmcnnflag == 1)
	{
		iRes = mxnet2ncnn(pcProto, pcModel, pcNcnnParam, pcNcnnBin);
	}

	if(darknet2tmcnnflag == 1)
	{
		iRes = darknet2ncnn((char*)pcProto, (char*)pcModel, (char*)pcNcnnParam, (char*)pcNcnnBin);
	}

	if(runflag == 1)
	{
		//iRes = ncnn_run(pcNcnnParam, pcNcnnBin, pcMtcnn, pcImage1, pcImage2);
	}

	if (tmcmdflag == 1)
	{
		/* checkout board configuration if validate */
		if(pcBoardConfig == NULL){
			print_usage(true);
			exit(0);
		}
    	status = mkdir("./data",0777);
		//read config file
		iRes = nNetWork.getBoardInformation(pcBoardConfig, pcBoardName); //获取板子信息，ddr内存等，要保存数据到ps还是pl，最大可用内存等等。
		if (iRes < 0)
		{
			tool_log(LOG_ERROR, "Board Information Error!");
			return -1;
		}
		//load param file and bin file
		nNetWork.loadParamFile(pcNcnnParam);

		nNetWork.loadBinFile(pcNcnnBin, pcChannelNum);
		
		//print layer construct
		nNetWork.printConstruct();
		nNetWork.printAllDDRInfo();

		tool_log(LOG_COMMON, "Translate ncnn model into FPGA model successfully!\n");

		if (!memcmp(pcArchType, X86, sizeof(X86)))
		{
			status = system("objcopy -I binary -O elf64-x86-64 -B i386 model.bin tmmodel.elf");
		}
		else if (!memcmp(pcArchType, RK, sizeof(RK)))
		{
			status = system("aarch64-linux-gnu-objcopy -I binary -O elf64-littleaarch64 -B aarch64 model.bin model.elf");
		}
		else if (!memcmp(pcArchType, ZYNQ, sizeof(ZYNQ)))
		{
			status = system("arm-linux-gnueabihf-objcopy -I binary -O elf32-littlearm -B arm tmmodel.bin model.elf");
		}
		else
		{
			tool_log(LOG_ERROR, "do not support this arch type!");
		}
	}

	return status;
}
