/*************************************************************************
 *                    All Rights True Micro Reserved                     *
 *************************************************************************
 *************************************************************************
 * Filename	   : log.cpp
 * Description : printf log
 * Version	   : 1.0
 * History     :
   MengXiangsheng 2019-08-10  Create
*************************************************************************/
#include "log.h"

/****************************************************************************************
* Macro Definitions
****************************************************************************************/
#define LOG_BUFF_SIZE		1024
#define LOG_FILE_MAX_SIZE   10 * 1024 * 1024

/****************************************************************************************
* Module Variable Definitions
****************************************************************************************/
enum log_level log_level = LOG_DEBUG;
static unsigned char log_out_method = 0;
static FILE *f_log = NULL;
pthread_mutex_t log_lock;

/******************************************************************************************
 * Function Name : tmtool_log_init
 * Description   :
 * Parameters    :
 *
 * returns       :
*******************************************************************************************/
int tmtool_log_init(enum log_level level, unsigned char log_out)
{
	int ret = TmToolSuccess;
	log_out_method = log_out;
	if (log_out_method == LOG_OUTPUT_FILE) {
		f_log = fopen(LOG_FILE0_NAME, "a");
		if(!f_log){
			printf("F[%s], L[%d], FUNC[%s]: ERROR, open log file[%s] fail[%d]\n", __FILE__,
				__LINE__, __FUNCTION__, LOG_FILE0_NAME, errno);
			ret = TmToolError_Failure;
		}
	}
	log_level = level;
	pthread_mutex_init(&log_lock, NULL);

	return ret;
}

/******************************************************************************************
 * Function Name : get_time_ms
 * Description   :
 * Parameters    :
 *
 * returns       :
*******************************************************************************************/
int64_t get_time_ms(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000 + (int64_t)(tv.tv_usec /1000);
}

void tool_vprint(char* fmt, va_list va_args)
{
	char buffer[LOG_BUFF_SIZE] = {0};
	vsnprintf(buffer, LOG_BUFF_SIZE-1, fmt, va_args);
	printf("%s\n", buffer);
}

void _print_color(int attr, int color, va_list vp, const char *fmt,...){
	char buffer[LOG_BUFF_SIZE] = {0};
	snprintf(buffer, LOG_BUFF_SIZE-1, "\x1b[%d;%dm%s\x1b[0m", attr, color + 30, fmt);
	tool_vprint(buffer, vp); 
}

/******************************************************************************************
 * Function Name : tmtool_log_real
 * Description   :
 * Parameters    :
 *
 * returns       :
*******************************************************************************************/
void tmtool_log_real(enum log_level level, const char *file,
	const int line, const char *func, const char *fmt, ...)
{
	char *log_buff = NULL;
	va_list vp;
	va_list aq;
	long int curtime;

	if(level > log_level)
		return;

	if (pthread_mutex_trylock(&log_lock) != 0)
		return;

	log_buff = (char *)calloc(1, LOG_BUFF_SIZE);
	if(!log_buff)
	{
		printf("L[%d], FUNC[%s]: ERROR, calloc log_buff fail\n", __LINE__, __FUNCTION__);
		return;
	}

	curtime = get_time_ms();
	sprintf(log_buff, "[%s][%d]: %s\n", file, line, fmt);

	va_start(vp, fmt);

	if (log_out_method == LOG_OUTPUT_FILE) 
	{
		va_start(aq, fmt);
		if (ftell(f_log) >= LOG_FILE_MAX_SIZE) 
		{
			fclose(f_log);
			if (rename(LOG_FILE0_NAME, LOG_FILE1_NAME)) 
			{
                remove(LOG_FILE1_NAME);
                rename(LOG_FILE0_NAME, LOG_FILE1_NAME);
            }

            f_log = fopen(LOG_FILE0_NAME, "a");
            if (NULL == f_log) 
			{
				printf("F[%s], L[%d], FUNC[%s]: ERROR, open log file[%s] fail\n", __FILE__,
				__LINE__, __FUNCTION__, LOG_FILE0_NAME);
				return;
            }
		}
		vfprintf(f_log, log_buff, aq);
		va_end(aq);
	}

	if(level == LOG_ERROR){
		_print_color(BRIGHT, RED, vp, log_buff);
	}else if(level == LOG_WARNING){
		_print_color(BRIGHT, YELLOW, vp, log_buff);
	}else{
		 vfprintf(stderr, log_buff, vp);
	}

	va_end(vp);
	pthread_mutex_unlock(&log_lock);
	if (log_buff)
		free(log_buff);

	return;
}

/******************************************************************************************
 * Function Name : tmtool_data_log_real
 * Description   :
 * Parameters    :
 *
 * returns       :
*******************************************************************************************/
void tmtool_data_log_real(enum log_level level, const char *fmt, ...)
{
	char *log_buff = NULL;
	va_list vp;
	va_list aq;

	if(level > log_level)
		return ;

	if (pthread_mutex_trylock(&log_lock) != 0)
		return;

	log_buff = (char *)calloc(1, LOG_BUFF_SIZE);
	if(!log_buff)
	{
		printf("F[%s], L[%d], FUNC[%s]: ERROR, calloc log_buff fail\n", __FILE__, __LINE__, __FUNCTION__);
		return ;
	}

	sprintf(log_buff, "%s", fmt);

	va_start(vp, fmt);

	if (log_out_method == LOG_OUTPUT_FILE) 
	{
		va_start(aq, fmt);
		if (ftell(f_log) >= LOG_FILE_MAX_SIZE) 
		{
			fclose(f_log);
			if (rename(LOG_FILE0_NAME, LOG_FILE1_NAME)) 
			{
                remove(LOG_FILE1_NAME);
                rename(LOG_FILE0_NAME, LOG_FILE1_NAME);
            }

            f_log = fopen(LOG_FILE0_NAME, "a");
            if (NULL == f_log) 
			{
				printf("F[%s], L[%d], FUNC[%s]: ERROR, open log file[%s] fail\n", __FILE__,
				__LINE__, __FUNCTION__, LOG_FILE0_NAME);
				return;
            }
		}
		vfprintf(f_log, log_buff, aq);
		va_end(aq);
	}
	vfprintf(stderr, log_buff, vp);

	va_end(vp);
	pthread_mutex_unlock(&log_lock);
	if (log_buff)
		free(log_buff);

	return;
}

/******************************************************************************************
 * Function Name : decode_board_log_quit
 * Description   :
 * Parameters    :
 *
 * returns       :
*******************************************************************************************/
int tmtool_log_quit(void)
{
	fclose(f_log);
	return TmToolSuccess;
}
