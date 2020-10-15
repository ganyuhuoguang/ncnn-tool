#ifndef	_LOG_H_
#define _LOG_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <sys/time.h>
#include <time.h>
#include <pthread.h>
#include <errno.h>
#include <string.h>
#include "toolError.h"

#define LOG_FILE0_NAME			"tmtool0.log"
#define LOG_FILE1_NAME			"tmtool1.log"

#define RESET           0
#define BRIGHT          1
#define DIM             2
#define UNDERLINE       4
#define BLINK           5
#define REVERSE         7
#define HIDDEN          8
 
#define BLACK           0
#define RED             1
#define GREEN           2
#define YELLOW          3
#define BLUE            4
#define MAGENTA         5
#define CYAN            6
#define WHITE           7

enum log_level{
	LOG_FATAL = 0,
	LOG_ERROR,
	LOG_WARNING,
	LOG_COMMON,
	LOG_INFO,
	LOG_DEBUG,
};

enum log_out_method_e{
	LOG_OUTPUT_UART = 0,
	LOG_OUTPUT_FILE,
};

extern enum log_level log_level;

int tmtool_log_init(enum log_level level, unsigned char log_out);
void tmtool_log_real(enum log_level level, const char *file, const int line, const char *func, const char *fmt, ...);
void tmtool_data_log_real(enum log_level level, const char *fmt, ...);
int tmtool_log_quit(void);

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define tmtool_log(level, ...)	\
		tmtool_log_real(level, __FILENAME__, __LINE__, __FUNCTION__, __VA_ARGS__)

#define tmtool_data_log(level, ...)	\
		tmtool_data_log_real(level, __VA_ARGS__)
#endif
