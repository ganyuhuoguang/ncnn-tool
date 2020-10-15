/*
 * ncnn.h
 *
 *  Created on: Jun 17, 2019
 *      Author: xiaolong.lu
 */
#ifndef TMTOOL_H_
#define TMTOOL_H_

int ncnn_run(int argc, char *argv[]);
int ncnn_run(const char* ncnnparam, const char* ncnnbin, const char* mtcnnpath, const char* image1, const char* image2);
#endif