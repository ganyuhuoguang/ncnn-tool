/*
 * caffe2ncnn.h
 *
 *  Created on: Jun 17, 2019
 *      Author: xiaolong.lu
 */


int caffe2ncnn(int argc, char *argv[]);
int caffe2ncnn(const char* caffeproto, const char* caffemodel, const char* ncnn_prototxt, const char* ncnn_modelbin, const char* table, int quantizelevel);


