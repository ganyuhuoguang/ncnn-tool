// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "layer_type.h"
#include "layer.h"

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types_c.h>

#include "platform.h"
#include "net.h"
#include "tmtool.h"
#include "mtcnn.h"
#include <log.h>

#if TMTOOL_VULKAN
#include "gpu.h"
#endif // TMTOOL_VULKAN

using namespace tmtool;
using namespace std;


#define _MSC_VER 1900
#define CHANNEL  8

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

std::vector<std::string> splitString_1(const std::string &str,
	const char delimiter) {
	std::vector<std::string> splited;
	std::string s(str);
	size_t pos;

	while ((pos = s.find(delimiter)) != std::string::npos) {
		std::string sec = s.substr(0, pos);

		if (!sec.empty()) {
			splited.push_back(s.substr(0, pos));
		}

		s = s.substr(pos + 1);
	}

	splited.push_back(s);

	return splited;
}

float simd_dot_1(const float* x, const float* y, const long& len) {
	float inner_prod = 0.0f;
	__m128 X, Y; // 128-bit values
	__m128 acc = _mm_setzero_ps(); // set to (0, 0, 0, 0)
	float temp[4];

	long i;
	for (i = 0; i + 4 < len; i += 4) {
		X = _mm_loadu_ps(x + i); // load chunk of 4 floats
		Y = _mm_loadu_ps(y + i);
		acc = _mm_add_ps(acc, _mm_mul_ps(X, Y));
	}
	_mm_storeu_ps(&temp[0], acc); // store acc into an array
	inner_prod = temp[0] + temp[1] + temp[2] + temp[3];

	// add the remaining values
	for (; i < len; ++i) {
		inner_prod += x[i] * y[i];
	}
	return inner_prod;
}

float CalcSimilarity_1(const float* fc1, const float* fc2, long dim) 
{
	return simd_dot_1(fc1, fc2, dim)
		/ (sqrt(simd_dot_1(fc1, fc1, dim))
		* sqrt(simd_dot_1(fc2, fc2, dim)));
}

int test_picture() {
	char *model_path = "../models/run";
	MTCNN mtcnn(model_path);

	clock_t start_time = clock();

	cv::Mat image;
	image = cv::imread("./sample.jpg");
	tmtool::Mat ncnn_img = tmtool::Mat::from_pixels(image.data, tmtool::Mat::PIXEL_BGR2RGB, image.cols, image.rows);
	std::vector<Bbox> finalBbox;

#if(MAXFACEOPEN==1)
	mtcnn.detectMaxFace(ncnn_img, finalBbox);
#else
	mtcnn.detect(ncnn_img, finalBbox);
#endif

	const int num_box = finalBbox.size();
	std::vector<cv::Rect> bbox;
	bbox.resize(num_box);
	for (int i = 0; i < num_box; i++) {
		bbox[i] = cv::Rect(finalBbox[i].x1, finalBbox[i].y1, finalBbox[i].x2 - finalBbox[i].x1 + 1, finalBbox[i].y2 - finalBbox[i].y1 + 1);

		for (int j = 0; j<5; j = j + 1)
		{
	//		cv::circle(image, cvPoint(finalBbox[i].ppoint[j], finalBbox[i].ppoint[j + 5]), 2, CV_RGB(0, 255, 0), CV_FILLED);
		}
	}
	for (vector<cv::Rect>::iterator it = bbox.begin(); it != bbox.end(); it++) {
		rectangle(image, (*it), cv::Scalar(0, 0, 255), 2, 8, 0);
	}

	imshow("face_detection", image);
	clock_t finish_time = clock();
	double total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;
	std::cout << "time" << total_time * 1000 << "ms" << std::endl;

	cv::waitKey(0);
	return 1;
}

cv::Mat getsrc_roi(std::vector<cv::Point2f> x0, std::vector<cv::Point2f> dst)
{
	int size = dst.size();
	cv::Mat A = cv::Mat::zeros(size * 2, 4, CV_32FC1);
	cv::Mat B = cv::Mat::zeros(size * 2, 1, CV_32FC1);

	//[ x1 -y1 1 0] [a]       [x_1]
	//[ y1  x1 0 1] [b]   =   [y_1]
	//[ x2 -y2 1 0] [c]       [x_2]
	//[ y2  x2 0 1] [d]       [y_2]	

	for (int i = 0; i < size; i++)
	{
		A.at<float>(i << 1, 0) = x0[i].x;// roi_dst[i].x;
		A.at<float>(i << 1, 1) = -x0[i].y;
		A.at<float>(i << 1, 2) = 1;
		A.at<float>(i << 1, 3) = 0;
		A.at<float>(i << 1 | 1, 0) = x0[i].y;
		A.at<float>(i << 1 | 1, 1) = x0[i].x;
		A.at<float>(i << 1 | 1, 2) = 0;
		A.at<float>(i << 1 | 1, 3) = 1;

		B.at<float>(i << 1) = dst[i].x;
		B.at<float>(i << 1 | 1) = dst[i].y;
	}

	cv::Mat roi = cv::Mat::zeros(2, 3, A.type());
	cv::Mat AT = A.t();
	cv::Mat ATA = A.t() * A;
	cv::Mat R = ATA.inv() * AT * B;

	//roi = [a -b c;b a d ];

	roi.at<float>(0, 0) = R.at<float>(0, 0);
	roi.at<float>(0, 1) = -R.at<float>(1, 0);
	roi.at<float>(0, 2) = R.at<float>(2, 0);
	roi.at<float>(1, 0) = R.at<float>(1, 0);
	roi.at<float>(1, 1) = R.at<float>(0, 0);
	roi.at<float>(1, 2) = R.at<float>(3, 0);
	return roi;

}


cv::Mat faceAlign(cv::Mat image, MTCNN *mtcnn)
{
	double dst_landmark[10] = {
		38.2946, 73.5318, 55.0252, 41.5493, 70.7299,
		51.6963, 51.5014, 71.7366, 92.3655, 92.2041 };
	vector<cv::Point2f>coord5points;
	vector<cv::Point2f>facePointsByMtcnn;
	for (int i = 0; i < 5; i++) {
		coord5points.push_back(cv::Point2f(dst_landmark[i], dst_landmark[i + 5]));
	}
	char *model_path = "../models/run";
	(model_path);
	clock_t start_time = clock();

	tmtool::Mat ncnn_img = tmtool::Mat::from_pixels(image.data, tmtool::Mat::PIXEL_BGR2RGB, image.cols, image.rows);
	std::vector<Bbox> finalBbox;

#if(MAXFACEOPEN==1)
	mtcnn->detectMaxFace(ncnn_img, finalBbox);
#else
	mtcnn->detect(ncnn_img, finalBbox);
#endif

    const int num_box = finalBbox.size(); //������������Ĭ��һ������
	std::vector<cv::Rect> bbox;
	bbox.resize(num_box);
	for (int i = 0; i < num_box; i++) {
		for (int j = 0; j<5; j = j + 1)
		{
			//cv::circle(image, cvPoint(finalBbox[i].ppoint[j], finalBbox[i].ppoint[j + 5]), 2, CV_RGB(0, 255, 0), CV_FILLED);
			facePointsByMtcnn.push_back(cvPoint(finalBbox[i].ppoint[j], finalBbox[i].ppoint[j + 5]));
		}
	}

	cv::Mat warp_mat = cv::estimateRigidTransform(facePointsByMtcnn, coord5points, false);
	if (warp_mat.empty()) {
		warp_mat = getsrc_roi(facePointsByMtcnn, coord5points);
	}
	warp_mat.convertTo(warp_mat, CV_32FC1);
	cv::Mat alignFace = cv::Mat::zeros(112, 112, image.type());
	warpAffine(image, alignFace, warp_mat, alignFace.size());
	return alignFace;
}

static cv::Mat img_prepare(cv::Mat img, int img_w, int img_h, int w, int h)
{    
    int new_w = img_w * std::min(float(w)/float(img_w), float(h)/float(img_h));
    int new_h = img_h * std::min(float(w)/float(img_w), float(h)/float(img_h));
    cv::Mat resize; 
    cv::resize(img, resize, cv::Size(new_w, new_h), 0, 0, 2);
    cv::Mat canvas = cv::Mat::zeros(w, h, img.type());
	canvas.setTo(cv::Scalar(128, 128, 128));
    for (int i = 0; i < new_h; i++) 
    {
        for (int j = 0; j < new_w; j++)
        {
            unsigned char * wc = canvas.ptr(i+(h-new_h)/2, j+(w-new_w)/2);
            unsigned char * rc = resize.ptr(i, j);
            wc[0] = rc[0];
            wc[1] = rc[1];
            wc[2] = rc[2];
        } 
    }
    return canvas;
}

static char** label_load(char* fn)
{
    char** labels = (char**)malloc(1 * sizeof(char *));
    FILE* fp = fopen(fn, "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);

    int index = 0;
    char* line = NULL;
    size_t len = 0;
    while ((getline(&line, &len, fp)) != -1) 
    {
        labels = (char**)realloc(labels, (index+1)*sizeof(char*));
        labels[index] = (char*)malloc(strlen(line)*sizeof(char));
        strcpy(labels[index++], line);
    }
    fclose(fp);

    if (line) free(line);
    
    return labels;
}

static void draw_objects(const cv::Mat& bgr, int res, std::vector<tmtool::Mat> result, char** class_names)
{
    
    float factor = std::min(1.f, float(res)/float(bgr.cols));
   
    std::vector<Object> objects;

    for (int i=0; i<result[0].h; i++)
    {
        const float* values = result[0].row(i);

        Object object;
        object.label = values[0];
        object.prob = values[1];
        object.rect.x = int((values[2] - (res-factor*bgr.cols)/2.f) / factor);
        object.rect.y = int((values[3] - (res-factor*bgr.rows)/2.f) / factor);
        object.rect.width = int((values[4] - (res-factor*bgr.cols)/2.f) / factor) - object.rect.x;
        object.rect.height = int((values[5] - (res-factor*bgr.rows)/2.f) / factor) - object.rect.y;
        objects.push_back(object);
    }
    
    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob, obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y),
                                      cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
    
    cv::imshow("image", image);
    cv::waitKey(0);
}

void feature_process(const std::vector<tmtool::Mat>& src, std::vector<tmtool::Mat>& dst, int classes, int num_box, float conf_threshold, float nms_threshold, tmtool::Mat biases, int input_size, int img_w, int img_h)
{
    tmtool::Layer* yolov3 = tmtool::create_layer(tmtool::LayerType::Yolov3DetectionModifiedOutput);
    tmtool::ParamDict pd;
    pd.set(0, classes);
    pd.set(1, num_box);
    pd.set(2, conf_threshold);
    pd.set(3, nms_threshold);
    pd.set(4, biases);
    pd.set(5, input_size);
    pd.set(6, img_w);
    pd.set(7, img_h);
	
    yolov3->load_param(pd);
    tmtool::Option opt;

    // for (int q=0; q<src[0].c; q++)
    // {
    //     const float* ptr = src[0].channel(q);
	// 	printf("----src[0]: %f= \n",ptr);
	// }

    yolov3->forward(src, dst, opt);
	
    delete yolov3;
}

static int YOLOv3_detect(const cv::Mat& bgr,const char* ncnnparam, const char* ncnnbin)
{
    int input_width = 416;
    char** labels = label_load("../models/run/coco.labels");
    int anchor_list[18] = {116,90,  156,198,  373,326,  30,61,  62,45,  59,119,  10,13,  16,30,  33,23};
    tmtool::Mat anchors;
    anchors.create(18);
    for (int i=0; i<18; i++) anchors[i] = anchor_list[i];

    Net test;
    test.load_param(ncnnparam);
    test.load_model(ncnnbin);
    
    
    cv::Mat input = img_prepare(bgr, bgr.cols, bgr.rows, input_width, input_width);
    tmtool::Mat in = tmtool::Mat::from_pixels(input.data, tmtool::Mat::PIXEL_BGR, input.cols, input.rows);
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1.0/255.0, 1.0/255.0, 1.0/255.0};
    in.substract_mean_normalize(mean_vals, norm_vals);

    float qn = 8.22200018571974f;
    FILE *a = fopen("yolo_feature.txt", "w");
    char cPrintBuf[4];

    
    Extractor ex = test.create_extractor();
    tmtool::Mat out0, out1, out2;
    std::vector<tmtool::Mat> calculated;
    std::vector<tmtool::Mat> processed;
    ex.input("data", in);
    ex.extract("out0", out0);//out0
    ex.extract("out1", out1);//out1
    ex.extract("out2", out2);

    calculated.push_back(out0);
    calculated.push_back(out1);
    calculated.push_back(out2);

    int addnum = 0;//add 0
    if(0 == out0.c % 8)
    {
        addnum = 0;
    }
    else
    {
        addnum = 8 - out0.c % 8;
    }

    char *cCharBuf = (char *)malloc((out0.c + addnum)*out0.h*out0.w);

    // //print int8
    for (int q=0; q<(out0.c + addnum); q++)
    {
        const float* ptr;
        if((addnum != 0)&&(q == out0.c + addnum -1))
        {

            for (int j=0; j<out0.w*out0.h; j++)
            {
                //cCharBuf[(q/8)*8*out0.h*out0.w+8*j+q%8] = 0x00;//8 channel reset
                cCharBuf[q*out0.h*out0.w+j] = 0x00;//no reset
            }
        }
        else
        {
            ptr = out0.channel(q);
            for (int j=0; j<out0.w*out0.h; j++)
            {
                float f = qn * ptr[j];
                signed char c;
                int int32 = round(f);
                if (int32 > 127) 
                {
                    c = 127;
                }
                else if (int32 < -128) 
                {
                    c = -128;
                }
                else
                {
                    c = (signed char)int32;
                }
                //cCharBuf[(q/8)*8*out0.h*out0.w+8*j+q%8] = c;//8 channel reset
                cCharBuf[q*out0.h*out0.w+j] = c;//no reset
            }
        }
        

    }  
	
    for (int q=0; q<(out0.c + addnum)*out0.h*out0.w/4; q++)
    { 
        memcpy(cPrintBuf,&cCharBuf[q*4],4);
        for(int k=0;k<4;k++)
        {
            fprintf(a,"%02x",cPrintBuf[3-k]&0xff);
        }
    	fprintf(a,"\n");
    }
	
    free(cCharBuf);
    fclose(a);

    feature_process(calculated, processed, 80, 3, 0.5f, 0.45f, anchors, input_width, bgr.cols, bgr.rows);
	
    draw_objects(bgr, input_width, processed, labels);

    return 0;
}

static int detect_mobilenet(const cv::Mat& bgr, std::vector<Object>& objects,const char* ncnnparam, const char* ncnnbin)
{
    tmtool::Net mobilenet;

#if NCNN_VULKAN
    mobilenet.opt.use_vulkan_compute = true;
#endif // NCNN_VULKAN

    // model is converted from https://github.com/chuanqi305/MobileNet-SSD
    // and can be downloaded from https://drive.google.com/open?id=0ByaKLD9QaPtucWk0Y0dha1VVY0U
    mobilenet.load_param(ncnnparam);
    mobilenet.load_model(ncnnbin);

    const int target_size = 300;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    tmtool::Mat in = tmtool::Mat::from_pixels_resize(bgr.data, tmtool::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size, target_size);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.0/127.5,1.0/127.5,1.0/127.5};
    in.substract_mean_normalize(mean_vals, norm_vals);

    tmtool::Extractor ex = mobilenet.create_extractor();
    // ex.set_num_threads(4);

    ex.input("data", in);

    tmtool::Mat out;
    ex.extract("detection_out",out);  //mbox_loc mbox_conf_flatten mbox_priorbox 
    FILE *a = fopen("ssd_feature.txt", "w");
    char cPrintBuf[4];
    //float *cPrintBuf = (float *)malloc(((out.c/CHANNEL+1)*CHANNEL)*out.w*out.h*4);

 
    //add 0 for permute fp32 
    // for(int outid=0;outid<(out.c/CHANNEL+1)*CHANNEL; outid++)
    // {
    //     if(outid < out.c)
    //     {
    //         const float* ptr = out.channel(outid);
    //         for(int kid = 0;kid < out.w*out.h;kid++)
    //         {
    //             cPrintBuf[(outid/CHANNEL)*CHANNEL*out.w*out.h+CHANNEL*kid+outid%CHANNEL]  = ptr[kid];
    //         }
    //     }
    //     else
    //     {
    //          for(int kid = 0;kid < out.w*out.h;kid++)
    //          {
    //             cPrintBuf[(outid/CHANNEL)*CHANNEL*out.w*out.h+CHANNEL*kid+outid%CHANNEL]  = 0.f;
    //          }
    //     }
    // }
    // for (int j=0; j<((out.c/CHANNEL+1)*CHANNEL)*out.w*out.h; j++)
    // {
    //         char pc[4];
    //         memcpy(&pc,cPrintBuf+j,4);
    //        // fprintf(a, "%f",cPrintBuf[j]);
    //         for(int k=0;k<4;k++)
    //         {
    //             fprintf(a, "%02x", pc[3-k]&0xff);
    //         }
    //         fprintf(a,"\n");
    // }



    //no add 0 for permute fp32 
    for (int q=0; q<out.c; q++)
    {
        const float* ptr = out.channel(q);
        for (int j=0; j<out.w*out.h; j++)
        {
            char *pc = (char*)&ptr[j];
            for(int i=0; i<4; i++)
            {
                //cPrintBuf[i] = *(pc+3-i);
                cPrintBuf[i] = *(pc+i);
                fprintf(a, "%02x", cPrintBuf[i]&0xff);
            }
            //if(((j+1) % 4) == 0&&j!=0)
                fprintf(a,"\n");
            //fprintf(a, "%f\n", ptr[j]);//for float
        }
    }

    //free(lastpcCharBuf);
    //free(cPrintBuf);
    fclose(a);

    printf("%d %d %d\n", out.w, out.h, out.c);
    objects.clear();
    for (int i=0; i<out.h; i++)
    {
        const float* values = out.row(i);

        Object object;
        object.label = values[0];
        object.prob = values[1];
        object.rect.x = values[2] * img_w;
        object.rect.y = values[3] * img_h;
        object.rect.width = values[4] * img_w - object.rect.x;
        object.rect.height = values[5] * img_h - object.rect.y;

        objects.push_back(object);
    }

    return 0;
}

static void draw_objects_ssd(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {"background",
        "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair",
        "cow", "diningtable", "dog", "horse",
        "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor"};

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y),
                                      cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("image", image);
    cv::waitKey(0);
}

static int detect_squeezenet(const cv::Mat& bgr, std::vector<float>& cls_scores, const char* ncnnparam, const char* ncnnbin)
{
    tmtool::Net squeezenet;

#if TMTOOL_VULKAN
    squeezenet.opt.use_vulkan_compute = true;
#endif // TMTOOL_VULKAN

    squeezenet.load_param(ncnnparam);//    ResNet-18-nbnfcnp
    squeezenet.load_model(ncnnbin);// ResNet-50-nbnfcnp

    // cv::Mat bgr0 = cv::Mat::zeros(bgr.cols, bgr.rows, bgr.type());
	// bgr0.setTo(cv::Scalar(1,1,1));
    // for resnet
    tmtool::Mat in = tmtool::Mat::from_pixels_resize(bgr.data, tmtool::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 224, 224);
    const float mean_vals[3] = {103.939f, 116.779f, 123.68f};

    in.substract_mean_normalize(mean_vals, 0);

    tmtool::Extractor ex = squeezenet.create_extractor();

    ex.input("data", in);

    tmtool::Mat out;
    //resnet18
    //conv1 res2a_branch1   res2a_branch2a  res2a_branch2b  res2b_branch2a  res2b_branch2b   res3a_branch1  res3a_branch2a  res3a_branch2b
    //res3b_branch2a   res3b_branch2b  res4a_branch1  res4a_branch2a  res4a_branch2b   res4b_branch2a   res4b_branch2b   res5a_branch1   res5a_branch2a
    //res5a_branch2b   res5b_branch2a  res5b_branch2b  fc1000

    ex.extract("prob", out);
    cls_scores.resize(out.w*out.h*out.c);
    printf("out.w: %d\nout.h: %d\nout.c: %d\n", out.w, out.h, out.c);
    FILE *a = fopen("out.txt", "w");
    char cPrintBuf[4];
    float qn = 66.2934097363f;//fc qn

    float aa[256] = {
    1766.9539940911668, 1071.9990068429852, 2036.780673153885, 1044.9586919002643, 770.4786926599533, 1132.3467416796952, 732.2585813172899, 1204.2795014240264, 
    1194.2155694957505, 1415.0173598437157, 1649.7115758193754, 1196.3653959639144, 1027.177026245434, 843.3386293008132, 1602.1520611187548, 1725.1367108218665, 
    1774.2659411103618, 2059.4148943226473, 930.9077894207043, 1471.6984474325866, 1395.5432951476412, 1046.7215720180782, 1083.6825550051983, 1578.9273685969508, 
    2146.6777141936004, 1833.3341352586424, 1008.1036403304112, 1096.7620828868223, 1452.2147968940744, 1243.2940874416997, 1309.7481227404924, 648.1691245704926, 
    1160.5883386743094, 1088.3465781124596, 1239.9343442576503, 1072.0515953775932, 1308.4174316144863, 588.7423995207623, 1766.451172626929, 582.5069860917889, 
    1248.1886729284852, 1766.7334937998783, 959.3927265445183, 1495.3714960012514, 1545.1525315012789, 1021.6339858774763, 1079.3647876094544, 1054.135696865289, 
    603.793009034906, 3006.2262317031964, 1239.6095459886292, 547.8224962254882, 1114.360492072849, 1475.689535445236, 1246.8469915708465, 1574.3552599835893, 
    1718.8018091046329, 1174.030181443924, 1628.5768880093074, 1062.151559277974, 1320.3651594629537, 1269.2463828419043, 901.7172365139573, 1378.2731690427663, 
    1070.9193864435017, 1613.0083709521969, 1615.8616143494783, 755.8257530166693, 2089.5526917362545, 876.3982481136949, 1000.937037154044, 1561.7827497998712, 
    1591.589655408098, 486.82588561511193, 1687.022306738914, 1319.4361231161406, 1294.0225243171285, 1300.2028418929863, 1272.3945539862086, 1261.8281378030965, 
    714.2302863633446, 1614.8518875044988, 916.2444034899058, 775.209817378553, 1152.7893328759774, 1836.6407358780978, 773.02816898598, 1163.843787476113, 
    772.4520482180174, 1056.715098427901, 1823.1124882122554, 1271.8653582934242, 1335.489840910809, 1728.8650352159746, 1154.5559397380732, 1519.4744660938377, 
    1133.3835732752184, 642.4686352354402, 1440.8078372217317, 1327.5854434281346, 1806.2499748067164, 1800.034854185853, 1139.4849748821518, 1078.7401110936444, 
    1369.0538553717547, 940.6166333292572, 553.4420656129602, 2182.4557310672, 1395.81744535618
    };

    for (int j=0; j<out.w*out.h*out.c; j++)
    {
        cls_scores[j] = out[j*4];

        // fprintf(a, "%f\n", out[j]);
    }

    for (int q=0; q<out.c; q++)
    {
            const float* ptr = out.channel(q);

            for (int j=0; j<out.w*out.h; j++)
            {
                // cls_scores[j] = out[j];  

                // float f = qn * out[j*4];
                // signed char c;
                // int int32 = round(f);
                // // printf(" k = %f\n",k);
                // // printf(" int32 = %d\n",int32);
                // // if (int32 > 127) 
                // // {
                // //     c = 127;
                // // }
                // // else if (int32 < -128) 
                // // {
                // //     c = -128;
                // // }
                // // else
                // // {
                // //     c = (signed char)int32;
                // // }

                // // if(((j) % 4) == 0)
                // //     fprintf(a,"0x");

                fprintf(a, "%f\n", ptr[j]);
                // fprintf(a,"%02x",c&0xff);
                // if(((j+1) % 4) == 0&&j!=0)
                //     fprintf(a,"\n");

             // fprintf(a, "%f\n", ptr[j]);

            // printf("%f\n", out[j*4]);
            }
    }   
    for (int i=0; i<out.c/8; i++)
    {
            //cls_scores[j] = out[j];
            
          // eltwise print and pooling print 
            for(int j = 0;j < out.w*out.h;j++)
            {
                for(int k=0;k<8;k++)
                {

                    float t;
                    // printf(" t = %f\n",t);
                    if(k<4)
                    {
                        memcpy(&t,&out[(i*out.w*out.h*8+(3-k)*out.w*out.h+j)],sizeof(float));  
                    }
                    else
                    {
                        memcpy(&t,&out[(i*out.w*out.h*8+(11-k)*out.w*out.h+j)],sizeof(float)); 
                    }
                    
                    float f =  t * qn *aa[i*8+k];
                    // signed char c;
                    int int32 = round(f);
                    // printf(" k = %f\n",k);
                    // printf(" int32 = %d\n",int32);
                    // if (int32 > 127) 
                    // {
                    //     c = 127;
                    // }
                    // else if (int32 < -128) 
                    // {
                    //     c = -128;
                    // }
                    // else
                    // {
                    //     c = (signed char)int32;
                    // }
                    fprintf(a,"%d\n",int32);
                    // if(((k) % 4) == 0)
                    //     fprintf(a,"0x");


                    // fprintf(a,"%02x",c&0xff);
                    // if(((k+1) % 4) == 0&&k!=0)
                    //     fprintf(a,"\n");
                }
            }
           // fprintf(a, "%f\n", out[j]);
    }


    //print  int32 
    for (int j=0; j<out.w*out.h*out.c; j++)
    {
        memcpy(cPrintBuf,&out[j*4],sizeof(cPrintBuf));
        fprintf(a,"0x");
        for(int k=0; k< 4 ;k++)
        {
            // outfile<<cPrintBuf[k];
            fprintf(a,"%02x",cPrintBuf[3-k]&0xff);
        }
        fprintf(a,"\n");
    }

    fclose(a);
    
    return 0;

}

static int print_topk(const std::vector<float>& cls_scores, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector< std::pair<float, int> > vec;
    vec.resize(size);
    for (int i=0; i<size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater< std::pair<float, int> >());

    // print topk and score
    for (int i=0; i<topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
    }

    return 0;
}

int ncnn_run(const char* ncnnparam, const char* ncnnbin, const char* mtcnnpath, const char* image1, const char* image2)
{
    const char* name1 = image1;
	const char* name2 = image2;
    char out[40];
	if (NULL != strstr(ncnnparam, "yolov3"))//output yolov2 feature
	{
		const char* imagepath = name1;
		cv::Mat m = cv::imread(imagepath, 1);
		if (m.empty())
		{
			fprintf(stderr, "cv::imread %s failed\n", imagepath);
			return -1;
		}
		YOLOv3_detect(m,ncnnparam,ncnnbin);
        tmtool_log(LOG_COMMON, "Run yolov3 model successfully!\n");
        return 0;
	}
	else if(NULL != strstr(ncnnparam, "ncnn"))//output facenet feature
	{
		FILE *file1 = fopen("../models/run/n1.txt","w");
		FILE *file2 = fopen("../models/run/n2.txt","w");
		
		MTCNN *mtcnn = new MTCNN(mtcnnpath);
		cv::Mat image1 = cv::imread(name1);
		cv::Mat alignedFace1 = faceAlign(image1, mtcnn);
		
		// sprintf(out,"out_%s", name1);
		// imwrite(out, alignedFace1);

		cv::Mat image2 = cv::imread(name2);
		cv::Mat alignedFace2 = faceAlign(image2, mtcnn);

		// sprintf(out,"out_%s", name2);
		// imwrite(out, alignedFace2);
		
		tmtool::Net squeezenet;
		squeezenet.load_param(ncnnparam);
		squeezenet.load_model(ncnnbin); 

		tmtool::Extractor ex1 = squeezenet.create_extractor();
		ex1.set_light_mode(true);
		tmtool::Extractor ex2 = squeezenet.create_extractor();
		ex2.set_light_mode(true);

		tmtool::Mat out1;
		tmtool::Mat out2; 

		tmtool::Mat in1 = tmtool::Mat::from_pixels(alignedFace1.data, tmtool::Mat::PIXEL_RGB2BGR, alignedFace1.cols, alignedFace1.rows);
		tmtool::Mat in2 = tmtool::Mat::from_pixels(alignedFace2.data, tmtool::Mat::PIXEL_RGB2BGR, alignedFace2.cols, alignedFace2.rows);
		
		long t1 = clock();
		ex1.input("data", in1);
		ex1.extract("fc1", out1);

		float feat1[out1.c*out1.w*out1.h];
		for (int j=0; j<out1.c*out1.w*out1.h; j++)
		{      
			// cout<<out1[j]<<"\n";
			feat1[j] = out1[j*4];
			// fprintf(file1,"%f ",out1[j*4]);
		}
		ex2.input("data", in2);
		ex2.extract("fc1", out2);
		float feat2[out2.c];
		for (int j=0; j<out2.c; j++)
		{
			feat2[j] = out2[j*4];
			// fprintf(file2,"%f ",out2[j]);
		}
		long t2 = clock();

		float sim = CalcSimilarity_1(feat1, feat2, 128);
		fprintf(stderr, "time:%f,sim:%f\n", (t2 - t1) / 2.0,sim);
        tmtool_log(LOG_COMMON, "Run facenet model successfully!\n");
        return 0;
	}
	else if(NULL != strstr(ncnnparam, "ResNet"))//output ResNet feature
	{
        const char* imagepath = name1;

        cv::Mat m = cv::imread(imagepath, 1);
        if (m.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", imagepath);
            return -1;
        }

        #if TMTOOL_VULKAN
            tmtool::create_gpu_instance();
        #endif // TMTOOL_VULKAN

            std::vector<float> cls_scores;
            detect_squeezenet(m, cls_scores,ncnnparam,ncnnbin);

        #if TMTOOL_VULKAN
            tmtool::destroy_gpu_instance();
        #endif // TMTOOL_VULKAN

            print_topk(cls_scores, 3);
            tmtool_log(LOG_COMMON, "Run ResNet model successfully!\n");
            return 0;
	}
	else if(NULL != strstr(ncnnparam, "mssd"))//output ssd feature
	{
		const char* imagepath = name1;

		cv::Mat m = cv::imread(imagepath, 1);
		if (m.empty())
		{
			fprintf(stderr, "cv::imread %s failed\n", imagepath);
			return -1;
		}

		#if TMTOOL_VULKAN
			tmtool::create_gpu_instance();
		#endif // TMTOOL_VULKAN

			std::vector<Object> objects;
			detect_mobilenet(m, objects,ncnnparam,ncnnbin);

		#if TMTOOL_VULKAN
			tmtool::destroy_gpu_instance();
		#endif // TMTOOL_VULKAN

			draw_objects_ssd(m, objects);
            tmtool_log(LOG_COMMON, "Run mobile-ssd model successfully!\n");
			return 0;

	}
	else
	{
		tmtool_log(LOG_COMMON, "do not support this model type!!\n");
        return -1;
	}
	
}
