// created by Huiran.Du for processing output feature of YOLO v3, 2019-10-19

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "layer_type.h"
#include "layer.h"
#include "platform.h"
#include "net.h"
#if NCNN_VULKAN
#include "gpu.h"
#endif // NCNN_VULKAN

using namespace ncnn;

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static cv::Mat img_prepare(cv::Mat img, int img_w, int img_h, int w, int h)
{    
    int new_w = img_w * std::min(float(w)/float(img_w), float(h)/float(img_h));
    int new_h = img_h * std::min(float(w)/float(img_w), float(h)/float(img_h));
    cv::Mat resize; 
    cv::resize(img, resize, cv::Size(new_w, new_h), 0, 0, CV_INTER_CUBIC);
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

static void draw_objects(const cv::Mat& bgr, int res, std::vector<Mat> result, char** class_names)
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

void feature_process(const std::vector<Mat>& src, std::vector<Mat>& dst, int classes, int num_box, float conf_threshold, float nms_threshold, Mat biases, int input_size, int img_w, int img_h)
{
    ncnn::Layer* yolov3 = ncnn::create_layer(ncnn::LayerType::Yolov3DetectionModifiedOutput);

    ncnn::ParamDict pd;
    pd.set(0, classes);
    pd.set(1, num_box);
    pd.set(2, conf_threshold);
    pd.set(3, nms_threshold);
    pd.set(4, biases);
    pd.set(5, input_size);
    pd.set(6, img_w);
    pd.set(7, img_h);

    yolov3->load_param(pd);

    ncnn::Option opt;
    yolov3->forward(src, dst, opt);

    delete yolov3;
}

static int YOLOv3_detect(const cv::Mat& bgr)
{
    int input_width = 416;
    char** labels = label_load("coco.labels");
    int anchor_list[18] = {116,90,  156,198,  373,326,  30,61,  62,45,  59,119,  10,13,  16,30,  33,23,};
    Mat anchors;
    anchors.create(18);
    for (int i=0; i<18; i++) anchors[i] = anchor_list[i];

    Net test;
    test.load_param("yolov3-nb-8.param");
    test.load_model("yolov3-nb-8.bin");
    
    cv::Mat input = img_prepare(bgr, bgr.cols, bgr.rows, input_width, input_width);
    Mat in = Mat::from_pixels(input.data, Mat::PIXEL_BGR, input.cols, input.rows);
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1.0/255.0, 1.0/255.0, 1.0/255.0};
    in.substract_mean_normalize(mean_vals, norm_vals);
    
    Extractor ex = test.create_extractor();
    Mat out0, out1, out2;
    std::vector<Mat> calculated;
    std::vector<Mat> processed;
    ex.input("data", in);
    ex.extract("out0", out0);
    ex.extract("out1", out1);
    ex.extract("out2", out2);

    calculated.push_back(out0);
    calculated.push_back(out1);
    calculated.push_back(out2);

    feature_process(calculated, processed, 80, 3, 0.5f, 0.45f, anchors, input_width, bgr.cols, bgr.rows);

    draw_objects(bgr, input_width, processed, labels);

    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];
    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    YOLOv3_detect(m);

    return 0;
}
