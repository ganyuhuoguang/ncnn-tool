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

#ifndef LAYER_YOLOV3DETECTIONMODIFIEDOUTPUT_H
#define LAYER_YOLOV3DETECTIONMODIFIEDOUTPUT_H
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "layer.h"

namespace tmtool {


struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

class Yolov3DetectionModifiedOutput : public Layer
{
public:
	Yolov3DetectionModifiedOutput();
    ~Yolov3DetectionModifiedOutput();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    int num_class;
    int num_box;
    float confidence_threshold;
    float nms_threshold;
    Mat biases;
    int input;
};

} // namespace tmtool

#endif // LAYER_YOLOV3DETECTIONMODIFIEDOUTPUT_H
