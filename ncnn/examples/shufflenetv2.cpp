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
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "platform.h"
#include "net.h"
#if TMTOOL_VULKAN
#include "gpu.h"
#endif // TMTOOL_VULKAN

static int detect_shufflenetv2(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    tmtool::Net shufflenetv2;

#if TMTOOL_VULKAN
    shufflenetv2.opt.use_vulkan_compute = true;
#endif // TMTOOL_VULKAN

    // https://github.com/miaow1988/ShuffleNet_V2_pytorch_caffe
    // models can be downloaded from https://github.com/miaow1988/ShuffleNet_V2_pytorch_caffe/releases
    shufflenetv2.load_param("shufflenet_v2_x0.5.param");
    shufflenetv2.load_model("shufflenet_v2_x0.5.bin");

    tmtool::Mat in = tmtool::Mat::from_pixels_resize(bgr.data, tmtool::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 224, 224);

    const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
    in.substract_mean_normalize(0, norm_vals);

    tmtool::Extractor ex = shufflenetv2.create_extractor();

    ex.input("data", in);

    tmtool::Mat out;
    ex.extract("fc", out);

    // manually call softmax on the fc output
    // convert result into probability
    // skip if your model already has softmax operation
    {
        tmtool::Layer* softmax = tmtool::create_layer("Softmax");

        tmtool::ParamDict pd;
        softmax->load_param(pd);

        softmax->forward_inplace(out);

        delete softmax;
    }

    out = out.reshape(out.w * out.h * out.c);

    cls_scores.resize(out.w);
    for (int j=0; j<out.w; j++)
    {
        cls_scores[j] = out[j];
    }

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

int shufflenetv2(int argc, char** argv)
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

#if TMTOOL_VULKAN
    tmtool::create_gpu_instance();
#endif // TMTOOL_VULKAN

    std::vector<float> cls_scores;
    detect_shufflenetv2(m, cls_scores);

#if TMTOOL_VULKAN
    tmtool::destroy_gpu_instance();
#endif // TMTOOL_VULKAN

    print_topk(cls_scores, 3);

    return 0;
}
