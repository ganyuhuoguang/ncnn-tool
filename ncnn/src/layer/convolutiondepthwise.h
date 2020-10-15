// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef LAYER_CONVOLUTIONDEPTHWISE_H
#define LAYER_CONVOLUTIONDEPTHWISE_H

#include "layer.h"

namespace tmtool {

class ConvolutionDepthWise : public Layer
{
public:
    ConvolutionDepthWise();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int create_requantize_op(void);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    // param
    int num_output;
    int kernel_w;
    int kernel_h;
    int dilation_w;
    int dilation_h;
    int stride_w;
    int stride_h;
    int pad_w;
    int pad_h;
    int bias_term;

    int weight_data_size;
    int group;

    int int8_scale_term;

    // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
    int activation_type;
    Mat activation_params;

    // model
    Mat weight_data;
    Mat bias_data;

    Mat weight_data_int8_scales;
    Mat bottom_blob_int8_scales;
    float top_blob_int8_scale;

    bool use_int8_inference;
    bool use_int8_requantize;

    std::vector<tmtool::Layer*> quantize_ops;
    std::vector<tmtool::Layer*> dequantize_ops;
    std::vector<tmtool::Layer*> requantize_ops;

    // merge de/requantize op into convolutiondepthwise op
    std::vector<float> dequantize_scales;
    std::vector<float> requantize_scales;
};

} // namespace tmtool

#endif // LAYER_CONVOLUTIONDEPTHWISE_H
