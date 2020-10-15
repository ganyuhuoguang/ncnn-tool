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

#include "region.h"
#include <algorithm>
#include "layer_type.h"

namespace tmtool {

DEFINE_LAYER_CREATOR(Region)

Region::Region()
{
    one_blob_only = true;
    support_inplace = false;
}

int Region::load_param(const ParamDict& pd)
{
    classes = pd.get(0, 20);
    coords = pd.get(1, 2);
    boxes_of_each_grid = pd.get(2, 5);
    softmax = pd.get(3, 0);

    return 0;
}

float logistic_activate(float x) 
{ 
    return (float) (1. / (1 + exp(-x))); 
}

int softmax_calc(Mat& input, int idx, int length)
 {
    int i;
    float output[length];
    float sum = 0;
    float largest = -FLT_MAX;
    for (i = 0; i < length; ++i) {
        if (input[i+idx] > largest) largest = input[i+idx];
    }
    for (i = 0; i < length; ++i) {
        float e = exp(input[i+idx] - largest);
        sum += e;
        output[i] = e;
    }
    for (i = 0; i < length; ++i) {
        output[i] /= sum;
        input[i+idx] = output[i];
    }
    
    return 0;
}

int flatten(Mat& in, const Mat& data, int area, int channel) 
{
  int i, c;
    for (c = 0; c < channel; ++c) {
        for (i = 0; i < area; ++i) {
            int i1 = c * area + i;
            int i2 = i * channel + c;
            in[i2] = data[i1];
        }
    }

  return 0;
}

int Region::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int c = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int size = coords + classes + 1;

    top_blob.create(w * h * c, elemsize, opt.blob_allocator);
    flatten(top_blob, bottom_blob, w * h, c);

    for (int i = 0; i < h * w * boxes_of_each_grid; ++i) {
        int index = size * i;
        top_blob[index + 4] = logistic_activate(top_blob[index + 4]);
    }

    if (softmax) {
        for (int i = 0; i < h * w * boxes_of_each_grid; ++i) {
            int index = size * i;
            softmax_calc(top_blob, index+5, classes);
        }
    }
    
    return 0;

}

} // namespace ncnn
