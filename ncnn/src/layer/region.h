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

#ifndef LAYER_REGION_H
#define LAYER_REGION_H

#define FLT_MAX 3.402823466e+38F
#define FLT_MIN 1.175494351e-38F

#include "layer.h"

namespace tmtool {

class Region : public Layer
{
public:
    Region();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    int classes;
    int coords;
    int boxes_of_each_grid;
    int softmax;
};

} // namespace ncnn

#endif // LAYER_SHUFFLECHANNEL_H
