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

#ifndef TMTOOL_LAYER_H
#define TMTOOL_LAYER_H

#include <stdio.h>
#include <string>
#include <vector>
#include <math.h>
#include "platform.h"
#include "mat.h"
#include "modelbin.h"
#include "option.h"
#include "paramdict.h"

#if TMTOOL_VULKAN
#include <vulkan/vulkan.h>
#include "command.h"
#include "pipeline.h"
#endif // TMTOOL_VULKAN

namespace tmtool {

class Layer
{
public:
    // empty
    Layer();
    // virtual destructor
    virtual ~Layer();

    // load layer specific parameter from parsed dict
    // return 0 if success
    virtual int load_param(const ParamDict& pd);

    // load layer specific weight data from model binary
    // return 0 if success
    virtual int load_model(const ModelBin& mb);

    // layer implementation specific setup
    // return 0 if success
    virtual int create_pipeline(const Option& opt = Option());

    // layer implementation specific clean
    // return 0 if success
    virtual int destroy_pipeline(const Option& opt = Option());

public:
    // one input and one output blob  表示该层为单输入单输出
    bool one_blob_only;

    // support inplace inference   表示是否可以进行就地运算
    //可以在输入数据的基础上直接修改得到输出数据，
    //但是卷积过程有重复部分，如果修改会对后面的计算产生影响。因此对于前向推理函数就会有两种方式，加上刚才的是否是单输入单输出，一共有四个推理函数。
    bool support_inplace;

    // support vulkan compute
    bool support_vulkan;

    // accept input blob with packed storage
    bool support_packing;

public:
    // implement inference
    // return 0 if success
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt = Option()) const;
    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt = Option()) const;

    // implement inplace inference
    // return 0 if success
    virtual int forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt = Option()) const;
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt = Option()) const;

#if TMTOOL_VULKAN
public:
    // upload weight blob from host to device
    virtual int upload_model(VkTransfer& cmd, const Option& opt = Option());

public:
    // implement inference
    // return 0 if success
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt = Option()) const;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt = Option()) const;

    // implement inplace inference
    // return 0 if success
    virtual int forward_inplace(std::vector<VkMat>& bottom_top_blobs, VkCompute& cmd, const Option& opt = Option()) const;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt = Option()) const;

public:
    // assigned immediately after creating this layer
    const VulkanDevice* vkdev;
#endif // TMTOOL_VULKAN

public:
    // layer type index
    int typeindex;
#if TMTOOL_STRING
    // layer type name
    std::string type;
    // layer name
    std::string name;
#endif // TMTOOL_STRING
    // blob index which this layer needs as input
    std::vector<int> bottoms;
    // blob index which this layer produces as output
    std::vector<int> tops;
};

// layer factory function
typedef Layer* (*layer_creator_func)();

struct layer_registry_entry
{
#if TMTOOL_STRING
    // layer type name
    const char* name;
#endif // TMTOOL_STRING
    // layer factory entry
    layer_creator_func creator;
};

#if TMTOOL_STRING
// get layer type from type name
int layer_to_index(const char* type);
// create layer from type name
Layer* create_layer(const char* type);
#endif // TMTOOL_STRING
// create layer from layer type  注册器
Layer* create_layer(int index);

#define DEFINE_LAYER_CREATOR(name) \
    ::tmtool::Layer* name##_layer_creator() { return new name; }

} // namespace tmtool

#endif // TMTOOL_LAYER_H
