// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "hardsigmoid_vulkan.h"

namespace tmtool {

DEFINE_LAYER_CREATOR(HardSigmoid_vulkan)

HardSigmoid_vulkan::HardSigmoid_vulkan()
{
    support_vulkan = true;

    pipeline_hardsigmoid = 0;
    pipeline_hardsigmoid_pack4 = 0;
}

int HardSigmoid_vulkan::create_pipeline(const Option& opt)
{
    std::vector<vk_specialization_type> specializations(2);
    specializations[0].f = alpha;
    specializations[1].f = beta;

    // pack1
    {
        pipeline_hardsigmoid = new Pipeline(vkdev);
        pipeline_hardsigmoid->set_optimal_local_size_xyz();
        pipeline_hardsigmoid->create("hardsigmoid", opt, specializations, 1, 5);
    }

    // pack4
    {
        pipeline_hardsigmoid_pack4 = new Pipeline(vkdev);
        pipeline_hardsigmoid_pack4->set_optimal_local_size_xyz();
        pipeline_hardsigmoid_pack4->create("hardsigmoid_pack4", opt, specializations, 1, 5);
    }

    return 0;
}

int HardSigmoid_vulkan::destroy_pipeline(const Option& opt)
{
    delete pipeline_hardsigmoid;
    pipeline_hardsigmoid = 0;

    delete pipeline_hardsigmoid_pack4;
    pipeline_hardsigmoid_pack4 = 0;

    return 0;
}

int HardSigmoid_vulkan::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
{
    int elempack = bottom_top_blob.elempack;

    std::vector<VkMat> bindings(1);
    bindings[0] = bottom_top_blob;

    std::vector<vk_constant_type> constants(5);
    constants[0].i = bottom_top_blob.dims;
    constants[1].i = bottom_top_blob.w;
    constants[2].i = bottom_top_blob.h;
    constants[3].i = bottom_top_blob.c;
    constants[4].i = bottom_top_blob.cstep;

    const Pipeline* pipeline = elempack == 4 ? pipeline_hardsigmoid_pack4 : pipeline_hardsigmoid;

    cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);

    return 0;
}

} // namespace tmtool
