// BUG1989 is pleased to support the open source community by supporting ncnn available.
//
// Copyright (C) 2019 BUG1989. All rights reserved.
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
#include <string.h>
#include <limits.h>
#include <math.h>
#include <assert.h>

#include <fstream>
#include <vector>
#include <set>
#include <limits>
#include <map>
#include <algorithm>

// ncnn public header
#include "net.h"
#include "layer.h"
#include "layer_type.h"

// ncnn private header
#include "layer/batchnorm.h"
#include "layer/bias.h"
#include "layer/binaryop.h"
#include "layer/clip.h"
#include "layer/concat.h"
#include "layer/convolution.h"
#include "layer/convolutiondepthwise.h"
#include "layer/crop.h"
#include "layer/deconvolution.h"
#include "layer/deconvolutiondepthwise.h"
#include "layer/detectionoutput.h"
#include "layer/dropout.h"
#include "layer/eltwise.h"
#include "layer/elu.h"
#include "layer/exp.h"
#include "layer/flatten.h"
#include "layer/innerproduct.h"
#include "layer/input.h"
#include "layer/instancenorm.h"
#include "layer/interp.h"
#include "layer/log.h"
#include "layer/lrn.h"
#include "layer/mvn.h"
#include "layer/normalize.h"
#include "layer/padding.h"
#include "layer/permute.h"
#include "layer/pooling.h"
#include "layer/power.h"
#include "layer/prelu.h"
#include "layer/priorbox.h"
#include "layer/proposal.h"
#include "layer/psroipooling.h"
#include "layer/quantize.h"
#include "layer/reduction.h"
#include "layer/relu.h"
#include "layer/reorg.h"
#include "layer/requantize.h"
#include "layer/reshape.h"
#include "layer/roialign.h"
#include "layer/roipooling.h"
#include "layer/scale.h"
#include "layer/slice.h"
#include "layer/shufflechannel.h"
#include "layer/softmax.h"
#include "layer/threshold.h"
#include "layer/unaryop.h"
#include "layer/yolodetectionoutput.h"
#include "layer/yolov3detectionoutput.h"


static bool read_int8scale_table(const char* filepath, std::map<std::string, std::vector<float> >& blob_int8scale_table, std::map<std::string, std::vector<float> >& weight_int8scale_table)
{
    blob_int8scale_table.clear();
    weight_int8scale_table.clear();

    FILE* fp = fopen(filepath, "rb");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", filepath);
        return false;
    }

    bool in_scale_vector = false;

    std::string keystr;
    std::vector<float> scales;

    while (!feof(fp))
    {
        char key[256];
        int nscan = fscanf(fp, "%255s", key);
        if (nscan != 1)
        {
            break;
        }

        if (in_scale_vector)
        {
            float scale = 1.f;
            int nscan = sscanf(key, "%f", &scale);
            if (nscan == 1)
            {
                scales.push_back(scale);
                continue;
            }
            else
            {
                // XYZ_param_N pattern
                if (strstr(keystr.c_str(), "_param_"))
                {
                    weight_int8scale_table[ keystr ] = scales;
                }
                else
                {
                    blob_int8scale_table[ keystr ] = scales;
                }

                keystr.clear();
                scales.clear();

                in_scale_vector = false;
            }
        }

        if (!in_scale_vector)
        {
            keystr = key;

            in_scale_vector = true;
        }
    }

    if (in_scale_vector)
    {
        // XYZ_param_N pattern
        if (strstr(keystr.c_str(), "_param_"))
        {
            weight_int8scale_table[ keystr ] = scales;
        }
        else
        {
            blob_int8scale_table[ keystr ] = scales;
        }
    }

    fclose(fp);

    return true;
}

class NetQuantize : public tmtool::Net
{
public:
    // 0=fp32 1=fp16 2=int8
    int storage_type;
    std::map<std::string, std::vector<float> > blob_int8scale_table;
    std::map<std::string, std::vector<float> > weight_int8scale_table; 

public:
    int quantize_convolution();
    int quantize_convolutiondepthwise();
    int quantize_innerproduct();

public:
    int fprintf_param_int_array(int id, const tmtool::Mat& m, FILE* pp);
    int fprintf_param_float_array(int id, const tmtool::Mat& m, FILE* pp);

    int fwrite_weight_tag_data(int tag, const tmtool::Mat& data, FILE* bp);
    int fwrite_weight_data(const tmtool::Mat& data, FILE* bp);

    int save(const char* parampath, const char* binpath);
};

int NetQuantize::quantize_convolution()
{
    const int layer_count = layers.size();
    for (int i=0; i<layer_count; i++)
    {
        // find convoultion layer
        if (layers[i]->type != "Convolution")
            continue;

        // find convolution layer
        std::map<std::string, std::vector<float> >::iterator iter_data = blob_int8scale_table.find(layers[i]->name);
        if (iter_data == weight_int8scale_table.end())
            continue;

        char key[256];
        sprintf(key, "%s_param_0", layers[i]->name.c_str());
        std::map<std::string, std::vector<float> >::iterator iter = weight_int8scale_table.find(key);
        if (iter == weight_int8scale_table.end())
        {
            fprintf(stderr, "this layer need to be quantized, but no scale param!\n");
            return -1;
        }
            
        // Convolution - quantize weight from fp32 to int8
        tmtool::Convolution* convolution = (tmtool::Convolution*)layers[i];

        std::vector<float> weight_data_int8_scales = iter->second;

        fprintf(stderr, "quantize_convolution %s\n", convolution->name.c_str());

        {
            tmtool::Mat int8_weight_data(convolution->weight_data_size, (size_t)1u);
            if (int8_weight_data.empty())
                return -100;

            const int weight_data_size_output = convolution->weight_data_size / convolution->num_output;

            // quantize weight to int8
            for (int n=0; n<convolution->num_output; n++)
            {
                tmtool::Layer* op = tmtool::create_layer(tmtool::LayerType::Quantize);

                tmtool::ParamDict pd;
                pd.set(0, weight_data_int8_scales[n]);// scale

                op->load_param(pd);

                tmtool::Option opt;
                opt.blob_allocator = int8_weight_data.allocator;

                const tmtool::Mat weight_data_n = convolution->weight_data.range(weight_data_size_output * n, weight_data_size_output);
                tmtool::Mat int8_weight_data_n = int8_weight_data.range(weight_data_size_output * n, weight_data_size_output);
                op->forward(weight_data_n, int8_weight_data_n, opt);

                delete op;
            }

            convolution->weight_data = int8_weight_data;
        }

        convolution->int8_scale_term = 2;
    }

    return 0;
}

int NetQuantize::quantize_convolutiondepthwise()
{
    const int layer_count = layers.size();
    for (int i=0; i<layer_count; i++)
    {
        // find convoultion layer
        if (layers[i]->type != "ConvolutionDepthWise")
            continue;

        // find convolutiondepthwise layer
        std::map<std::string, std::vector<float> >::iterator iter_data = blob_int8scale_table.find(layers[i]->name);
        if (iter_data == weight_int8scale_table.end())
            continue;

        char key[256];
        sprintf(key, "%s_param_0", layers[i]->name.c_str());
        std::map<std::string, std::vector<float> >::iterator iter = weight_int8scale_table.find(key);
        if (iter == weight_int8scale_table.end())
        {
            fprintf(stderr, "this layer need to be quantized, but no scale param!\n");
            return -1;
        }
            
        // Convolution - quantize weight from fp32 to int8
        tmtool::ConvolutionDepthWise* convdw = (tmtool::ConvolutionDepthWise*)layers[i];

        std::vector<float> weight_data_int8_scales = iter->second;

        fprintf(stderr, "quantize_convolution %s\n", convdw->name.c_str());

        {
            tmtool::Mat int8_weight_data(convdw->weight_data_size, (size_t)1u);
            if (int8_weight_data.empty())
                return -100;

            const int weight_data_size_output = convdw->weight_data_size / convdw->group;

            // quantize weight to int8
            for (int n=0; n<convdw->group; n++)
            {
                tmtool::Layer* op = tmtool::create_layer(tmtool::LayerType::Quantize);

                tmtool::ParamDict pd;
                pd.set(0, weight_data_int8_scales[n]);// scale

                op->load_param(pd);

                tmtool::Option opt;
                opt.blob_allocator = int8_weight_data.allocator;

                const tmtool::Mat weight_data_n = convdw->weight_data.range(weight_data_size_output * n, weight_data_size_output);
                tmtool::Mat int8_weight_data_n = int8_weight_data.range(weight_data_size_output * n, weight_data_size_output);
                op->forward(weight_data_n, int8_weight_data_n, opt);

                delete op;
            }

            convdw->weight_data = int8_weight_data;
        }

        convdw->int8_scale_term = 1;
    }

    return 0;
}

int NetQuantize::quantize_innerproduct()
{
    const int layer_count = layers.size();
    for (int i=0; i<layer_count; i++)
    {
        // find convoultion layer
        if (layers[i]->type != "InnerProduct")
            continue;

        // find InnerProduct layer
        std::map<std::string, std::vector<float> >::iterator iter_data = blob_int8scale_table.find(layers[i]->name);
        if (iter_data == weight_int8scale_table.end())
            continue;

        char key[256];
        sprintf(key, "%s_param_0", layers[i]->name.c_str());
        std::map<std::string, std::vector<float> >::iterator iter = weight_int8scale_table.find(key);
        if (iter == weight_int8scale_table.end())
        {
            fprintf(stderr, "this layer need to be quantized, but no scale param!\n");
            return -1;
        }
            
        // InnerProduct - quantize weight from fp32 to int8
        tmtool::InnerProduct* fc = (tmtool::InnerProduct*)layers[i];

        std::vector<float> weight_data_int8_scales = iter->second;

        fprintf(stderr, "quantize_convolution %s\n", fc->name.c_str());

        {
            tmtool::Mat int8_weight_data(fc->weight_data_size, (size_t)1u);
            if (int8_weight_data.empty())
                return -100;

            const int weight_data_size_output = fc->weight_data_size / fc->num_output;

            // quantize weight to int8
            for (int n=0; n<fc->num_output; n++)
            {
                tmtool::Layer* op = tmtool::create_layer(tmtool::LayerType::Quantize);

                tmtool::ParamDict pd;
                pd.set(0, weight_data_int8_scales[n]);// scale

                op->load_param(pd);

                tmtool::Option opt;
                opt.blob_allocator = int8_weight_data.allocator;

                const tmtool::Mat weight_data_n = fc->weight_data.range(weight_data_size_output * n, weight_data_size_output);
                tmtool::Mat int8_weight_data_n = int8_weight_data.range(weight_data_size_output * n, weight_data_size_output);
                op->forward(weight_data_n, int8_weight_data_n, opt);

                delete op;
            }

            fc->weight_data = int8_weight_data;
        }

        fc->int8_scale_term = 2;
    }

    return 0;
}

int NetQuantize::fprintf_param_int_array(int id, const tmtool::Mat& m, FILE* pp)
{
    const int count = m.w;
    const int* ptr = m;

    fprintf(pp, " -%d=%d", 23300 + id, count);
    for (int i=0; i<count; i++)
    {
        fprintf(pp, ",%d", ptr[i]);
    }

    return 0;
}

int NetQuantize::fprintf_param_float_array(int id, const tmtool::Mat& m, FILE* pp)
{
    const int count = m.w;
    const float* ptr = m;

    fprintf(pp, " -%d=%d", 23300 + id, count);
    for (int i=0; i<count; i++)
    {
        fprintf(pp, ",%f", ptr[i]);
    }

    return 0;
}

static inline size_t alignSize(size_t sz, int n)
{
    return (sz + n-1) & -n;
}

int NetQuantize::fwrite_weight_tag_data(int tag, const tmtool::Mat& data, FILE* bp)
{
    int p0 = ftell(bp);

    tmtool::Mat data_flattened = data.reshape(data.w * data.h * data.c);

    if (data.elemsize == 1)
        tag = 0x000D4B38; // int8 magic

    fwrite(&tag, sizeof(int), 1, bp);
    fwrite(data_flattened.data, data_flattened.elemsize, data_flattened.w, bp);

    // padding to 32bit align
    int nwrite = ftell(bp) - p0;
    int nalign = alignSize(nwrite, 4);
    unsigned char padding[4] = {0x00, 0x00, 0x00, 0x00};
    fwrite(padding, sizeof(unsigned char), nalign - nwrite, bp);

    return 0;
}

int NetQuantize::fwrite_weight_data(const tmtool::Mat& data, FILE* bp)
{
    int p0 = ftell(bp);

    tmtool::Mat data_flattened = data.reshape(data.w * data.h * data.c);
    fwrite(data_flattened.data, data_flattened.elemsize, data_flattened.w, bp);

    // padding to 32bit align
    int nwrite = ftell(bp) - p0;
    int nalign = alignSize(nwrite, 4);
    unsigned char padding[4] = {0x00, 0x00, 0x00, 0x00};
    fwrite(padding, sizeof(unsigned char), nalign - nwrite, bp);

    return 0;
}

int NetQuantize::save(const char* parampath, const char* binpath)
{
    FILE* pp = fopen(parampath, "wb");
    FILE* bp = fopen(binpath, "wb");

    fprintf(pp, "7767517\n");

    const int layer_count = layers.size();

    int layer_count_fused = 0;
    std::set<std::string> blob_names;
    for (int i=0; i<layer_count; i++)
    {
        const tmtool::Layer* layer = layers[i];
        if (layer->type == "ncnnfused")
            continue;

        layer_count_fused++;

        int bottom_count = layer->bottoms.size();
        for (int j=0; j<bottom_count; j++)
        {
            int bottom_blob_index = layer->bottoms[j];
            blob_names.insert(blobs[bottom_blob_index].name);
        }

        int top_count = layer->tops.size();
        for (int j=0; j<top_count; j++)
        {
            int top_blob_index = layer->tops[j];
            blob_names.insert(blobs[top_blob_index].name);
        }
    }

    int blob_count_fused = blob_names.size();

    fprintf(pp, "%d %d\n", layer_count_fused, blob_count_fused);

    for (int i=0; i<layer_count; i++)
    {
        const tmtool::Layer* layer = layers[i];
        if (layer->type == "ncnnfused")
            continue;

        int bottom_count = layer->bottoms.size();
        int top_count = layer->tops.size();

        fprintf(pp, "%-24s %-24s %d %d", layer->type.c_str(), layer->name.c_str(), bottom_count, top_count);

        for (int j=0; j<bottom_count; j++)
        {
            int bottom_blob_index = layer->bottoms[j];
            fprintf(pp, " %s", blobs[bottom_blob_index].name.c_str());
        }
        for (int j=0; j<top_count; j++)
        {
            int top_blob_index = layer->tops[j];
            fprintf(pp, " %s", blobs[top_blob_index].name.c_str());
        }

        tmtool::Layer* layer_default = tmtool::create_layer(layer->typeindex);

        tmtool::ParamDict pd;
        layer_default->load_param(pd);

#define fprintf_param_value(format, phase) \
        { if (op->phase != op_default->phase) fprintf(pp, format, op->phase); }

        if (layer->type == "BatchNorm")
        {
            tmtool::BatchNorm* op = (tmtool::BatchNorm*)layer;
            tmtool::BatchNorm* op_default = (tmtool::BatchNorm*)layer_default;

            fprintf_param_value(" 0=%d", channels)
            fprintf_param_value(" 1=%f", eps)

            fwrite_weight_data(op->slope_data, bp);
            fwrite_weight_data(op->mean_data, bp);
            fwrite_weight_data(op->var_data, bp);
            fwrite_weight_data(op->bias_data, bp);
        }
        else if (layer->type == "Bias")
        {
            tmtool::Bias* op = (tmtool::Bias*)layer;
            tmtool::Bias* op_default = (tmtool::Bias*)layer_default;

            fprintf_param_value(" 0=%d", bias_data_size)

            fwrite_weight_data(op->bias_data, bp);
        }
        else if (layer->type == "BinaryOp")
        {
            tmtool::BinaryOp* op = (tmtool::BinaryOp*)layer;
            tmtool::BinaryOp* op_default = (tmtool::BinaryOp*)layer_default;

            fprintf_param_value(" 0=%d", op_type)
            fprintf_param_value(" 1=%d", with_scalar)
            fprintf_param_value(" 2=%f", b)
        }
        else if (layer->type == "Clip")
        {
            tmtool::Clip* op = (tmtool::Clip*)layer;
            tmtool::Clip* op_default = (tmtool::Clip*)layer_default;

            fprintf_param_value(" 0=%f", min)
            fprintf_param_value(" 1=%f", max)
        }
        else if (layer->type == "Concat")
        {
            tmtool::Concat* op = (tmtool::Concat*)layer;
            tmtool::Concat* op_default = (tmtool::Concat*)layer_default;

            fprintf_param_value(" 0=%d", axis)
        }
        else if (layer->type == "Convolution")
        {
            tmtool::Convolution* op = (tmtool::Convolution*)layer;
            tmtool::Convolution* op_default = (tmtool::Convolution*)layer_default;

            fprintf_param_value(" 0=%d", num_output)
            fprintf_param_value(" 1=%d", kernel_w)
            { if (op->kernel_h != op->kernel_w) fprintf(pp, " 11=%d", op->kernel_h); }
            fprintf_param_value(" 2=%d", dilation_w)
            { if (op->dilation_h != op->dilation_w) fprintf(pp, " 12=%d", op->dilation_h); }
            fprintf_param_value(" 3=%d", stride_w)
            { if (op->stride_h != op->stride_w) fprintf(pp, " 13=%d", op->stride_h); }
            fprintf_param_value(" 4=%d", pad_w)
            { if (op->pad_h != op->pad_w) fprintf(pp, " 14=%d", op->pad_h); }
            fprintf_param_value(" 5=%d", bias_term)
            fprintf_param_value(" 6=%d", weight_data_size)
            fprintf_param_value(" 8=%d", int8_scale_term)
            fprintf_param_value(" 9=%d", activation_type)
            { if (!op->activation_params.empty()) fprintf_param_float_array(10, op->activation_params, pp); }

            fwrite_weight_tag_data(0, op->weight_data, bp);
            fwrite_weight_data(op->bias_data, bp);

            // write int8_scale data
            if (op->int8_scale_term)
            {            
                std::vector<float> weight_int8scale;
                std::vector<float> blob_int8scale;

                char key[256];
                sprintf(key, "%s_param_0", layer->name.c_str());
                if (weight_int8scale_table.find(std::string(key)) != weight_int8scale_table.end())
                {
                    weight_int8scale = weight_int8scale_table[std::string(key)];
                }

                if (blob_int8scale_table.find(layer->name) != blob_int8scale_table.end())
                {
                    blob_int8scale = blob_int8scale_table[layer->name];
                }

                // write int8_scale data
                fwrite(weight_int8scale.data(), sizeof(float), weight_int8scale.size(), bp);
                fwrite(blob_int8scale.data(), sizeof(float), blob_int8scale.size(), bp);
            }
        }
        else if (layer->type == "ConvolutionDepthWise")
        {
            tmtool::ConvolutionDepthWise* op = (tmtool::ConvolutionDepthWise*)layer;
            tmtool::ConvolutionDepthWise* op_default = (tmtool::ConvolutionDepthWise*)layer_default;

            fprintf_param_value(" 0=%d", num_output)
            fprintf_param_value(" 1=%d", kernel_w)
            { if (op->kernel_h != op->kernel_w) fprintf(pp, " 11=%d", op->kernel_h); }
            fprintf_param_value(" 2=%d", dilation_w)
            { if (op->dilation_h != op->dilation_w) fprintf(pp, " 12=%d", op->dilation_h); }
            fprintf_param_value(" 3=%d", stride_w)
            { if (op->stride_h != op->stride_w) fprintf(pp, " 13=%d", op->stride_h); }
            fprintf_param_value(" 4=%d", pad_w)
            { if (op->pad_h != op->pad_w) fprintf(pp, " 14=%d", op->pad_h); }
            fprintf_param_value(" 5=%d", bias_term)
            fprintf_param_value(" 6=%d", weight_data_size)
            fprintf_param_value(" 7=%d", group)
            fprintf_param_value(" 8=%d", int8_scale_term)
            fprintf_param_value(" 9=%d", activation_type)
            { if (!op->activation_params.empty()) fprintf_param_float_array(10, op->activation_params, pp); }

            fwrite_weight_tag_data(0, op->weight_data, bp);
            fwrite_weight_data(op->bias_data, bp);

            // write int8_scale data
            if (op->int8_scale_term)
            {            
                std::vector<float> weight_int8scale;
                std::vector<float> blob_int8scale;

                char key[256];
                sprintf(key, "%s_param_0", layer->name.c_str());
                if (weight_int8scale_table.find(std::string(key)) != weight_int8scale_table.end())
                {
                    weight_int8scale = weight_int8scale_table[std::string(key)];
                }

                if (blob_int8scale_table.find(layer->name) != blob_int8scale_table.end())
                {
                    blob_int8scale = blob_int8scale_table[layer->name];
                }

                // write int8_scale data
                fwrite(weight_int8scale.data(), sizeof(float), weight_int8scale.size(), bp);
                fwrite(blob_int8scale.data(), sizeof(float), blob_int8scale.size(), bp);
            }            
        }
        else if (layer->type == "Crop")
        {
            tmtool::Crop* op = (tmtool::Crop*)layer;
            tmtool::Crop* op_default = (tmtool::Crop*)layer_default;

            fprintf_param_value(" 0=%d", woffset)
            fprintf_param_value(" 1=%d", hoffset)
            fprintf_param_value(" 2=%d", coffset)
            fprintf_param_value(" 3=%d", outw)
            fprintf_param_value(" 4=%d", outh)
            fprintf_param_value(" 5=%d", outc)
        }
        else if (layer->type == "Deconvolution")
        {
            tmtool::Deconvolution* op = (tmtool::Deconvolution*)layer;
            tmtool::Deconvolution* op_default = (tmtool::Deconvolution*)layer_default;

            fprintf_param_value(" 0=%d", num_output)
            fprintf_param_value(" 1=%d", kernel_w)
            { if (op->kernel_h != op->kernel_w) fprintf(pp, " 11=%d", op->kernel_h); }
            fprintf_param_value(" 2=%d", dilation_w)
            { if (op->dilation_h != op->dilation_w) fprintf(pp, " 12=%d", op->dilation_h); }
            fprintf_param_value(" 3=%d", stride_w)
            { if (op->stride_h != op->stride_w) fprintf(pp, " 13=%d", op->stride_h); }
            fprintf_param_value(" 4=%d", pad_w)
            { if (op->pad_h != op->pad_w) fprintf(pp, " 14=%d", op->pad_h); }
            fprintf_param_value(" 5=%d", bias_term)
            fprintf_param_value(" 6=%d", weight_data_size)
            fprintf_param_value(" 9=%d", activation_type)
            { if (!op->activation_params.empty()) fprintf_param_float_array(10, op->activation_params, pp); }

            fwrite_weight_tag_data(0, op->weight_data, bp);
            fwrite_weight_data(op->bias_data, bp);
        }
        else if (layer->type == "DeconvolutionDepthWise")
        {
            tmtool::DeconvolutionDepthWise* op = (tmtool::DeconvolutionDepthWise*)layer;
            tmtool::DeconvolutionDepthWise* op_default = (tmtool::DeconvolutionDepthWise*)layer_default;

            fprintf_param_value(" 0=%d", num_output)
            fprintf_param_value(" 1=%d", kernel_w)
            { if (op->kernel_h != op->kernel_w) fprintf(pp, " 11=%d", op->kernel_h); }
            fprintf_param_value(" 2=%d", dilation_w)
            { if (op->dilation_h != op->dilation_w) fprintf(pp, " 12=%d", op->dilation_h); }
            fprintf_param_value(" 3=%d", stride_w)
            { if (op->stride_h != op->stride_w) fprintf(pp, " 13=%d", op->stride_h); }
            fprintf_param_value(" 4=%d", pad_w)
            { if (op->pad_h != op->pad_w) fprintf(pp, " 14=%d", op->pad_h); }
            fprintf_param_value(" 5=%d", bias_term)
            fprintf_param_value(" 6=%d", weight_data_size)
            fprintf_param_value(" 7=%d", group)
            fprintf_param_value(" 9=%d", activation_type)
            { if (!op->activation_params.empty()) fprintf_param_float_array(10, op->activation_params, pp); }

            fwrite_weight_tag_data(0, op->weight_data, bp);
            fwrite_weight_data(op->bias_data, bp);
        }
        else if (layer->type == "DetectionOutput")
        {
            tmtool::DetectionOutput* op = (tmtool::DetectionOutput*)layer;
            tmtool::DetectionOutput* op_default = (tmtool::DetectionOutput*)layer_default;

            fprintf_param_value(" 0=%d", num_class)
            fprintf_param_value(" 1=%f", nms_threshold)
            fprintf_param_value(" 2=%d", nms_top_k)
            fprintf_param_value(" 3=%d", keep_top_k)
            fprintf_param_value(" 4=%f", confidence_threshold)
            fprintf_param_value(" 5=%f", variances[0])
            fprintf_param_value(" 6=%f", variances[1])
            fprintf_param_value(" 7=%f", variances[2])
            fprintf_param_value(" 8=%f", variances[3])
        }
        else if (layer->type == "Dropout")
        {
            tmtool::Dropout* op = (tmtool::Dropout*)layer;
            tmtool::Dropout* op_default = (tmtool::Dropout*)layer_default;

            fprintf_param_value(" 0=%f", scale)
        }
        else if (layer->type == "Eltwise")
        {
            tmtool::Eltwise* op = (tmtool::Eltwise*)layer;
            tmtool::Eltwise* op_default = (tmtool::Eltwise*)layer_default;

            fprintf_param_value(" 0=%d", op_type)
            { if (!op->coeffs.empty()) fprintf_param_float_array(1, op->coeffs, pp); }
        }
        else if (layer->type == "ELU")
        {
            tmtool::ELU* op = (tmtool::ELU*)layer;
            tmtool::ELU* op_default = (tmtool::ELU*)layer_default;

            fprintf_param_value(" 0=%f", alpha)
        }
        else if (layer->type == "Exp")
        {
            tmtool::Exp* op = (tmtool::Exp*)layer;
            tmtool::Exp* op_default = (tmtool::Exp*)layer_default;

            fprintf_param_value(" 0=%f", base)
            fprintf_param_value(" 1=%f", scale)
            fprintf_param_value(" 2=%f", shift)
        }
        else if (layer->type == "InnerProduct")
        {
            tmtool::InnerProduct* op = (tmtool::InnerProduct*)layer;
            tmtool::InnerProduct* op_default = (tmtool::InnerProduct*)layer_default;

            fprintf_param_value(" 0=%d", num_output)
            fprintf_param_value(" 1=%d", bias_term)
            fprintf_param_value(" 2=%d", weight_data_size)
            fprintf_param_value(" 8=%d", int8_scale_term)
            fprintf_param_value(" 9=%d", activation_type)
            { if (!op->activation_params.empty()) fprintf_param_float_array(10, op->activation_params, pp); }

            fwrite_weight_tag_data(0, op->weight_data, bp);
            fwrite_weight_data(op->bias_data, bp);

            // write int8_scale data
            if (op->int8_scale_term)
            {            
                std::vector<float> weight_int8scale;
                std::vector<float> blob_int8scale;

                char key[256];
                sprintf(key, "%s_param_0", layer->name.c_str());
                if (weight_int8scale_table.find(std::string(key)) != weight_int8scale_table.end())
                {
                    weight_int8scale = weight_int8scale_table[std::string(key)];
                }

                if (blob_int8scale_table.find(layer->name) != blob_int8scale_table.end())
                {
                    blob_int8scale = blob_int8scale_table[layer->name];
                }

                // write int8_scale data
                fwrite(weight_int8scale.data(), sizeof(float), weight_int8scale.size(), bp);
                fwrite(blob_int8scale.data(), sizeof(float), blob_int8scale.size(), bp);
            }            
        }
        else if (layer->type == "Input")
        {
            tmtool::Input* op = (tmtool::Input*)layer;
            tmtool::Input* op_default = (tmtool::Input*)layer_default;

            fprintf_param_value(" 0=%d", w)
            fprintf_param_value(" 1=%d", h)
            fprintf_param_value(" 2=%d", c)
        }
        else if (layer->type == "InstanceNorm")
        {
            tmtool::InstanceNorm* op = (tmtool::InstanceNorm*)layer;
            tmtool::InstanceNorm* op_default = (tmtool::InstanceNorm*)layer_default;

            fprintf_param_value(" 0=%d", channels)
            fprintf_param_value(" 1=%f", eps)
        }
        else if (layer->type == "Interp")
        {
            tmtool::Interp* op = (tmtool::Interp*)layer;
            tmtool::Interp* op_default = (tmtool::Interp*)layer_default;

            fprintf_param_value(" 0=%d", resize_type)
            fprintf_param_value(" 1=%f", height_scale)
            fprintf_param_value(" 2=%f", width_scale)
            fprintf_param_value(" 3=%d", output_height)
            fprintf_param_value(" 4=%d", output_width)
        }
        else if (layer->type == "Log")
        {
            tmtool::Log* op = (tmtool::Log*)layer;
            tmtool::Log* op_default = (tmtool::Log*)layer_default;

            fprintf_param_value(" 0=%f", base)
            fprintf_param_value(" 1=%f", scale)
            fprintf_param_value(" 2=%f", shift)
        }
        else if (layer->type == "LRN")
        {
            tmtool::LRN* op = (tmtool::LRN*)layer;
            tmtool::LRN* op_default = (tmtool::LRN*)layer_default;

            fprintf_param_value(" 0=%d", region_type)
            fprintf_param_value(" 1=%d", local_size)
            fprintf_param_value(" 2=%f", alpha)
            fprintf_param_value(" 3=%f", beta)
            fprintf_param_value(" 4=%f", bias)
        }
        else if (layer->type == "MVN")
        {
            tmtool::MVN* op = (tmtool::MVN*)layer;
            tmtool::MVN* op_default = (tmtool::MVN*)layer_default;

            fprintf_param_value(" 0=%d", normalize_variance)
            fprintf_param_value(" 1=%d", across_channels)
            fprintf_param_value(" 2=%f", eps)
        }
        else if (layer->type == "Normalize")
        {
            tmtool::Normalize* op = (tmtool::Normalize*)layer;
            tmtool::Normalize* op_default = (tmtool::Normalize*)layer_default;

            fprintf_param_value(" 0=%d", across_spatial)
            fprintf_param_value(" 1=%d", channel_shared)
            fprintf_param_value(" 2=%f", eps)
            fprintf_param_value(" 3=%d", scale_data_size)
            fprintf_param_value(" 4=%d", across_channel)

            fwrite_weight_data(op->scale_data, bp);
        }
        else if (layer->type == "Padding")
        {
            tmtool::Padding* op = (tmtool::Padding*)layer;
            tmtool::Padding* op_default = (tmtool::Padding*)layer_default;

            fprintf_param_value(" 0=%d", top)
            fprintf_param_value(" 1=%d", bottom)
            fprintf_param_value(" 2=%d", left)
            fprintf_param_value(" 3=%d", right)
            fprintf_param_value(" 4=%d", type)
            fprintf_param_value(" 5=%f", value)
        }
        else if (layer->type == "Permute")
        {
            tmtool::Permute* op = (tmtool::Permute*)layer;
            tmtool::Permute* op_default = (tmtool::Permute*)layer_default;

            fprintf_param_value(" 0=%d", order_type)
        }
        else if (layer->type == "Pooling")
        {
            tmtool::Pooling* op = (tmtool::Pooling*)layer;
            tmtool::Pooling* op_default = (tmtool::Pooling*)layer_default;

            fprintf_param_value(" 0=%d", pooling_type)
            fprintf_param_value(" 1=%d", kernel_w)
            { if (op->kernel_h != op->kernel_w) fprintf(pp, " 11=%d", op->kernel_h); }
            fprintf_param_value(" 2=%d", stride_w)
            { if (op->stride_h != op->stride_w) fprintf(pp, " 12=%d", op->stride_h); }
            fprintf_param_value(" 3=%d", pad_left)
            { if (op->pad_top != op->pad_left) fprintf(pp, " 13=%d", op->pad_top); }
            { if (op->pad_right != op->pad_left) fprintf(pp, " 14=%d", op->pad_right); }
            { if (op->pad_bottom != op->pad_top) fprintf(pp, " 15=%d", op->pad_bottom); }
            fprintf_param_value(" 4=%d", global_pooling)
            fprintf_param_value(" 5=%d", pad_mode)
        }
        else if (layer->type == "Power")
        {
            tmtool::Power* op = (tmtool::Power*)layer;
            tmtool::Power* op_default = (tmtool::Power*)layer_default;

            fprintf_param_value(" 0=%f", power)
            fprintf_param_value(" 1=%f", scale)
            fprintf_param_value(" 2=%f", shift)
        }
        else if (layer->type == "PReLU")
        {
            tmtool::PReLU* op = (tmtool::PReLU*)layer;
            tmtool::PReLU* op_default = (tmtool::PReLU*)layer_default;

            fprintf_param_value(" 0=%d", num_slope)

            fwrite_weight_data(op->slope_data, bp);
        }
        else if (layer->type == "PriorBox")
        {
            tmtool::PriorBox* op = (tmtool::PriorBox*)layer;
            tmtool::PriorBox* op_default = (tmtool::PriorBox*)layer_default;

            { if (!op->min_sizes.empty()) fprintf_param_float_array(0, op->min_sizes, pp); }
            { if (!op->max_sizes.empty()) fprintf_param_float_array(1, op->max_sizes, pp); }
            { if (!op->aspect_ratios.empty()) fprintf_param_float_array(2, op->aspect_ratios, pp); }
            fprintf_param_value(" 3=%f", variances[0])
            fprintf_param_value(" 4=%f", variances[1])
            fprintf_param_value(" 5=%f", variances[2])
            fprintf_param_value(" 6=%f", variances[3])
            fprintf_param_value(" 7=%d", flip)
            fprintf_param_value(" 8=%d", clip)
            fprintf_param_value(" 9=%d", image_width)
            fprintf_param_value(" 10=%d", image_height)
            fprintf_param_value(" 11=%f", step_width)
            fprintf_param_value(" 12=%f", step_height)
            fprintf_param_value(" 13=%f", offset)
        }
        else if (layer->type == "Proposal")
        {
            tmtool::Proposal* op = (tmtool::Proposal*)layer;
            tmtool::Proposal* op_default = (tmtool::Proposal*)layer_default;

            fprintf_param_value(" 0=%d", feat_stride)
            fprintf_param_value(" 1=%d", base_size)
            fprintf_param_value(" 2=%d", pre_nms_topN)
            fprintf_param_value(" 3=%d", after_nms_topN)
            fprintf_param_value(" 4=%f", nms_thresh)
            fprintf_param_value(" 5=%d", min_size)
        }
        else if (layer->type == "PSROIPooling")
        {
            tmtool::PSROIPooling* op = (tmtool::PSROIPooling*)layer;
            tmtool::PSROIPooling* op_default = (tmtool::PSROIPooling*)layer_default;

            fprintf_param_value(" 0=%d", pooled_width)
            fprintf_param_value(" 1=%d", pooled_height)
            fprintf_param_value(" 2=%f", spatial_scale)
            fprintf_param_value(" 3=%d", output_dim)
        }
        else if (layer->type == "Quantize")
        {
            tmtool::Quantize* op = (tmtool::Quantize*)layer;
            tmtool::Quantize* op_default = (tmtool::Quantize*)layer_default;

            fprintf_param_value(" 0=%f", scale)
        }
        else if (layer->type == "Reduction")
        {
            tmtool::Reduction* op = (tmtool::Reduction*)layer;
            tmtool::Reduction* op_default = (tmtool::Reduction*)layer_default;

            fprintf_param_value(" 0=%d", operation)
            fprintf_param_value(" 1=%d", dim)
            fprintf_param_value(" 2=%f", coeff)
        }
        else if (layer->type == "ReLU")
        {
            tmtool::ReLU* op = (tmtool::ReLU*)layer;
            tmtool::ReLU* op_default = (tmtool::ReLU*)layer_default;

            fprintf_param_value(" 0=%f", slope)
        }
        else if (layer->type == "Reorg")
        {
            tmtool::Reorg* op = (tmtool::Reorg*)layer;
            tmtool::Reorg* op_default = (tmtool::Reorg*)layer_default;

            fprintf_param_value(" 0=%d", stride)
        }
        else if (layer->type == "Requantize")
        {
            tmtool::Requantize* op = (tmtool::Requantize*)layer;
            tmtool::Requantize* op_default = (tmtool::Requantize*)layer_default;

            fprintf_param_value(" 0=%f", scale_in)
            fprintf_param_value(" 1=%f", scale_out)
            fprintf_param_value(" 2=%d", bias_term)
            fprintf_param_value(" 3=%d", bias_data_size)
            fprintf_param_value(" 4=%d", fusion_relu)
        }
        else if (layer->type == "Reshape")
        {
            tmtool::Reshape* op = (tmtool::Reshape*)layer;
            tmtool::Reshape* op_default = (tmtool::Reshape*)layer_default;

            fprintf_param_value(" 0=%d", w)
            fprintf_param_value(" 1=%d", h)
            fprintf_param_value(" 2=%d", c)
            fprintf_param_value(" 3=%d", permute)
        }
        else if (layer->type == "ROIAlign")
        {
            tmtool::ROIAlign* op = (tmtool::ROIAlign*)layer;
            tmtool::ROIAlign* op_default = (tmtool::ROIAlign*)layer_default;

            fprintf_param_value(" 0=%d", pooled_width)
            fprintf_param_value(" 1=%d", pooled_height)
            fprintf_param_value(" 2=%f", spatial_scale)
        }
        else if (layer->type == "ROIPooling")
        {
            tmtool::ROIPooling* op = (tmtool::ROIPooling*)layer;
            tmtool::ROIPooling* op_default = (tmtool::ROIPooling*)layer_default;

            fprintf_param_value(" 0=%d", pooled_width)
            fprintf_param_value(" 1=%d", pooled_height)
            fprintf_param_value(" 2=%f", spatial_scale)
        }
        else if (layer->type == "Scale")
        {
            tmtool::Scale* op = (tmtool::Scale*)layer;
            tmtool::Scale* op_default = (tmtool::Scale*)layer_default;

            fprintf_param_value(" 0=%d", scale_data_size)
            fprintf_param_value(" 1=%d", bias_term)

            fwrite_weight_data(op->scale_data, bp);
            fwrite_weight_data(op->bias_data, bp);
        }
        else if (layer->type == "ShuffleChannel")
        {
            tmtool::ShuffleChannel* op = (tmtool::ShuffleChannel*)layer;
            tmtool::ShuffleChannel* op_default = (tmtool::ShuffleChannel*)layer_default;

            fprintf_param_value(" 0=%d", group)
        }
        else if (layer->type == "Slice")
        {
            tmtool::Slice* op = (tmtool::Slice*)layer;
            tmtool::Slice* op_default = (tmtool::Slice*)layer_default;

            { if (!op->slices.empty()) fprintf_param_int_array(0, op->slices, pp); }
            fprintf_param_value(" 1=%d", axis)
        }
        else if (layer->type == "Softmax")
        {
            tmtool::Softmax* op = (tmtool::Softmax*)layer;
            tmtool::Softmax* op_default = (tmtool::Softmax*)layer_default;

            fprintf_param_value(" 0=%d", axis)

            // HACK
            if (op->axis != 0)
            {
                int fixbug0 = 1;
                fprintf(pp, " 1=%d", fixbug0);
            }
        }
        else if (layer->type == "Threshold")
        {
            tmtool::Threshold* op = (tmtool::Threshold*)layer;
            tmtool::Threshold* op_default = (tmtool::Threshold*)layer_default;

            fprintf_param_value(" 0=%f", threshold)
        }
        else if (layer->type == "UnaryOp")
        {
            tmtool::UnaryOp* op = (tmtool::UnaryOp*)layer;
            tmtool::UnaryOp* op_default = (tmtool::UnaryOp*)layer_default;

            fprintf_param_value(" 0=%d", op_type)
        }
        else if (layer->type == "YoloDetectionOutput")
        {
            tmtool::YoloDetectionOutput* op = (tmtool::YoloDetectionOutput*)layer;
            tmtool::YoloDetectionOutput* op_default = (tmtool::YoloDetectionOutput*)layer_default;

            fprintf_param_value(" 0=%d", num_class)
            fprintf_param_value(" 1=%d", num_box)
            fprintf_param_value(" 2=%f", confidence_threshold)
            fprintf_param_value(" 3=%f", nms_threshold)
            { if (!op->biases.empty()) fprintf_param_float_array(4, op->biases, pp); }
        }
        else if (layer->type == "Yolov3DetectionOutput")
        {
            tmtool::Yolov3DetectionOutput* op = (tmtool::Yolov3DetectionOutput*)layer;
            tmtool::Yolov3DetectionOutput* op_default = (tmtool::Yolov3DetectionOutput*)layer_default;

            fprintf_param_value(" 0=%d", num_class)
            fprintf_param_value(" 1=%d", num_box)
            fprintf_param_value(" 2=%f", confidence_threshold)
            fprintf_param_value(" 3=%f", nms_threshold)
            { if (!op->biases.empty()) fprintf_param_float_array(4, op->biases, pp); }
            { if (!op->mask.empty()) fprintf_param_int_array(5, op->mask, pp); }
            { if (!op->anchors_scale.empty()) fprintf_param_float_array(6, op->anchors_scale, pp); }
        }

#undef fprintf_param_value

        fprintf(pp, "\n");

        delete layer_default;
    }

    fclose(pp);
    fclose(bp);

    return 0;
}

int ncnn2int8(int argc, char** argv)
{
    if (argc != 6)
    {
        fprintf(stderr, "usage: %s [inparam] [inbin] [outparam] [outbin] [calibration table]\n", argv[0]);
        return -1;
    }

    const char* inparam = argv[1];
    const char* inbin = argv[2];
    const char* outparam = argv[3];
    const char* outbin = argv[4];
    const char* int8scale_table_path = argv[5];

    NetQuantize quantizer;

    // parse the calibration scale table
    if (int8scale_table_path)
    {
        bool s2 = read_int8scale_table(int8scale_table_path, quantizer.blob_int8scale_table, quantizer.weight_int8scale_table);
        if (!s2)
        {
            fprintf(stderr, "read_int8scale_table failed\n");
            return -1;
        }
    }

    quantizer.load_param(inparam);
    quantizer.load_model(inbin);
    
    quantizer.quantize_convolution();
    quantizer.quantize_convolutiondepthwise();
    quantizer.quantize_innerproduct();

    quantizer.save(outparam, outbin);

    return 0;
}
