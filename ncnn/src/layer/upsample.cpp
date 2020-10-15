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

#include "upsample.h"


namespace tmtool {
#define CHANNEL  8
DEFINE_LAYER_CREATOR(Upsample)

Upsample::Upsample()
{
    one_blob_only = true;
    support_inplace = false;
}

int Upsample::load_param(const ParamDict& pd)
{
    scale = pd.get(0, 0);

    return 0;
}

void upsample_cpu(const Mat& in, int w, int h, int c, int stride, Mat& out)
{
    int i, j, k;
    for(k = 0; k < c; ++k){
        const float* ptr = in.channel(k);
        float* outptr = out.channel(k);
            for(j = 0; j < h*stride; ++j){
                for(i = 0; i < w*stride; ++i){
                    int in_index = (j/stride)*w + i/stride;
                    int out_index = j*w*stride + i;
                    outptr[out_index] = ptr[in_index];
            }
        }
    }
}

int Upsample::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    int outw = w * scale;
    int outh = h * scale;
    int outc = channels;

    top_blob.create(outw, outh, outc, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    //#pragma omp parallel for num_threads(opt.num_threads)
    upsample_cpu(bottom_blob, w, h, channels, scale, top_blob);
    // reorg_cpu(bottom_blob, w, h, channels, stride, 0, top_blob);
    float *cPrintBuf = (float *)malloc(outc*outw*outh*4);

    FILE *a;
    char PrintResult[40];
    static int i = 1;
    sprintf(PrintResult,"./upsample_%d.txt",i++);
    a = fopen(PrintResult, "w");
    for(int outid=0;outid<outc; outid++)
    {
        const float* ptr = top_blob.channel(outid);
        for(int kid = 0;kid < outw*outh;kid++)
        {
            cPrintBuf[(outid/8)*CHANNEL*outw*outh+CHANNEL*kid+outid%CHANNEL]  = ptr[kid];
        }
    }
    for (int j=0; j<outc*outw*outh; j++)
    {
            char pc[4];
            memcpy(&pc,cPrintBuf+j,4);
           // fprintf(a, "%f",cPrintBuf[j]);
            for(int k=0;k<4;k++)
            {
                fprintf(a, "%02x", pc[3-k]&0xff);
            }
            fprintf(a,"\n");
    }
    free(cPrintBuf);
    fclose(a);
    return 0;
}

} // namespace ncnn
