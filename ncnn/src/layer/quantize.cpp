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

#include "quantize.h"

#include <math.h>
//#define debug 
#ifdef debug
static int lastchannels = 0;
static int lastsize = 0;
static int convdw_index = 0;
static int dw_sperated = 0;
static std::vector<signed char> sbuf;
#endif

namespace tmtool {

DEFINE_LAYER_CREATOR(Quantize)

Quantize::Quantize()
{
    one_blob_only = true;
    support_inplace = false;
}

int Quantize::load_param(const ParamDict& pd)
{
    scale = pd.get(0, 1.f);

    return 0;
}

static inline signed char float2int8(float v)
{
    int int32 = round(v);
    if (int32 > 127) return 127;
    if (int32 < -128) return -128;
    return (signed char)int32;
}

#ifdef debug
static char print_feature(int size,int channels,const int* ptr)
{
        FILE *wscalefile;
        char PrintResult[40];
        static int i = 1;
        if (channels == 1)
        {
           convdw_index +=  size;
        }
        
        if((channels != 1)&&(lastchannels == 1))
        {
            sprintf(PrintResult,"./feature/3_quantize_dw.txt");
            wscalefile = fopen(PrintResult,"w");

            //for depthwise
            for(int outid=0; outid<2048/8; outid++)
			{
				for(int kid=0; kid<lastsize; kid++)
				{
					for(int w1=0; w1<8; w1++)
					{
                        if(w1<4)
                        {                 
                            fprintf(wscalefile,"%02x",sbuf[outid*lastsize*8+(3-w1)*lastsize+kid]&0xff);
                            if(w1 == 3)
                            {
                                fprintf(wscalefile,"\n");
                            }
                        }								
                        else if((w1>3)&&(w1<8))
                        {
                            fprintf(wscalefile,"%02x",sbuf[outid*lastsize*8+(11-w1)*lastsize+kid]&0xff);
                            if(w1 == 7)
                            {
                                    fprintf(wscalefile,"\n"); 
                            }
                        }   
					 }
				}
			}
        }
        else
        {

          
        }
        lastsize = size;
        lastchannels = channels;
        if(channels == 1)//depthwise
        {
            sprintf(PrintResult,"./feature/dw/%d.txt", dw_sperated++);
            wscalefile = fopen(PrintResult,"w");
            for (int i=0; i<size; i++)
            {
                fprintf(wscalefile,"%02x\n",ptr[i]);
            }
        }
        else 
        {
            sprintf(PrintResult,"./feature/3_quantize%d.txt",i);
            i++;
            wscalefile = fopen(PrintResult,"w");


            if((channels >= 8))
            {
                for (int i=0; i<size*channels/4; i++)
                {
                    for(int k=0;k<4;k++)
                    {
                        fprintf(wscalefile,"%02x",ptr[(4*(i+1)-k-1)]&0xff);
                    }
                    fprintf(wscalefile,"\n");  
                }
            }

            if(channels == 3)//first conv
            {
                for (int i=0; i<size*channels/3; i++)
                {
                    fprintf(wscalefile,"00");
                    for(int k=0;k<3;k++)
                    {
                        fprintf(wscalefile,"%02x",ptr[(3*(i+1)-k-1)]&0xff);
                    }
                    fprintf(wscalefile,"\n");  
                }
            }
            
        }
        fclose(wscalefile);
        return 0;
}
#endif

int Quantize::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int dims = bottom_blob.dims;

    if (dims == 1)
    {
        int w = bottom_blob.w;

        top_blob.create(w, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const float* ptr = bottom_blob;
        signed char* outptr = top_blob;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i=0; i<w; i++)
        {
            outptr[i] = float2int8(ptr[i] * scale);
        }
    }

    if (dims == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int size = w * h;

        top_blob.create(w, h, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const float* ptr = bottom_blob;
        signed char* outptr = top_blob;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i=0; i<size; i++)
        {
            outptr[i] = float2int8(ptr[i] * scale);
        }
    }

    if (dims == 3)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;
        int size = w * h;
        int *cPrintBuf = (int *)malloc(size*channels*4); 

        top_blob.create(w, h, channels, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)

        #ifdef debug
        if(lastchannels != 1 && channels == 1)
        {
            convdw_index = 0;
            sbuf.resize(2048*size);
        }
        #endif

        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            signed char* outptr = top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                outptr[i] = float2int8(ptr[i] * scale);
                if (channels >= 8)
                {
                    cPrintBuf[(q/8)*8*size+8*i+q%8]  = float2int8(ptr[i] * scale);
                }
                if (channels == 3)//first conv
                {
                    cPrintBuf[3*i+q]  = float2int8(ptr[i] * scale);
                } 
                if (channels == 1)//depthwise
                {
                    cPrintBuf[i]  = float2int8(ptr[i] * scale);
                    #ifdef debug
                    sbuf[i+convdw_index] = float2int8(ptr[i] * scale);
                    #endif
                } 
            }
        }
        #ifdef debug
        print_feature(size,channels,cPrintBuf);
        #endif
        free(cPrintBuf);
    }

    return 0;
}

} // namespace tmtool
