// Layer Declaration header
//
// This file is auto-generated by cmake, don't edit it.

#include "layer/absval.h"
namespace tmtool {
class AbsVal_final : virtual public AbsVal
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = AbsVal::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = AbsVal::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(AbsVal_final)
} // namespace tmtool

#include "layer/batchnorm.h"
namespace tmtool {
class BatchNorm_final : virtual public BatchNorm
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = BatchNorm::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = BatchNorm::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(BatchNorm_final)
} // namespace tmtool

#include "layer/bias.h"
namespace tmtool {
class Bias_final : virtual public Bias
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Bias::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Bias::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Bias_final)
} // namespace tmtool

#include "layer/bnll.h"
namespace tmtool {
class BNLL_final : virtual public BNLL
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = BNLL::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = BNLL::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(BNLL_final)
} // namespace tmtool

#include "layer/concat.h"
namespace tmtool {
class Concat_final : virtual public Concat
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Concat::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Concat::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Concat_final)
} // namespace tmtool

#include "layer/convolution.h"
#include "layer/x86/convolution_x86.h"
namespace tmtool {
class Convolution_final : virtual public Convolution, virtual public Convolution_x86
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Convolution::create_pipeline(opt); if (ret) return ret; }
        { int ret = Convolution_x86::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Convolution_x86::destroy_pipeline(opt); if (ret) return ret; }
        { int ret = Convolution::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Convolution_final)
} // namespace tmtool

#include "layer/crop.h"
namespace tmtool {
class Crop_final : virtual public Crop
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Crop::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Crop::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Crop_final)
} // namespace tmtool

#include "layer/deconvolution.h"
namespace tmtool {
class Deconvolution_final : virtual public Deconvolution
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Deconvolution::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Deconvolution::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Deconvolution_final)
} // namespace tmtool

#include "layer/dropout.h"
namespace tmtool {
class Dropout_final : virtual public Dropout
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Dropout::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Dropout::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Dropout_final)
} // namespace tmtool

#include "layer/eltwise.h"
namespace tmtool {
class Eltwise_final : virtual public Eltwise
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Eltwise::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Eltwise::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Eltwise_final)
} // namespace tmtool

#include "layer/elu.h"
namespace tmtool {
class ELU_final : virtual public ELU
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = ELU::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = ELU::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(ELU_final)
} // namespace tmtool

#include "layer/embed.h"
namespace tmtool {
class Embed_final : virtual public Embed
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Embed::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Embed::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Embed_final)
} // namespace tmtool

#include "layer/exp.h"
namespace tmtool {
class Exp_final : virtual public Exp
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Exp::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Exp::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Exp_final)
} // namespace tmtool

#include "layer/flatten.h"
namespace tmtool {
class Flatten_final : virtual public Flatten
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Flatten::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Flatten::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Flatten_final)
} // namespace tmtool

#include "layer/innerproduct.h"
namespace tmtool {
class InnerProduct_final : virtual public InnerProduct
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = InnerProduct::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = InnerProduct::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(InnerProduct_final)
} // namespace tmtool

#include "layer/input.h"
namespace tmtool {
class Input_final : virtual public Input
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Input::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Input::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Input_final)
} // namespace tmtool

#include "layer/log.h"
namespace tmtool {
class Log_final : virtual public Log
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Log::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Log::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Log_final)
} // namespace tmtool

#include "layer/lrn.h"
namespace tmtool {
class LRN_final : virtual public LRN
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = LRN::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = LRN::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(LRN_final)
} // namespace tmtool

#include "layer/memorydata.h"
namespace tmtool {
class MemoryData_final : virtual public MemoryData
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = MemoryData::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = MemoryData::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(MemoryData_final)
} // namespace tmtool

#include "layer/mvn.h"
namespace tmtool {
class MVN_final : virtual public MVN
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = MVN::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = MVN::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(MVN_final)
} // namespace tmtool

#include "layer/pooling.h"
namespace tmtool {
class Pooling_final : virtual public Pooling
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Pooling::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Pooling::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Pooling_final)
} // namespace tmtool

#include "layer/power.h"
namespace tmtool {
class Power_final : virtual public Power
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Power::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Power::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Power_final)
} // namespace tmtool

#include "layer/prelu.h"
namespace tmtool {
class PReLU_final : virtual public PReLU
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = PReLU::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = PReLU::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(PReLU_final)
} // namespace tmtool

#include "layer/proposal.h"
namespace tmtool {
class Proposal_final : virtual public Proposal
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Proposal::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Proposal::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Proposal_final)
} // namespace tmtool

#include "layer/reduction.h"
namespace tmtool {
class Reduction_final : virtual public Reduction
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Reduction::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Reduction::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Reduction_final)
} // namespace tmtool

#include "layer/relu.h"
namespace tmtool {
class ReLU_final : virtual public ReLU
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = ReLU::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = ReLU::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(ReLU_final)
} // namespace tmtool

#include "layer/reshape.h"
namespace tmtool {
class Reshape_final : virtual public Reshape
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Reshape::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Reshape::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Reshape_final)
} // namespace tmtool

#include "layer/roipooling.h"
namespace tmtool {
class ROIPooling_final : virtual public ROIPooling
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = ROIPooling::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = ROIPooling::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(ROIPooling_final)
} // namespace tmtool

#include "layer/region.h"
namespace tmtool {
class Region_final : virtual public Region
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Region::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Region::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Region_final)
} // namespace ncnn

#include "layer/scale.h"
namespace tmtool {
class Scale_final : virtual public Scale
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Scale::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Scale::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Scale_final)
} // namespace tmtool

#include "layer/sigmoid.h"
namespace tmtool {
class Sigmoid_final : virtual public Sigmoid
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Sigmoid::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Sigmoid::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Sigmoid_final)
} // namespace tmtool

#include "layer/slice.h"
namespace tmtool {
class Slice_final : virtual public Slice
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Slice::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Slice::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Slice_final)
} // namespace tmtool

#include "layer/softmax.h"
namespace tmtool {
class Softmax_final : virtual public Softmax
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Softmax::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Softmax::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Softmax_final)
} // namespace tmtool

#include "layer/split.h"
namespace tmtool {
class Split_final : virtual public Split
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Split::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Split::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Split_final)
} // namespace tmtool

#include "layer/tanh.h"
namespace tmtool {
class TanH_final : virtual public TanH
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = TanH::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = TanH::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(TanH_final)
} // namespace tmtool

#include "layer/threshold.h"
namespace tmtool {
class Threshold_final : virtual public Threshold
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Threshold::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Threshold::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Threshold_final)
} // namespace tmtool

#include "layer/binaryop.h"
namespace tmtool {
class BinaryOp_final : virtual public BinaryOp
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = BinaryOp::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = BinaryOp::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(BinaryOp_final)
} // namespace tmtool

#include "layer/unaryop.h"
namespace tmtool {
class UnaryOp_final : virtual public UnaryOp
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = UnaryOp::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = UnaryOp::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(UnaryOp_final)
} // namespace tmtool

#include "layer/convolutiondepthwise.h"
#include "layer/x86/convolutiondepthwise_x86.h"
namespace tmtool {
class ConvolutionDepthWise_final : virtual public ConvolutionDepthWise, virtual public ConvolutionDepthWise_x86
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = ConvolutionDepthWise::create_pipeline(opt); if (ret) return ret; }
        { int ret = ConvolutionDepthWise_x86::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = ConvolutionDepthWise_x86::destroy_pipeline(opt); if (ret) return ret; }
        { int ret = ConvolutionDepthWise::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(ConvolutionDepthWise_final)
} // namespace tmtool

#include "layer/padding.h"
namespace tmtool {
class Padding_final : virtual public Padding
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Padding::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Padding::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Padding_final)
} // namespace tmtool

#include "layer/squeeze.h"
namespace tmtool {
class Squeeze_final : virtual public Squeeze
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Squeeze::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Squeeze::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Squeeze_final)
} // namespace tmtool

#include "layer/expanddims.h"
namespace tmtool {
class ExpandDims_final : virtual public ExpandDims
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = ExpandDims::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = ExpandDims::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(ExpandDims_final)
} // namespace tmtool

#include "layer/normalize.h"
namespace tmtool {
class Normalize_final : virtual public Normalize
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Normalize::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Normalize::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Normalize_final)
} // namespace tmtool

#include "layer/permute.h"
namespace tmtool {
class Permute_final : virtual public Permute
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Permute::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Permute::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Permute_final)
} // namespace tmtool

#include "layer/priorbox.h"
namespace tmtool {
class PriorBox_final : virtual public PriorBox
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = PriorBox::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = PriorBox::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(PriorBox_final)
} // namespace tmtool

#include "layer/detectionoutput.h"
namespace tmtool {
class DetectionOutput_final : virtual public DetectionOutput
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = DetectionOutput::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = DetectionOutput::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(DetectionOutput_final)
} // namespace tmtool

#include "layer/interp.h"
namespace tmtool {
class Interp_final : virtual public Interp
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Interp::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Interp::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Interp_final)
} // namespace tmtool

#include "layer/deconvolutiondepthwise.h"
namespace tmtool {
class DeconvolutionDepthWise_final : virtual public DeconvolutionDepthWise
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = DeconvolutionDepthWise::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = DeconvolutionDepthWise::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(DeconvolutionDepthWise_final)
} // namespace tmtool

#include "layer/shufflechannel.h"
namespace tmtool {
class ShuffleChannel_final : virtual public ShuffleChannel
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = ShuffleChannel::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = ShuffleChannel::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(ShuffleChannel_final)
} // namespace tmtool

#include "layer/instancenorm.h"
namespace tmtool {
class InstanceNorm_final : virtual public InstanceNorm
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = InstanceNorm::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = InstanceNorm::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(InstanceNorm_final)
} // namespace tmtool

#include "layer/clip.h"
namespace tmtool {
class Clip_final : virtual public Clip
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Clip::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Clip::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Clip_final)
} // namespace tmtool

#include "layer/reorg.h"
namespace tmtool {
class Reorg_final : virtual public Reorg
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Reorg::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Reorg::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Reorg_final)
} // namespace tmtool

#include "layer/yolodetectionoutput.h"
namespace tmtool {
class YoloDetectionOutput_final : virtual public YoloDetectionOutput
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = YoloDetectionOutput::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = YoloDetectionOutput::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(YoloDetectionOutput_final)
} // namespace tmtool

#include "layer/quantize.h"
namespace tmtool {
class Quantize_final : virtual public Quantize
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Quantize::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Quantize::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Quantize_final)
} // namespace tmtool

#include "layer/dequantize.h"
namespace tmtool {
class Dequantize_final : virtual public Dequantize
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Dequantize::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Dequantize::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Dequantize_final)
} // namespace tmtool

#include "layer/yolov3detectionoutput.h"
namespace tmtool {
class Yolov3DetectionOutput_final : virtual public Yolov3DetectionOutput
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Yolov3DetectionOutput::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Yolov3DetectionOutput::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Yolov3DetectionOutput_final)
} // namespace tmtool

#include "layer/psroipooling.h"
namespace tmtool {
class PSROIPooling_final : virtual public PSROIPooling
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = PSROIPooling::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = PSROIPooling::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(PSROIPooling_final)
} // namespace tmtool

#include "layer/packing.h"
namespace tmtool {
class Packing_final : virtual public Packing
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Packing::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Packing::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Packing_final)
} // namespace tmtool

#include "layer/requantize.h"
namespace tmtool {
class Requantize_final : virtual public Requantize
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Requantize::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Requantize::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Requantize_final)
} // namespace tmtool

#include "layer/cast.h"
namespace tmtool {
class Cast_final : virtual public Cast
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Cast::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Cast::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Cast_final)
} // namespace tmtool

#include "layer/hardsigmoid.h"
namespace tmtool {
class HardSigmoid_final : virtual public HardSigmoid
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = HardSigmoid::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = HardSigmoid::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(HardSigmoid_final)
} // namespace tmtool

#include "layer/selu.h"
namespace tmtool {
class SELU_final : virtual public SELU
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = SELU::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = SELU::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(SELU_final)
} // namespace tmtool

#include "layer/upsample.h"
namespace tmtool {
class Upsample_final : virtual public Upsample
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Upsample::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Upsample::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Upsample_final)
} // namespace tmtool

#include "layer/yolov3detectionmodifiedoutput.h"
namespace tmtool {
class Yolov3DetectionModifiedOutput_final : virtual public Yolov3DetectionModifiedOutput
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Yolov3DetectionModifiedOutput::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Yolov3DetectionModifiedOutput::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Yolov3DetectionModifiedOutput_final)
} // namespace tmtool


