# -*- coding: utf-8 -*-

# BUG1989 is pleased to support the open source community by supporting ncnn available.
#
# Copyright (C) 2019 BUG1989. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""
This project modified from BUG1989 version, fixed some setting for dpu and added model structure convertion.
Huiran.Du, 31/10/2019.
"""


from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
import copy
import sys
import os

# to ignore mass information from caffe and etc (option).
os.environ['GLOG_minloglevel'] = '2'
import warnings
warnings.simplefilter("ignore", UserWarning)

import caffe
import caffe.proto.caffe_pb2 as caffe_pb2
import datetime
from google.protobuf import text_format
from scipy import stats

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


# param defination for command line running.
def parse_args():
    parser = argparse.ArgumentParser(description='model calibration for int8 inference')
    parser.add_argument('--struct', dest='struct', help="prototxt", type=str)
    parser.add_argument('--weights', dest='weights', help='caffemodel', type=str)
    parser.add_argument('--mean', dest='mean', help='preprocess mean values', type=float, nargs=3)
    parser.add_argument('--norm', dest='norm', help='normalize value', type=float, nargs=1, default=1.0)
    parser.add_argument('--images', dest='images', help='calibration images directory', type=str)
    parser.add_argument('--gpu', dest='gpu', help='use gpu to forward', type=int, default=0)

    args = parser.parse_args()
    return args, parser


global args, parser
args, parser = parse_args()

# value range of quantization.
QUANTIZE_NUM = 127
# quantity of bin for generating the activations' histogram.
INTERVAL_NUM = 2048

# mark layers for bn merge process
bn_maps = {}
# mark layers for fully-connected and pooling layers for convertion
fc = []
pool = []
# mark layers for quantization
quantize_layer_lists = []


def find_top_after_bn(layers, name, top):
    '''
    Find continuous layers'(BN,Scale) top blob for conv layer.
    :param layers: entire network structure
    :param name: layer name
    :param top: original convolution layer's top blob
    :return: top blob of the merged convolution layer
    '''
    bn_maps[name] = {}
    for l in layers:
        if len(l.bottom) == 0:
            continue
        if l.bottom[0] == top and l.type == "BatchNorm":
            bn_maps[name]["bn"] = l.name
            top = l.top[0]
        if l.bottom[0] == top and l.type == "Scale":
            bn_maps[name]["scale"] = l.name
            top = l.top[0]
    return top


def preprocess_nfcnp(net, expected_proto, new_proto):
    '''
    Convert convolution depth-wise layer to group convolution layer.
    Convert average pooling layer to convolution depth-wise layer.
    Convert fully connected layer to convolution layer.
    Eliminate the dropout layer.
    :param net: the original network init by caffe
    :param expected_proto: the prototxt file of original network
    :param new_proto: propose prototxt file for converted network
    :return: none
    '''
    net_specs = caffe_pb2.NetParameter()
    net_specs2 = caffe_pb2.NetParameter()
    with open(expected_proto, "r") as fp:
        text_format.Merge(str(fp.read()), net_specs)

    net_specs2.MergeFrom(net_specs)
    layers = net_specs.layer
    num_layers = len(layers)

    for i in range(num_layers - 1, -1, -1):
        del net_specs2.layer[i]

    for idx in range(num_layers):
        layer_src = layers[idx]
        if layer_src.type == "InnerProduct":
            name = layer_src.name
            bottom = layer_src.bottom[0]
            top = layer_src.top[0]
            out_c = layer_src.inner_product_param.num_output

            if len(net.blobs[bottom].data.shape) == 2:
                kernel = 4
            elif net.blobs[bottom].data.shape[2] == 1:
                kernel = 4
            else:
                kernel = net.blobs[bottom].data.shape[2]

            reshape = net_specs2.layer.add()
            reshape.name = name + "_reshape"
            reshape.type = "Reshape"
            reshape.bottom.extend([bottom])
            reshape.top.extend([name + "_reshape"])
            reshape.reshape_param.shape.dim.extend([0, -1, kernel, kernel])

            conv = net_specs2.layer.add()
            conv.name = name
            conv.type = "Convolution"
            conv.bottom.extend([name + "_reshape"])
            conv.top.extend([top])
            conv.convolution_param.num_output = out_c
            conv.convolution_param.pad.extend([0])
            conv.convolution_param.kernel_size.extend([kernel])

            fc.append(name)

        elif layer_src.type == "ConvolutionDepthwise":
            name = layer_src.name
            bottom = layer_src.bottom[0]
            top = layer_src.top[0]
            out_c = layer_src.convolution_depthwise_param.num_output
            kernel = layer_src.convolution_depthwise_param.kernel_size
            stride = layer_src.convolution_depthwise_param.stride
            pad = layer_src.convolution_depthwise_param.pad

            conv = net_specs2.layer.add()
            conv.name = name
            conv.type = "Convolution"
            conv.bottom.extend([bottom])
            conv.top.extend([top])
            conv.convolution_param.num_output = out_c
            conv.convolution_param.group = out_c
            conv.convolution_param.kernel_size.extend([kernel])
            conv.convolution_param.stride.extend([stride])
            conv.convolution_param.pad.extend([pad])

        elif layer_src.type == "Pooling":
            if layer_src.pooling_param.pool == 1:
                name = layer_src.name
                bottom = layer_src.bottom[0]
                top = layer_src.top[0]
                out_c = net.blobs[bottom].data.shape[1]
                kernel = layer_src.pooling_param.kernel_size
                stride = layer_src.pooling_param.stride
                pad = layer_src.pooling_param.pad

                conv = net_specs2.layer.add()
                conv.name = name
                conv.type = "Convolution"
                conv.bottom.extend([bottom])
                conv.top.extend([top])
                conv.convolution_param.num_output = out_c
                conv.convolution_param.group = out_c
                conv.convolution_param.kernel_size.extend([kernel])
                conv.convolution_param.stride.extend([stride])
                conv.convolution_param.pad.extend([pad])

                pool.append(name)

            else:
                layer = net_specs2.layer.add()
                layer.MergeFrom(layer_src)

        elif layer_src.type == "Dropout":
            continue

        else:
            layer = net_specs2.layer.add()
            layer.MergeFrom(layer_src)

    with open(new_proto, "w") as fp:
        fp.write("{}".format(net_specs2))


def pre_process_nb(expected_proto, new_proto):
    '''
    Merge batchnorm, scale layer to convolution layer.
    :param expected_proto: the prototxt file of original network
    :param new_proto: propose prototxt file for converted network
    :return: none
    '''
    net_specs = caffe_pb2.NetParameter()
    net_specs2 = caffe_pb2.NetParameter()
    with open(expected_proto, "r") as fp:
        text_format.Merge(str(fp.read()), net_specs)

    net_specs2.MergeFrom(net_specs)
    layers = net_specs.layer
    num_layers = len(layers)

    for i in range(num_layers - 1, -1, -1):
        del net_specs2.layer[i]

    for idx in range(num_layers):
        l = layers[idx]
        if l.type == "BatchNorm" or l.type == "Scale":
            continue
        elif l.type == "Convolution":
            top = find_top_after_bn(layers, l.name, l.top[0])
            bn_maps[l.name]["type"] = l.type
            layer = net_specs2.layer.add()
            layer.MergeFrom(l)
            layer.top[0] = top
            layer.convolution_param.bias_term = True
        else:
            layer = net_specs2.layer.add()
            layer.MergeFrom(l)

    with open(new_proto, "w") as fp:
        fp.write("{}".format(net_specs2))


def load_weights_nfcnp(net1, nofc):
    '''
    Generate and convert weights from original network to new network for fully-connected and average pooling layer.
    :param net1: the original network init by caffe
    :param nofc: the new network init by caffe
    :return: none
    '''
    listkeys = nofc.params.keys()
    for idx, key in enumerate(listkeys):
        if type(nofc.params[key]) is caffe._caffe.BlobVec:
            if (key not in fc) and (key not in pool):
                source = net1.params[key]
                for i, w in enumerate(source):
                    nofc.params[key][i].data[...] = w.data
            elif key in fc:
                oc, ic, a, b = nofc.params[key][0].data.shape
                nofc.params[key][0].data[...] = np.reshape(net1.params[key][0].data, (oc, ic, a, b))
                nofc.params[key][1].data[...] = net1.params[key][1].data
            elif key in pool:
                oc, ic, a, b = nofc.params[key][0].data.shape
                nofc.params[key][0].data[...] = np.ones((oc, ic, a, b)) * (1 / (a * b))
                nofc.params[key][1].data[...] = np.zeros(nofc.params[key][1].data.shape)


def load_weights_nb(net, nobn):
    '''
    Generate weights from original network to new network for batchnorm merging.
    :param net: the original network init by caffe
    :param nobn: the new network init by caffe
    :return: none
    '''
    if sys.version_info > (3, 0):
        listkeys = nobn.params.keys()
    else:
        listkeys = nobn.params.iterkeys()
    for key in listkeys:
        if type(nobn.params[key]) is caffe._caffe.BlobVec:
            conv = net.params[key]
            if key not in bn_maps or "bn" not in bn_maps[key]:
                for i, w in enumerate(conv):
                    nobn.params[key][i].data[...] = w.data
            else:
                # print(key)
                bn = net.params[bn_maps[key]["bn"]]
                scale = net.params[bn_maps[key]["scale"]]
                wt = conv[0].data
                channels = 0
                if bn_maps[key]["type"] == "Convolution":
                    channels = wt.shape[0]
                elif bn_maps[key]["type"] == "Deconvolution":
                    channels = wt.shape[1]
                else:
                    print("error type " + bn_maps[key]["type"])
                    exit(-1)
                bias = np.zeros(channels)
                if len(conv) > 1:
                    bias = conv[1].data
                mean = bn[0].data
                var = bn[1].data
                scalef = bn[2].data

                scales = scale[0].data
                shift = scale[1].data

                if scalef != 0:
                    scalef = 1. / scalef
                mean = mean * scalef
                var = var * scalef
                rstd = 1. / np.sqrt(var + 1e-5)
                if bn_maps[key]["type"] == "Convolution":
                    rstd1 = rstd.reshape((channels, 1, 1, 1))
                    scales1 = scales.reshape((channels, 1, 1, 1))
                    wt = wt * rstd1 * scales1
                else:
                    rstd1 = rstd.reshape((1, channels, 1, 1))
                    scales1 = scales.reshape((1, channels, 1, 1))
                    wt = wt * rstd1 * scales1
                bias = (bias - mean) * rstd * scales + shift

                nobn.params[key][0].data[...] = wt
                nobn.params[key][1].data[...] = bias

    print("\r convert stage 1/5: 100%", end="")


class QuantizeLayer:
    def __init__(self, name, blob_name, group_num):
        self.name = name
        self.blob_name = blob_name
        self.group_num = group_num
        self.weight_scale = np.zeros(group_num)
        self.blob_max = 0.0
        self.blob_distubution_interval = 0.0
        self.blob_distubution = np.zeros(INTERVAL_NUM)
        self.blob_threshold = 0
        self.blob_scale = 1.0
        self.group_zero = np.zeros(group_num)

    def quantize_weight(self, weight_data):
        '''
        Quantize weight for convolution layer.
        :param weight_data: weight array of convolution layer
        :return: none
        '''
        blob_group_data = np.array_split(weight_data, self.group_num)
        for i, group_data in enumerate(blob_group_data):
            max_val = np.max(group_data)
            min_val = np.min(group_data)
            threshold = max(abs(max_val), abs(min_val))
            if threshold < 0.0001:
                self.weight_scale[i] = 0
                self.group_zero[i] = 1
            else:
                self.weight_scale[i] = QUANTIZE_NUM / threshold

    def initial_blob_max(self, blob_data):
        '''
        Get the absolute max value of a blob.
        :param blob_data: output feature blob array
        :return: none
        '''
        max_val = np.max(blob_data)
        min_val = np.min(blob_data)
        self.blob_max = max(self.blob_max, max(abs(max_val), abs(min_val)))

    def initial_blob_distubution_interval(self):
        '''
        Get the interval of blob value for threshold seeking.
        :return: none
        '''
        self.blob_distubution_interval = self.blob_max / INTERVAL_NUM

    def initial_histograms(self, blob_data):
        '''
        Generate histogram of output feature for threshold seeking.
        :param blob_data: outoput feature blob array
        :return: none
        '''
        th = self.blob_max
        hist, hist_edge = np.histogram(blob_data, bins=INTERVAL_NUM, range=(0, th))
        self.blob_distubution += hist

    def quantize_blob(self):
        '''
        Get the feature scale by seeking threshold.
        :return: none
        '''
        distribution = np.array(self.blob_distubution)
        threshold_bin = threshold_distribution(distribution)
        self.blob_threshold = threshold_bin
        threshold = (threshold_bin + 0.5) * self.blob_distubution_interval
        self.blob_scale = QUANTIZE_NUM / threshold
    
    
def threshold_distribution(distribution, target_bin=128):
    '''
    Seek the suitable threshold for quantizing output feature.
    :param distribution: distribution of output feature
    :param target_bin: target margin of quantization
    :return: threshold value for quantizing the output feature
    '''
    distribution = distribution[1:]
    length = distribution.size
    threshold_sum = sum(distribution[target_bin:])
    kl_divergence = np.zeros(length - target_bin)

    for threshold in range(target_bin, length):
        sliced_nd_hist = copy.deepcopy(distribution[:threshold])

        p = sliced_nd_hist.copy()
        p[threshold-1] += threshold_sum
        threshold_sum = threshold_sum - distribution[threshold]

        is_nonzeros = (p != 0).astype(np.int64)
        quantized_bins = np.zeros(target_bin, dtype=np.int64)
        num_merged_bins = sliced_nd_hist.size // target_bin
        
        for j in range(target_bin):
            start = j * num_merged_bins
            stop = start + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()
        quantized_bins[-1] += sliced_nd_hist[target_bin * num_merged_bins:].sum()
        
        q = np.zeros(sliced_nd_hist.size, dtype=np.float64)
        for j in range(target_bin):
            start = j * num_merged_bins
            if j == target_bin - 1:
                stop = -1
            else:
                stop = start + num_merged_bins
            norm = is_nonzeros[start:stop].sum()
            if norm != 0:
                q[start:stop] = float(quantized_bins[j]) / float(norm)
        q[p == 0] = 0
        p[p == 0] = 0.0001
        q[q == 0] = 0.0001

        kl_divergence[threshold - target_bin] = stats.entropy(p, q)

    min_kl_divergence = np.argmin(kl_divergence)
    threshold_value = min_kl_divergence + target_bin

    return threshold_value


def net_forward(net, image_path, transformer):
    '''
    Forward using caffe by injecting preprocessed image.
    :param net: network init by caffe
    :param image_path: file path of image file
    :param transformer: settings for preprocess image
    :return: none
    '''
    image = caffe.io.load_image(image_path)
    net.blobs['data'].data[...] = transformer.preprocess('data', image)
    net.forward()


def file_name(file_dir):
    '''
    Return files list of a folder
    :param file_dir: folder
    :return: files list
    '''
    files_path = []

    for root, d, files in os.walk(file_dir):
        for name in files:
            file_path = root + "/" + name
            files_path.append(file_path)

    return files_path


def network_prepare(net, mean, norm):
    '''
    Build the preprocess settings for network's input.
    :param net: network init by caffe
    :param mean: mean values for input preprocessing
    :param norm: norm value for inut preprocessing
    :return: preprocess settings
    '''
    img_mean = np.array(mean)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', img_mean)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_input_scale('data', norm)

    return transformer  


def weight_quantize(net, net_file, group_on):
    '''
    Quantize weights for convolution layer.
    :param net: network init by caffe
    :param net_file: network structure file (prototxt file)
    :param group_on: for support group convolution
    :return: none
    '''
    params = caffe_pb2.NetParameter()
    with open(net_file) as f:
        text_format.Merge(f.read(), params)

    for i, layer in enumerate(params.layer):
        if layer.type == "Convolution" or layer.type == "ConvolutionDepthwise":
            weight_blob = net.params[layer.name][0].data
            if group_on == 1:
                quanitze_layer = QuantizeLayer(layer.name, layer.bottom[0], layer.convolution_param.num_output)
            else:
                quanitze_layer = QuantizeLayer(layer.name, layer.bottom[0], 1)
            quanitze_layer.quantize_weight(weight_blob)
            quantize_layer_lists.append(quanitze_layer)
        print("\r convert stage 2/5: %d%%" % int((i + 1) / len(params.layer) * 100), end="")
    return None


def activation_quantize(net, transformer, images_files):
    '''
    Quantize the layer features.
    :param net: the original network init by caffe
    :param transformer: settings for input data preprocess
    :param images_files: files list of calibration set
    :return: none
    '''
    for i, image in enumerate(images_files):
        net_forward(net, image, transformer)
        for layer in quantize_layer_lists:
            blob = net.blobs[layer.blob_name].data[0].flatten()
            layer.initial_blob_max(blob)
        print("\r convert stage 3/5: %d%%    " % int((i+1)/len(images_files)*100), end="")
    for layer in quantize_layer_lists:
        layer.initial_blob_distubution_interval()

    for i, image in enumerate(images_files):
        net_forward(net, image, transformer)
        for layer in quantize_layer_lists:
            blob = net.blobs[layer.blob_name].data[0].flatten()
            layer.initial_histograms(blob)
        print("\r convert stage 4/5: %d%%    " % int((i+1)/len(images_files)*100), end="")

    for i, layer in enumerate(quantize_layer_lists):
        layer.quantize_blob()  
        print("\r convert stage 5/5: %d%%    " % int((i+1)/len(quantize_layer_lists)*100), end="")
    return None


def save_calibration_file(calibration_path):
    '''
    Calibration table output.
    :param calibration_path: table output path
    :return: none
    '''
    calibration_file = open(calibration_path, 'w')
    save_temp = []
    for layer in quantize_layer_lists:
        save_string = layer.name + "_param_0"
        for i in range(layer.group_num):
            save_string = save_string + " " + str(layer.weight_scale[i])
        save_temp.append(save_string)

    for layer in quantize_layer_lists:
        save_string = layer.name + " " + str(layer.blob_scale)
        save_temp.append(save_string)

    for data in save_temp:
        calibration_file.write(data + "\n")

    calibration_file.close()


def usage_info():
    '''
    Display error message.
    :return: none
    '''
    print("Params is illegal, Try: toolkit -h")


def main():
    print("\n ** Using tmsdk model convert tool ** \n")
    time_start = datetime.datetime.now()

    if args.struct is None or args.weights is None or args.mean is None or args.images is None:
        usage_info()
        return None

    mean = args.mean
    norm = args.norm[0] if args.norm != 1.0 else 1.0
    images_path = args.images

    group_on = 1

    if args.gpu != 0:
        caffe.set_device(0)
        caffe.set_mode_gpu()

    print("\r convert stage 0/5: 0%", end="")
    net = caffe.Net(args.struct, args.weights, caffe.TEST)
    preprocess_nfcnp(net, args.struct, "temp")
    net_nfcnp = caffe.Net("temp", caffe.TEST)
    load_weights_nfcnp(net, net_nfcnp)
    pre_process_nb("temp", args.struct.split('.')[0] + '-converted.prototxt')
    net_nb = caffe.Net(args.struct.split('.')[0] + '-converted.prototxt', caffe.TEST)
    load_weights_nb(net_nfcnp, net_nb)
    net_nb.save(args.struct.split('.')[0] + '-converted.caffemodel')

    os.remove("temp")

    transformer = network_prepare(net_nb, mean, norm)
    images_files = file_name(images_path)
    weight_quantize(net_nb, args.struct.split('.')[0] + '-converted.prototxt', group_on)
    activation_quantize(net_nb, transformer, images_files)
    save_calibration_file(args.struct.split('.')[0] + '-converted.table')
    time_end = datetime.datetime.now()

    print("\n All convertion done in %s" % (time_end - time_start))


if __name__ == "__main__":
    main()
