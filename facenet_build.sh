#!/bin/bash
#basepath=$(cd `dirname $0`; pwd)
#echo $basepath
#make
#grep "tmtool" /etc/bash.bashrc
#if [ $? -eq 0 ] ;then
#   sed -i '$d' /etc/bash.bashrc
#   sed -i '$a\alias tmtool=\'\'${basepath}'/out/tmtool'\' /etc/bash.bashrc
#   echo "tmtool bash have been updated !"
#else
#  sed -i '$a\alias tmtool=\'\'${basepath}'/out/tmtool'\' /etc/bash.bashrc
#fi
./tmtool ../models/caffe2tm/final.prototxt ../models/caffe2tm/final.caffemodel ../models/caffe2tm/ncnn.param ../models/caffe2tm/ncnn.bin 256 ../models/caffe2tm/final.table
./tmtool ../models/caffe2tm/ncnn.param ../models/caffe2tm/ncnn.bin 16 ../models/caffe2tm/board.ini ZC7100 ZYNQ
