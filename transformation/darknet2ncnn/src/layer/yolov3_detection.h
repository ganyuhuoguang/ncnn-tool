/**
 * @File   : yolov3_detection.h
 * @Author : damone (damonexw@gmail.com)
 * @Link   : 
 * @Date   : 11/3/2018, 3:23:53 PM
 */

#ifndef _YOLO_V3_DETECTION_H
#define _YOLO_V3_DETECTION_H

#include "layer.h"

namespace tmtool
{

::tmtool::Layer *Yolov3Detection_layer_creator();
class Yolov3Detection : public Layer
{
public:
  Yolov3Detection();
  ~Yolov3Detection();

  virtual int load_param(const ParamDict &pd);
  virtual int forward(const std::vector<Mat> &bottom_blobs, std::vector<Mat> &top_blobs, const Option &opt) const;

public:
  int classes;
  int box_num;
  int net_width;
  int net_height;
  int softmax_enable;

  float confidence_threshold;
  float nms_threshold;

  Mat biases;

  tmtool::Layer *softmax;
  tmtool::Layer *sigmoid;
};

} // namespace tmtool


#endif