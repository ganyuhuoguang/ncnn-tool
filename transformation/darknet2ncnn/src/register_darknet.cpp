/**
 * @File   : register_darknet.cpp
 * @Author : damone (damonexw@gmail.com)
 * @Link   :
 * @Date   : 10/30/2018, 4:52:30 PM
 */

#include "darknet2ncnn.h"
#include "layer/darknet_activation.h"
#include "layer/darknet_shortcut.h"
#include "layer/yolov1_detection.h"
#include "layer/yolov3_detection.h"

void register_darknet_layer(tmtool::Net &net)
{
  net.register_custom_layer("DarknetActivation", tmtool::DarknetActivation_layer_creator);
  net.register_custom_layer("DarknetShortcut", tmtool::DarknetShortcut_layer_creator);
  net.register_custom_layer("Yolov1Detection", tmtool::Yolov1Detection_layer_creator);
  net.register_custom_layer("Yolov3Detection", tmtool::Yolov3Detection_layer_creator);
}

DarknetLayerType get_darknet_layer_type_index(std::string layer_type)
{
  if ("DarknetActivation" == layer_type)
    return Darknet_Activition;
  else if ("DarknetShortcut" == layer_type)
    return Darknet_ShortCut;
  else if ("Yolov1Detection" == layer_type)
    return Darknet_Yolov1;
  else if ("Yolov3Detection" == layer_type)
    return Darknet_Yolov3;

  return (DarknetLayerType)tmtool::LayerType::CustomBit;
}

void register_darknet_layer_by_index(tmtool::Net &net)
{
  net.register_custom_layer(Darknet_Activition, tmtool::DarknetActivation_layer_creator);
  net.register_custom_layer(Darknet_ShortCut, tmtool::DarknetShortcut_layer_creator);
  net.register_custom_layer(Darknet_Yolov1, tmtool::Yolov1Detection_layer_creator);
  net.register_custom_layer(Darknet_Yolov3, tmtool::Yolov3Detection_layer_creator);
}
