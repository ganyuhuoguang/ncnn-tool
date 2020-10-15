/**
 * @File   : darknet2ncnn.h
 * @Author : damone (damonexw@gmail.com)
 * @Link   :
 * @Date   : 10/30/2018, 4:48:57 PM
 */

#ifndef _DARKNET2TMTOOL_H
#define _DARKNET2TMTOOL_H

#include <layer_type.h>
#include <net.h>


int darknet2ncnn(char* darknetcfg, char* darknetweight, char* ncnnparam, char* ncnnbin);
int darknet2ncnn(int argc, char** argv);
/**
 * @brief register support darknet layer to ncnn
 * 
 * @param net 
 */
void register_darknet_layer(tmtool::Net &net);


/**
 * @brief dakrnet custom layer indexs
 */
enum DarknetLayerType{
    Darknet_Activition = tmtool::LayerType::CustomBit  + 1,
    Darknet_ShortCut,
    Darknet_Yolov1,
    Darknet_Yolov3
};

/**
 * @brief get darknet customer layer index
 * @param laer_type
 */
DarknetLayerType get_darknet_layer_type_index(std::string layer_type);


/**
 * @brief register support darknet layer to ncnn, for index register
 * @param net 
 */
void register_darknet_layer_by_index(tmtool::Net &net);


#endif
