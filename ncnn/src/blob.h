// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef TMTOOL_BLOB_H
#define TMTOOL_BLOB_H

#include <string>
#include <vector>
#include "platform.h"

namespace tmtool {

class Blob
{
public:
    // empty
    Blob();

public:
#if TMTOOL_STRING
    // blob name
    std::string name;
#endif // TMTOOL_STRING
     //在网络层传递的过程中，进行数据流动的方式是通过自定义的blob实现的，对于blob通过生产者编号和消费者编号进行定义，
     //producer表示输出该blob的网络层编号，consumers表示以该blob作为输入的网络层编号，前者只能是一个制造者，后者可以是多个使用者。
    // layer index which produce this blob as output
    int producer;
    // layer index which need this blob as input
    std::vector<int> consumers;
};

} // namespace tmtool

#endif // TMTOOL_BLOB_H
