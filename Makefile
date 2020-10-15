CC:=gcc
CPLUSPLUS:=g++ -std=c++11

LIB := -L$(PWD)/lib/lib

GET_LOCAL_DIR    = $(patsubst %/,%,$(dir $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))))
MKDIR = if [ ! -d $(dir $@) ]; then mkdir -p $(dir $@); fi
MKDIRDATA = if [ ! -d $(dir $(BUILDDATA)) ]; then mkdir -p $(dir $(BUILDDATA)); fi
TOBUILDDIR = $(addprefix $(BUILDDIR)/,$(1))

LOCAL_DIR := $(GET_LOCAL_DIR)

BUILDOUT ?= $(LOCAL_DIR)/out
BUILDDATA ?= $(LOCAL_DIR)/out/data
BUILDDIR := $(BUILDOUT)
MODELDIR := $(LOCAL_DIR)/models

#copy tmcnn model to output directory
define copyModelToOut
	@echo "Ready to copy tmcnn model to output. This op may take a while!"
	test -d $(BUILDDIR)/models || mkdir -p $(BUILDDIR)/models
	@cp -r $(MODELDIR) $(BUILDDIR)
endef

INC_FLAGS := -I$(PWD)/lib/include
INC_FLAGS += -I$(PWD)/lib/include/google
INC_FLAGS += -I$(PWD)/lib/include/opencv
INC_FLAGS += -I$(PWD)/lib/include/ncnn

INC_FLAGS += -I$(PWD)/src
INC_FLAGS += -I$(PWD)/src/layer
INC_FLAGS += -I$(PWD)/ncnn
INC_FLAGS += -I$(PWD)/ncnn/benchmark
INC_FLAGS += -I$(PWD)/ncnn/examples
INC_FLAGS += -I$(PWD)/mtcnn
INC_FLAGS += -I$(PWD)/transformation
INC_FLAGS += -I$(PWD)/transformation/caffe2ncnn
INC_FLAGS += -I$(PWD)/transformation/mxnet2ncnn
INC_FLAGS += -I$(PWD)/transformation/onnx2ncnn
INC_FLAGS += -I$(PWD)/transformation/onnx2ncnn/build
INC_FLAGS += -I$(PWD)/transformation/quantize
INC_FLAGS += -I$(PWD)/transformation/darknet2ncnn/include
INC_FLAGS += -I$(PWD)/transformation/darknet2ncnn/src
INC_FLAGS += -I$(PWD)/transformation/darknet2ncnn/src/layer
INC_FLAGS += -I$(PWD)/transformation/darknet2ncnn/src/include

INC_FLAGS += -I$(PWD)/ncnn/src
INC_FLAGS += -I$(PWD)/ncnn/src/layer
INC_FLAGS += -I$(PWD)/ncnn/src/layer/arm
INC_FLAGS += -I$(PWD)/ncnn/src/layer/vulkan
INC_FLAGS += -I$(PWD)/ncnn/src/layer/x86

LDFLAGS := `pkg-config --libs opencv`

CFLAGS := -Wall -g $(INC_FLAGS) -lpthread -lz
CFLAGS += -lprotobuf-lite -lprotobuf
CFLAGS += `pkg-config --cflags opencv`
CFLAGS += -march=native
CFLAGS += -ldarknet -lncnn -lm -Ofast
CFLAGS += -fopenmp

SRCS := $(wildcard ./src/*.cpp)
SRCS += $(wildcard ./src/layer/*.cpp)
SRCS += $(wildcard ./ncnn/benchmark/*.cpp)
SRCS += $(wildcard ./ncnn/examples/*.cpp)
SRCS += $(wildcard ./mtcnn/*.cpp)
SRCS += $(wildcard ./ncnn/src/*.cpp)
SRCS += $(wildcard ./transformation/*.cpp)
SRCS += $(wildcard ./transformation/caffe2ncnn/*.cpp)
SRCS += $(wildcard ./transformation/mxnet2ncnn/*.cpp)
SRCS += $(wildcard ./transformation/onnx2ncnn/*.cpp)
SRCS += $(wildcard ./transformation/onnx2ncnn/build/*.cpp)
SRCS += $(wildcard ./transformation/quantize/*.cpp)
SRCS += $(wildcard ./transformation/darknet2ncnn/src/*.cpp)
SRCS += $(wildcard ./transformation/darknet2ncnn/src/layer/*.cpp)
SRCS += $(wildcard ./ncnn/src/layer/*.cpp)
SRCS += $(wildcard ./ncnn/src/layer/x86/*.cpp)

TARGET := tmtool
BIN := $(BUILDDIR)/$(TARGET)

MODULE_CPPOBJS := $(call TOBUILDDIR,$(patsubst %.cpp,%.o,$(SRCS)))
.PHONY : clean all

all: $(BIN)

$(BIN): $(MODULE_CPPOBJS)
	@$(CPLUSPLUS) -o $@ $^ $(LIB) $(CFLAGS) $(LDFLAGS)
	@$(call copyModelToOut)
	@echo "build tm-tool success."
	
$(MODULE_CPPOBJS): $(BUILDDIR)/%.o: %.cpp
	@$(MKDIR)
	@$(CPLUSPLUS) $(CFLAGS) -c $< -o $@

#.c.o:	
#	$(CC) -c $(CFLAGS) $< -o $@
#.cpp.o:
#	$(CPLUSPLUS) $(CFLAGS) -c $< -o $@
clean:
	@rm -rf out
