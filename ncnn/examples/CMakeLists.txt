
find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
if(NOT OpenCV_FOUND)
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)

set(TMTOOL_EXAMPLE_LINK_LIBRARIES ncnn ${OpenCV_LIBS})
if(TMTOOL_VULKAN)
    list(APPEND TMTOOL_EXAMPLE_LINK_LIBRARIES ${Vulkan_LIBRARY})
endif()

add_executable(squeezenet squeezenet.cpp)
target_link_libraries(squeezenet ${TMTOOL_EXAMPLE_LINK_LIBRARIES})

add_executable(fasterrcnn fasterrcnn.cpp)
target_link_libraries(fasterrcnn ${TMTOOL_EXAMPLE_LINK_LIBRARIES})

add_executable(rfcn rfcn.cpp)
target_link_libraries(rfcn ${TMTOOL_EXAMPLE_LINK_LIBRARIES})

add_executable(yolov2 yolov2.cpp)
target_link_libraries(yolov2 ${TMTOOL_EXAMPLE_LINK_LIBRARIES})

add_executable(yolov3 yolov3.cpp)
target_link_libraries(yolov3 ${TMTOOL_EXAMPLE_LINK_LIBRARIES})

add_executable(mobilenetv2ssdlite mobilenetv2ssdlite.cpp)
target_link_libraries(mobilenetv2ssdlite ${TMTOOL_EXAMPLE_LINK_LIBRARIES})

add_executable(mobilenetssd mobilenetssd.cpp)
target_link_libraries(mobilenetssd ${TMTOOL_EXAMPLE_LINK_LIBRARIES})

add_executable(squeezenetssd squeezenetssd.cpp)
target_link_libraries(squeezenetssd ${TMTOOL_EXAMPLE_LINK_LIBRARIES})

add_executable(shufflenetv2 shufflenetv2.cpp)
target_link_libraries(shufflenetv2 ${TMTOOL_EXAMPLE_LINK_LIBRARIES})

add_executable(peleenetssd_seg peleenetssd_seg.cpp)
target_link_libraries(peleenetssd_seg ${TMTOOL_EXAMPLE_LINK_LIBRARIES})
