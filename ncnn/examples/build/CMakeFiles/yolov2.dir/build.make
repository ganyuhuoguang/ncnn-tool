# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/true15/ncnn/examples

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/true15/ncnn/examples/build

# Include any dependencies generated for this target.
include CMakeFiles/yolov2.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/yolov2.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/yolov2.dir/flags.make

CMakeFiles/yolov2.dir/yolov2.o: CMakeFiles/yolov2.dir/flags.make
CMakeFiles/yolov2.dir/yolov2.o: ../yolov2.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/true15/ncnn/examples/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/yolov2.dir/yolov2.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/yolov2.dir/yolov2.o -c /home/true15/ncnn/examples/yolov2.cpp

CMakeFiles/yolov2.dir/yolov2.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolov2.dir/yolov2.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/true15/ncnn/examples/yolov2.cpp > CMakeFiles/yolov2.dir/yolov2.i

CMakeFiles/yolov2.dir/yolov2.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolov2.dir/yolov2.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/true15/ncnn/examples/yolov2.cpp -o CMakeFiles/yolov2.dir/yolov2.s

# Object files for target yolov2
yolov2_OBJECTS = \
"CMakeFiles/yolov2.dir/yolov2.o"

# External object files for target yolov2
yolov2_EXTERNAL_OBJECTS =

yolov2: CMakeFiles/yolov2.dir/yolov2.o
yolov2: CMakeFiles/yolov2.dir/build.make
yolov2: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.9
yolov2: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.9
yolov2: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.9
yolov2: CMakeFiles/yolov2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/true15/ncnn/examples/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable yolov2"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/yolov2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/yolov2.dir/build: yolov2

.PHONY : CMakeFiles/yolov2.dir/build

CMakeFiles/yolov2.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/yolov2.dir/cmake_clean.cmake
.PHONY : CMakeFiles/yolov2.dir/clean

CMakeFiles/yolov2.dir/depend:
	cd /home/true15/ncnn/examples/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/true15/ncnn/examples /home/true15/ncnn/examples /home/true15/ncnn/examples/build /home/true15/ncnn/examples/build /home/true15/ncnn/examples/build/CMakeFiles/yolov2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/yolov2.dir/depend

