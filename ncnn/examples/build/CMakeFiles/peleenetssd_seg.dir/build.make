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
include CMakeFiles/peleenetssd_seg.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/peleenetssd_seg.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/peleenetssd_seg.dir/flags.make

CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.o: CMakeFiles/peleenetssd_seg.dir/flags.make
CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.o: ../peleenetssd_seg.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/true15/ncnn/examples/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.o -c /home/true15/ncnn/examples/peleenetssd_seg.cpp

CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/true15/ncnn/examples/peleenetssd_seg.cpp > CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.i

CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/true15/ncnn/examples/peleenetssd_seg.cpp -o CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.s

# Object files for target peleenetssd_seg
peleenetssd_seg_OBJECTS = \
"CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.o"

# External object files for target peleenetssd_seg
peleenetssd_seg_EXTERNAL_OBJECTS =

peleenetssd_seg: CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.o
peleenetssd_seg: CMakeFiles/peleenetssd_seg.dir/build.make
peleenetssd_seg: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.9
peleenetssd_seg: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.9
peleenetssd_seg: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.9
peleenetssd_seg: CMakeFiles/peleenetssd_seg.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/true15/ncnn/examples/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable peleenetssd_seg"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/peleenetssd_seg.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/peleenetssd_seg.dir/build: peleenetssd_seg

.PHONY : CMakeFiles/peleenetssd_seg.dir/build

CMakeFiles/peleenetssd_seg.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/peleenetssd_seg.dir/cmake_clean.cmake
.PHONY : CMakeFiles/peleenetssd_seg.dir/clean

CMakeFiles/peleenetssd_seg.dir/depend:
	cd /home/true15/ncnn/examples/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/true15/ncnn/examples /home/true15/ncnn/examples /home/true15/ncnn/examples/build /home/true15/ncnn/examples/build /home/true15/ncnn/examples/build/CMakeFiles/peleenetssd_seg.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/peleenetssd_seg.dir/depend

