# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/cmake-gui

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zhenyang/Workspace/devel/project/vision/arctic_action_recog/extract_features/dense_flow

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zhenyang/Workspace/devel/project/vision/arctic_action_recog/extract_features/dense_flow/build_resize

# Include any dependencies generated for this target.
include CMakeFiles/resize_flow.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/resize_flow.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/resize_flow.dir/flags.make

CMakeFiles/resize_flow.dir/resize_flow.cpp.o: CMakeFiles/resize_flow.dir/flags.make
CMakeFiles/resize_flow.dir/resize_flow.cpp.o: ../resize_flow.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/zhenyang/Workspace/devel/project/vision/arctic_action_recog/extract_features/dense_flow/build_resize/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/resize_flow.dir/resize_flow.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/resize_flow.dir/resize_flow.cpp.o -c /home/zhenyang/Workspace/devel/project/vision/arctic_action_recog/extract_features/dense_flow/resize_flow.cpp

CMakeFiles/resize_flow.dir/resize_flow.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/resize_flow.dir/resize_flow.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/zhenyang/Workspace/devel/project/vision/arctic_action_recog/extract_features/dense_flow/resize_flow.cpp > CMakeFiles/resize_flow.dir/resize_flow.cpp.i

CMakeFiles/resize_flow.dir/resize_flow.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/resize_flow.dir/resize_flow.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/zhenyang/Workspace/devel/project/vision/arctic_action_recog/extract_features/dense_flow/resize_flow.cpp -o CMakeFiles/resize_flow.dir/resize_flow.cpp.s

CMakeFiles/resize_flow.dir/resize_flow.cpp.o.requires:
.PHONY : CMakeFiles/resize_flow.dir/resize_flow.cpp.o.requires

CMakeFiles/resize_flow.dir/resize_flow.cpp.o.provides: CMakeFiles/resize_flow.dir/resize_flow.cpp.o.requires
	$(MAKE) -f CMakeFiles/resize_flow.dir/build.make CMakeFiles/resize_flow.dir/resize_flow.cpp.o.provides.build
.PHONY : CMakeFiles/resize_flow.dir/resize_flow.cpp.o.provides

CMakeFiles/resize_flow.dir/resize_flow.cpp.o.provides.build: CMakeFiles/resize_flow.dir/resize_flow.cpp.o

# Object files for target resize_flow
resize_flow_OBJECTS = \
"CMakeFiles/resize_flow.dir/resize_flow.cpp.o"

# External object files for target resize_flow
resize_flow_EXTERNAL_OBJECTS =

resize_flow: CMakeFiles/resize_flow.dir/resize_flow.cpp.o
resize_flow: CMakeFiles/resize_flow.dir/build.make
resize_flow: /home/zhenyang/local/lib/libopencv_videostab.so.2.4.11
resize_flow: /home/zhenyang/local/lib/libopencv_video.so.2.4.11
resize_flow: /home/zhenyang/local/lib/libopencv_ts.a
resize_flow: /home/zhenyang/local/lib/libopencv_superres.so.2.4.11
resize_flow: /home/zhenyang/local/lib/libopencv_stitching.so.2.4.11
resize_flow: /home/zhenyang/local/lib/libopencv_photo.so.2.4.11
resize_flow: /home/zhenyang/local/lib/libopencv_ocl.so.2.4.11
resize_flow: /home/zhenyang/local/lib/libopencv_objdetect.so.2.4.11
resize_flow: /home/zhenyang/local/lib/libopencv_nonfree.so.2.4.11
resize_flow: /home/zhenyang/local/lib/libopencv_ml.so.2.4.11
resize_flow: /home/zhenyang/local/lib/libopencv_legacy.so.2.4.11
resize_flow: /home/zhenyang/local/lib/libopencv_imgproc.so.2.4.11
resize_flow: /home/zhenyang/local/lib/libopencv_highgui.so.2.4.11
resize_flow: /home/zhenyang/local/lib/libopencv_gpu.so.2.4.11
resize_flow: /home/zhenyang/local/lib/libopencv_flann.so.2.4.11
resize_flow: /home/zhenyang/local/lib/libopencv_features2d.so.2.4.11
resize_flow: /home/zhenyang/local/lib/libopencv_core.so.2.4.11
resize_flow: /home/zhenyang/local/lib/libopencv_contrib.so.2.4.11
resize_flow: /home/zhenyang/local/lib/libopencv_calib3d.so.2.4.11
resize_flow: /home/zhenyang/local/lib/libopencv_nonfree.so.2.4.11
resize_flow: /home/zhenyang/local/lib/libopencv_ocl.so.2.4.11
resize_flow: /home/zhenyang/local/lib/libopencv_gpu.so.2.4.11
resize_flow: /home/zhenyang/local/lib/libopencv_photo.so.2.4.11
resize_flow: /home/zhenyang/local/lib/libopencv_objdetect.so.2.4.11
resize_flow: /home/zhenyang/local/lib/libopencv_legacy.so.2.4.11
resize_flow: /home/zhenyang/local/lib/libopencv_video.so.2.4.11
resize_flow: /home/zhenyang/local/lib/libopencv_ml.so.2.4.11
resize_flow: /home/zhenyang/local/lib/libopencv_calib3d.so.2.4.11
resize_flow: /home/zhenyang/local/lib/libopencv_features2d.so.2.4.11
resize_flow: /home/zhenyang/local/lib/libopencv_highgui.so.2.4.11
resize_flow: /home/zhenyang/local/lib/libopencv_imgproc.so.2.4.11
resize_flow: /home/zhenyang/local/lib/libopencv_flann.so.2.4.11
resize_flow: /home/zhenyang/local/lib/libopencv_core.so.2.4.11
resize_flow: CMakeFiles/resize_flow.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable resize_flow"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/resize_flow.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/resize_flow.dir/build: resize_flow
.PHONY : CMakeFiles/resize_flow.dir/build

CMakeFiles/resize_flow.dir/requires: CMakeFiles/resize_flow.dir/resize_flow.cpp.o.requires
.PHONY : CMakeFiles/resize_flow.dir/requires

CMakeFiles/resize_flow.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/resize_flow.dir/cmake_clean.cmake
.PHONY : CMakeFiles/resize_flow.dir/clean

CMakeFiles/resize_flow.dir/depend:
	cd /home/zhenyang/Workspace/devel/project/vision/arctic_action_recog/extract_features/dense_flow/build_resize && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zhenyang/Workspace/devel/project/vision/arctic_action_recog/extract_features/dense_flow /home/zhenyang/Workspace/devel/project/vision/arctic_action_recog/extract_features/dense_flow /home/zhenyang/Workspace/devel/project/vision/arctic_action_recog/extract_features/dense_flow/build_resize /home/zhenyang/Workspace/devel/project/vision/arctic_action_recog/extract_features/dense_flow/build_resize /home/zhenyang/Workspace/devel/project/vision/arctic_action_recog/extract_features/dense_flow/build_resize/CMakeFiles/resize_flow.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/resize_flow.dir/depend

