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

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/arlmaster/pulsefromheadmotion

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/arlmaster/pulsefromheadmotion

# Include any dependencies generated for this target.
include CMakeFiles/pfhmain.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/pfhmain.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pfhmain.dir/flags.make

CMakeFiles/pfhmain.dir/src/pfhmain.cpp.o: CMakeFiles/pfhmain.dir/flags.make
CMakeFiles/pfhmain.dir/src/pfhmain.cpp.o: src/pfhmain.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/arlmaster/pulsefromheadmotion/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/pfhmain.dir/src/pfhmain.cpp.o"
	/usr/bin/g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/pfhmain.dir/src/pfhmain.cpp.o -c /home/arlmaster/pulsefromheadmotion/src/pfhmain.cpp

CMakeFiles/pfhmain.dir/src/pfhmain.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pfhmain.dir/src/pfhmain.cpp.i"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/arlmaster/pulsefromheadmotion/src/pfhmain.cpp > CMakeFiles/pfhmain.dir/src/pfhmain.cpp.i

CMakeFiles/pfhmain.dir/src/pfhmain.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pfhmain.dir/src/pfhmain.cpp.s"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/arlmaster/pulsefromheadmotion/src/pfhmain.cpp -o CMakeFiles/pfhmain.dir/src/pfhmain.cpp.s

CMakeFiles/pfhmain.dir/src/pfhmain.cpp.o.requires:
.PHONY : CMakeFiles/pfhmain.dir/src/pfhmain.cpp.o.requires

CMakeFiles/pfhmain.dir/src/pfhmain.cpp.o.provides: CMakeFiles/pfhmain.dir/src/pfhmain.cpp.o.requires
	$(MAKE) -f CMakeFiles/pfhmain.dir/build.make CMakeFiles/pfhmain.dir/src/pfhmain.cpp.o.provides.build
.PHONY : CMakeFiles/pfhmain.dir/src/pfhmain.cpp.o.provides

CMakeFiles/pfhmain.dir/src/pfhmain.cpp.o.provides.build: CMakeFiles/pfhmain.dir/src/pfhmain.cpp.o

# Object files for target pfhmain
pfhmain_OBJECTS = \
"CMakeFiles/pfhmain.dir/src/pfhmain.cpp.o"

# External object files for target pfhmain
pfhmain_EXTERNAL_OBJECTS =

bin/pfhmain: CMakeFiles/pfhmain.dir/src/pfhmain.cpp.o
bin/pfhmain: /opt/ros/hydro/lib/libopencv_calib3d.so
bin/pfhmain: /opt/ros/hydro/lib/libopencv_contrib.so
bin/pfhmain: /opt/ros/hydro/lib/libopencv_core.so
bin/pfhmain: /opt/ros/hydro/lib/libopencv_features2d.so
bin/pfhmain: /opt/ros/hydro/lib/libopencv_flann.so
bin/pfhmain: /opt/ros/hydro/lib/libopencv_gpu.so
bin/pfhmain: /opt/ros/hydro/lib/libopencv_highgui.so
bin/pfhmain: /opt/ros/hydro/lib/libopencv_imgproc.so
bin/pfhmain: /opt/ros/hydro/lib/libopencv_legacy.so
bin/pfhmain: /opt/ros/hydro/lib/libopencv_ml.so
bin/pfhmain: /opt/ros/hydro/lib/libopencv_nonfree.so
bin/pfhmain: /opt/ros/hydro/lib/libopencv_objdetect.so
bin/pfhmain: /opt/ros/hydro/lib/libopencv_photo.so
bin/pfhmain: /opt/ros/hydro/lib/libopencv_stitching.so
bin/pfhmain: /opt/ros/hydro/lib/libopencv_superres.so
bin/pfhmain: /opt/ros/hydro/lib/libopencv_ts.so
bin/pfhmain: /opt/ros/hydro/lib/libopencv_video.so
bin/pfhmain: /opt/ros/hydro/lib/libopencv_videostab.so
bin/pfhmain: lib/libpfhmlib.so
bin/pfhmain: CMakeFiles/pfhmain.dir/build.make
bin/pfhmain: CMakeFiles/pfhmain.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable bin/pfhmain"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pfhmain.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pfhmain.dir/build: bin/pfhmain
.PHONY : CMakeFiles/pfhmain.dir/build

CMakeFiles/pfhmain.dir/requires: CMakeFiles/pfhmain.dir/src/pfhmain.cpp.o.requires
.PHONY : CMakeFiles/pfhmain.dir/requires

CMakeFiles/pfhmain.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pfhmain.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pfhmain.dir/clean

CMakeFiles/pfhmain.dir/depend:
	cd /home/arlmaster/pulsefromheadmotion && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/arlmaster/pulsefromheadmotion /home/arlmaster/pulsefromheadmotion /home/arlmaster/pulsefromheadmotion /home/arlmaster/pulsefromheadmotion /home/arlmaster/pulsefromheadmotion/CMakeFiles/pfhmain.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pfhmain.dir/depend
