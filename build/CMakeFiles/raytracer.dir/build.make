# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.22.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.22.2/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/twanbolwerk/Documents/dev/raytracer-in-cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/twanbolwerk/Documents/dev/raytracer-in-cpp/build

# Include any dependencies generated for this target.
include CMakeFiles/raytracer.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/raytracer.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/raytracer.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/raytracer.dir/flags.make

CMakeFiles/raytracer.dir/src/main.cc.o: CMakeFiles/raytracer.dir/flags.make
CMakeFiles/raytracer.dir/src/main.cc.o: ../src/main.cc
CMakeFiles/raytracer.dir/src/main.cc.o: CMakeFiles/raytracer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/twanbolwerk/Documents/dev/raytracer-in-cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/raytracer.dir/src/main.cc.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/raytracer.dir/src/main.cc.o -MF CMakeFiles/raytracer.dir/src/main.cc.o.d -o CMakeFiles/raytracer.dir/src/main.cc.o -c /Users/twanbolwerk/Documents/dev/raytracer-in-cpp/src/main.cc

CMakeFiles/raytracer.dir/src/main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/raytracer.dir/src/main.cc.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/twanbolwerk/Documents/dev/raytracer-in-cpp/src/main.cc > CMakeFiles/raytracer.dir/src/main.cc.i

CMakeFiles/raytracer.dir/src/main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/raytracer.dir/src/main.cc.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/twanbolwerk/Documents/dev/raytracer-in-cpp/src/main.cc -o CMakeFiles/raytracer.dir/src/main.cc.s

# Object files for target raytracer
raytracer_OBJECTS = \
"CMakeFiles/raytracer.dir/src/main.cc.o"

# External object files for target raytracer
raytracer_EXTERNAL_OBJECTS =

raytracer: CMakeFiles/raytracer.dir/src/main.cc.o
raytracer: CMakeFiles/raytracer.dir/build.make
raytracer: CMakeFiles/raytracer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/twanbolwerk/Documents/dev/raytracer-in-cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable raytracer"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/raytracer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/raytracer.dir/build: raytracer
.PHONY : CMakeFiles/raytracer.dir/build

CMakeFiles/raytracer.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/raytracer.dir/cmake_clean.cmake
.PHONY : CMakeFiles/raytracer.dir/clean

CMakeFiles/raytracer.dir/depend:
	cd /Users/twanbolwerk/Documents/dev/raytracer-in-cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/twanbolwerk/Documents/dev/raytracer-in-cpp /Users/twanbolwerk/Documents/dev/raytracer-in-cpp /Users/twanbolwerk/Documents/dev/raytracer-in-cpp/build /Users/twanbolwerk/Documents/dev/raytracer-in-cpp/build /Users/twanbolwerk/Documents/dev/raytracer-in-cpp/build/CMakeFiles/raytracer.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/raytracer.dir/depend

