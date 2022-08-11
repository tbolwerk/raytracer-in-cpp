if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif()
cmake_minimum_required(VERSION 3.14)

project(raytracer)
set(CMAKE_CXX_STANDARD 17)

if(MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17 /W4 /O2")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -std=c++17  " )
	if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
	else()
		#Nothing for gcc
	endif()
endif()
add_executable(raytracer src/main.cc)

