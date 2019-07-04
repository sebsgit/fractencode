cmake_minimum_required(VERSION 3.1)

set(PROJECT_NAME "fractal_encode")
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 20)

add_executable(${PROJECT_NAME} "")

target_compile_options(${PROJECT_NAME} PUBLIC -Wall -Werror)

target_include_directories(${PROJECT_NAME}
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}
		${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/gsl/include/)

target_sources(${PROJECT_NAME} PRIVATE main.cpp)

target_sources(${PROJECT_NAME} PRIVATE
        image/sampler.cpp
        image/transform.cpp
        utils/utils.cpp
		utils/sse_debug.cpp
        image/imageutils.cpp
        image/ImageIO.cpp
		image/ImageStatistics.cpp
		image/partition/gridpartition.cpp 
        image/partition/quadtreepartition.cpp
        process/gaussian5x5.cpp
        process/sobel.cpp
        encode/edgeclassifier.cpp
		encode/Classifier2.cpp
		encode/EncodingEngine.cpp
		encode/EncodingEngine2.cpp)

if(NOT NO_AVX)
	target_compile_definitions(${PROJECT_NAME} PRIVATE "FRAC_WITH_AVX=1")
	set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-msse -msse2 -mavx -mavx2 -march=native")
	target_sources(${PROJECT_NAME} PRIVATE utils/sse_utils.cpp)
endif()

set(STBI_LIB "stbi_io")
add_library(${STBI_LIB} STATIC thirdparty/stb_image/stb_image_impl.c)
# suppress fopen warning
set_target_properties(${STBI_LIB} PROPERTIES COMPILE_FLAGS "-Wno-deprecated-declarations")
target_link_libraries(${PROJECT_NAME} ${STBI_LIB})
