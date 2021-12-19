cmake_minimum_required(VERSION 3.20)
project(deviltest C)

set(CMAKE_C_STANDARD 99)

add_executable(deviltest main.c conv_layer.c conv_layer.h custom_math.c custom_math.h core.h maxpool_layer.c maxpool_layer.h connected_layer.c connected_layer.h softmax_layer.c softmax_layer.h data.c data.h network.c network.h)
