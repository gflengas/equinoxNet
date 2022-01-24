cmake_minimum_required(VERSION 3.16)
project(equinoxNetW C)

set(CMAKE_C_STANDARD 99)

add_executable(equinoxNetW main.c connected_layer.c connected_layer.h conv_layer.c conv_layer.h custom_math.c custom_math.h core.h data.c data.h maxpool_layer.c maxpool_layer.h softmax_layer.c softmax_layer.h)
target_link_libraries(${PROJECT_NAME} m)