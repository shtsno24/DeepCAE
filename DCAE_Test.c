#include <stdio.h>
#include <stdint.h>

#include "array_printf.h"
#include "layers/padding2d.h"
#include "layers/up_sampling2d.h"
#include "layers/max_pooling2d.h"
#include "layers/conv2d.h"
#include "keras_mnist_DCAE/Conv2D_0.h"
#include "keras_mnist_DCAE/Conv2D_1.h"
#include "keras_mnist_DCAE/Conv2D_2.h"
#include "keras_mnist_DCAE/Conv2D_3.h"

uint16_t input_0_depth = 1, input_0_width = 28, input_0_width = 28;
float input_array[input_0_depth][input_0_height][input_0_width];


uint16_t Padding_0_depth = 1, Padding_0_height = 30, Padding_0_width = 30;
float Padding_0_array[Padding_0_depth][Padding_0_height][Padding_0_width];

uint16_t Conv2d_0_depth = 16, Conv2D_0_height = 28, Conv2D_0_width = 28;
float Conv2D_0_array[Conv2D_0_depth][Conv2D_0_height][Conv2D_0_width];

uint16_t MaxPool_0_depth = 16, MaxPool_0_height = 14, MaxPool_0_width = 14;
float Conv2D_0_array[MaxPool_0_depth][MaxPool_0_height][MaxPool_0_width];


uint16_t Padding_1_depth = 16, Padding_1_width = 16, Padding_1_width = 16;
float Padding_1_array[Padding_1_depth][Padding_1_height][Padding_1_width];

uint16_t Conv2d_1_depth = 16, Conv2D_1_height = 14, Conv2D_1_width = 14;
float Conv2D_0_array[Conv2D_1_depth][Conv2D_1_height][Conv2D_1_width];

uint16_t MaxPool_1_depth = 8, MaxPool_1_height = 7, MaxPool_1_width = 7;

uint16_t UpSampling_0_depth = 8, UpSampling_0_height = 14, UpSampling_0_width = 14;
uint16_t Padding_2_depth = 8, Padding_2_width = 16, Padding_2_width = 16;
uint16_t Conv2d_2_depth = 16, Conv2D_2_height = 14, Conv2D_2_width = 14;

uint16_t UpSampling_1_depth = 16, UpSampling_1_height = 28, UpSampling_1_width = 28;
uint16_t Padding_2_depth = 16, Padding_1_width = 30, Padding_0_width = 30;
uint16_t Conv2d_3_depth = 1, Conv2D_3_height = 28, Conv2D_3_width = 28;

uint16_t output_depth = 1, output_width = 28, output_width = 28;

