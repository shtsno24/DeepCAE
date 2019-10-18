/*
 * author : shtsno24
 * Date : 2019-10-18 16:49:00.537420
 *
 */
#pragma once
#include <stdint.h>

uint16_t Padding2D_0_depth = 1, Padding2D_0_height = 30, Padding2D_0_width = 30;
int16_t Padding2D_0_array[1][30][30];

uint16_t Conv2D_0_depth = 16, Conv2D_0_height = 28, Conv2D_0_width = 28;
int16_t Conv2D_0_array[16][28][28];

uint16_t MaxPooling2D_0_depth = 16, MaxPooling2D_0_height = 14, MaxPooling2D_0_width = 14;
int16_t MaxPooling2D_0_array[16][14][14];

uint16_t Padding2D_1_depth = 16, Padding2D_1_height = 16, Padding2D_1_width = 16;
int16_t Padding2D_1_array[16][16][16];

uint16_t Conv2D_1_depth = 8, Conv2D_1_height = 14, Conv2D_1_width = 14;
int16_t Conv2D_1_array[8][14][14];

uint16_t MaxPooling2D_1_depth = 8, MaxPooling2D_1_height = 7, MaxPooling2D_1_width = 7;
int16_t MaxPooling2D_1_array[8][7][7];

uint16_t Padding2D_2_depth = 8, Padding2D_2_height = 9, Padding2D_2_width = 9;
int16_t Padding2D_2_array[8][9][9];

uint16_t Conv2D_2_depth = 8, Conv2D_2_height = 7, Conv2D_2_width = 7;
int16_t Conv2D_2_array[8][7][7];

uint16_t UpSampling2D_0_depth = 8, UpSampling2D_0_height = 14, UpSampling2D_0_width = 14;
int16_t UpSampling2D_0_array[8][14][14];

uint16_t Padding2D_3_depth = 8, Padding2D_3_height = 16, Padding2D_3_width = 16;
int16_t Padding2D_3_array[8][16][16];

uint16_t Conv2D_3_depth = 16, Conv2D_3_height = 14, Conv2D_3_width = 14;
int16_t Conv2D_3_array[16][14][14];

uint16_t UpSampling2D_1_depth = 16, UpSampling2D_1_height = 28, UpSampling2D_1_width = 28;
int16_t UpSampling2D_1_array[16][28][28];

uint16_t Padding2D_4_depth = 16, Padding2D_4_height = 30, Padding2D_4_width = 30;
int16_t Padding2D_4_array[16][30][30];

uint16_t Conv2D_4_depth = 1, Conv2D_4_height = 28, Conv2D_4_width = 28;
int16_t Conv2D_4_array[1][28][28];
