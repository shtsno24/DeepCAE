/*
 * author : shtsno24
 * Date : 2019-10-15 13:42:28.659810
 *
 */
#pragma once
#include <stdint.h>

int main(void){
uint16_t Padding2D_0_depth = 0, Padding2D_0_height = 0, Padding2D_0_width = 0;
float Padding2D_0_array[0][30][30];

uint16_t Conv2D_0_depth = 16, Conv2D_0_height = 28, Conv2D_0_width = 28;
float Conv2D_0_array[16][28][28];

uint16_t MaxPooling2D_0_depth = 16, MaxPooling2D_0_height = 14, MaxPooling2D_0_width = 14;
float MaxPooling2D_0_array[16][14][14];

uint16_t Padding2D_1_depth = 16, Padding2D_1_height = 14, Padding2D_1_width = 14;
float Padding2D_1_array[16][16][16];

uint16_t Conv2D_1_depth = 8, Conv2D_1_height = 14, Conv2D_1_width = 14;
float Conv2D_1_array[8][14][14];

uint16_t MaxPooling2D_1_depth = 8, MaxPooling2D_1_height = 7, MaxPooling2D_1_width = 7;
float MaxPooling2D_1_array[8][7][7];

uint16_t UpSampling2D_0_depth = 8, UpSampling2D_0_height = 14, UpSampling2D_0_width = 14;
float UpSampling2D_0_array[8][14][14];

uint16_t Padding2D_2_depth = 8, Padding2D_2_height = 14, Padding2D_2_width = 14;
float Padding2D_2_array[8][16][16];

uint16_t Conv2D_2_depth = 16, Conv2D_2_height = 14, Conv2D_2_width = 14;
float Conv2D_2_array[16][14][14];

uint16_t UpSampling2D_1_depth = 16, UpSampling2D_1_height = 28, UpSampling2D_1_width = 28;
float UpSampling2D_1_array[16][28][28];

uint16_t Padding2D_3_depth = 16, Padding2D_3_height = 28, Padding2D_3_width = 28;
float Padding2D_3_array[16][30][30];

uint16_t Conv2D_3_depth = 1, Conv2D_3_height = 28, Conv2D_3_width = 28;
float Conv2D_3_array[1][28][28];

}
