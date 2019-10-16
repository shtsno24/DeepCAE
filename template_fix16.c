/*
 * author : shtsno24
 * Date : 2019-10-16 13:17:10.770135
 *
 */
#pragma once
#include <stdint.h>
#include <stdio.h>

#include "array_printf_fix16.h"
#include "arrays/arrays_fix16.h"
#include "layers/layers.h"
#include "weights/weights_fix16.h"

int main(void){
	uint16_t input_0_depth = 1, input_0_height = 28, input_0_width = 28;
	int16_t input_0_array[1][28][28];

	padding2d_fix16(1, 1,
	input_0_depth ,input_0_height ,input_0_width ,input_0_array,
	Padding2D_0_height ,Padding2D_0_width ,Padding2D_0_array);
	padding2d_fix16(1, 1,
	MaxPooling2D_0_depth ,MaxPooling2D_0_height ,MaxPooling2D_0_width ,MaxPooling2D_0_array,
	Padding2D_1_height ,Padding2D_1_width ,Padding2D_1_array);
	padding2d_fix16(1, 1,
	UpSampling2D_0_depth ,UpSampling2D_0_height ,UpSampling2D_0_width ,UpSampling2D_0_array,
	Padding2D_2_height ,Padding2D_2_width ,Padding2D_2_array);
	padding2d_fix16(1, 1,
	UpSampling2D_1_depth ,UpSampling2D_1_height ,UpSampling2D_1_width ,UpSampling2D_1_array,
	Padding2D_3_height ,Padding2D_3_width ,Padding2D_3_array);
	return(0);
}
