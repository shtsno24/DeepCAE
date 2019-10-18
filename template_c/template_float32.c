/*
 * author : shtsno24
 * Date : 2019-10-18 23:26:40.302012
 *
 */
#include <stdint.h>
#include <stdio.h>

#include "test_data/test_data.h"
#include "layers_c/array_printf_float32.h"
#include "arrays_c/arrays_float32.h"
#include "layers_c/layers.h"
#include "weights_c/weights_float32.h"

int main(void){
	uint16_t input_0_depth = 1, input_0_height = 28, input_0_width = 28;
	float input_0_array[1][28][28];

	FILE* fp = fopen("template_input_float32.tsv", "w");
	array_fprintf_2D_float32(input_0_height, input_0_width, input_0_array[0], '\t', fp);
	fclose(fp);

	padding2d_float32(1, 1,
	input_0_depth ,input_0_height ,input_0_width ,input_0_array,
	Padding2D_0_height ,Padding2D_0_width ,Padding2D_0_array);

	conv2d_float32(Padding2D_0_depth ,Padding2D_0_height ,Padding2D_0_width ,Padding2D_0_array,
	Conv2D_0_depth ,Conv2D_0_height ,Conv2D_0_width ,Conv2D_0_array,
	Conv2D_0_b,
	3, 3, Conv2D_0_w, 1);

	max_pooling2d_float32(2,
	Conv2D_0_depth ,Conv2D_0_height ,Conv2D_0_width ,Conv2D_0_array,
	MaxPooling2D_0_depth ,MaxPooling2D_0_height ,MaxPooling2D_0_width ,MaxPooling2D_0_array);

	padding2d_float32(1, 1,
	MaxPooling2D_0_depth ,MaxPooling2D_0_height ,MaxPooling2D_0_width ,MaxPooling2D_0_array,
	Padding2D_1_height ,Padding2D_1_width ,Padding2D_1_array);

	conv2d_float32(Padding2D_1_depth ,Padding2D_1_height ,Padding2D_1_width ,Padding2D_1_array,
	Conv2D_1_depth ,Conv2D_1_height ,Conv2D_1_width ,Conv2D_1_array,
	Conv2D_1_b,
	3, 3, Conv2D_1_w, 1);

	max_pooling2d_float32(2,
	Conv2D_1_depth ,Conv2D_1_height ,Conv2D_1_width ,Conv2D_1_array,
	MaxPooling2D_1_depth ,MaxPooling2D_1_height ,MaxPooling2D_1_width ,MaxPooling2D_1_array);

	padding2d_float32(1, 1,
	MaxPooling2D_1_depth ,MaxPooling2D_1_height ,MaxPooling2D_1_width ,MaxPooling2D_1_array,
	Padding2D_2_height ,Padding2D_2_width ,Padding2D_2_array);

	conv2d_float32(Padding2D_2_depth ,Padding2D_2_height ,Padding2D_2_width ,Padding2D_2_array,
	Conv2D_2_depth ,Conv2D_2_height ,Conv2D_2_width ,Conv2D_2_array,
	Conv2D_2_b,
	3, 3, Conv2D_2_w, 1);

	up_sampling2d_float32(2,
	Conv2D_2_depth ,Conv2D_2_height ,Conv2D_2_width ,Conv2D_2_array,
	UpSampling2D_0_depth ,UpSampling2D_0_height ,UpSampling2D_0_width ,UpSampling2D_0_array);

	padding2d_float32(1, 1,
	UpSampling2D_0_depth ,UpSampling2D_0_height ,UpSampling2D_0_width ,UpSampling2D_0_array,
	Padding2D_3_height ,Padding2D_3_width ,Padding2D_3_array);

	conv2d_float32(Padding2D_3_depth ,Padding2D_3_height ,Padding2D_3_width ,Padding2D_3_array,
	Conv2D_3_depth ,Conv2D_3_height ,Conv2D_3_width ,Conv2D_3_array,
	Conv2D_3_b,
	3, 3, Conv2D_3_w, 1);

	up_sampling2d_float32(2,
	Conv2D_3_depth ,Conv2D_3_height ,Conv2D_3_width ,Conv2D_3_array,
	UpSampling2D_1_depth ,UpSampling2D_1_height ,UpSampling2D_1_width ,UpSampling2D_1_array);

	padding2d_float32(1, 1,
	UpSampling2D_1_depth ,UpSampling2D_1_height ,UpSampling2D_1_width ,UpSampling2D_1_array,
	Padding2D_4_height ,Padding2D_4_width ,Padding2D_4_array);

	conv2d_float32(Padding2D_4_depth ,Padding2D_4_height ,Padding2D_4_width ,Padding2D_4_array,
	Conv2D_4_depth ,Conv2D_4_height ,Conv2D_4_width ,Conv2D_4_array,
	Conv2D_4_b,
	3, 3, Conv2D_4_w, 1);

	fp = fopen("template_output_float32.tsv", "w");
	array_fprintf_2D_float32(Conv2D_4_height, Conv2D_4_width, Conv2D_4_array[0], '\t', fp);
	fclose(fp);

	return(0);
}
