/*
 * author : shtsno24
 * Date : 2019-11-25 16:38:11.278572
 * Language : cpp
 * Precision : float32
 *
 */
#include <stdint.h>
#include <stdio.h>

#include "./../test_data/test_data.h"
#include "./../layers_cpp/array_printf_float32.h"
#include "./../arrays_cpp/arrays_float32.h"
#include "./../layers_cpp/layers.h"
#include "./../weights_cpp/weights_float32.h"

int network(float* input_data, float* output_data){
	for(int i = 0; i < input_0_depth * input_0_height * input_0_width; i++){
		MemBank_A[i] = input_data[i];
	}
	padding2d_float32(1, 1,
	input_0_depth, input_0_height, input_0_width, MemBank_A,
	Padding2D_0_height, Padding2D_0_width, MemBank_B);

	depthwise_conv2d_float32(Padding2D_0_depth, Padding2D_0_height, Padding2D_0_width, MemBank_B,
	SeparableConv2D_0_depth, SeparableConv2D_0_height, SeparableConv2D_0_width, MemBank_A,
	SeparableConv2D_0_b_d,
	3, 3, SeparableConv2D_0_w_d, 0);

	pointwise_conv2d_float32(Padding2D_0_depth, SeparableConv2D_0_height, SeparableConv2D_0_width, MemBank_A,
	SeparableConv2D_0_depth, SeparableConv2D_0_height, SeparableConv2D_0_width, MemBank_B,
	SeparableConv2D_0_b_p,
	1, 1, SeparableConv2D_0_w_p, 1);

	max_pooling2d_float32(2,
	SeparableConv2D_0_depth, SeparableConv2D_0_height, SeparableConv2D_0_width, MemBank_B,
	MaxPooling2D_0_depth, MaxPooling2D_0_height, MaxPooling2D_0_width, MemBank_A);

	padding2d_float32(1, 1,
	MaxPooling2D_0_depth, MaxPooling2D_0_height, MaxPooling2D_0_width, MemBank_A,
	Padding2D_1_height, Padding2D_1_width, MemBank_B);

	depthwise_conv2d_float32(Padding2D_1_depth, Padding2D_1_height, Padding2D_1_width, MemBank_B,
	SeparableConv2D_1_depth, SeparableConv2D_1_height, SeparableConv2D_1_width, MemBank_A,
	SeparableConv2D_1_b_d,
	3, 3, SeparableConv2D_1_w_d, 0);

	pointwise_conv2d_float32(Padding2D_1_depth, SeparableConv2D_1_height, SeparableConv2D_1_width, MemBank_A,
	SeparableConv2D_1_depth, SeparableConv2D_1_height, SeparableConv2D_1_width, MemBank_B,
	SeparableConv2D_1_b_p,
	1, 1, SeparableConv2D_1_w_p, 1);

	max_pooling2d_float32(2,
	SeparableConv2D_1_depth, SeparableConv2D_1_height, SeparableConv2D_1_width, MemBank_B,
	MaxPooling2D_1_depth, MaxPooling2D_1_height, MaxPooling2D_1_width, MemBank_A);

	padding2d_float32(1, 1,
	MaxPooling2D_1_depth, MaxPooling2D_1_height, MaxPooling2D_1_width, MemBank_A,
	Padding2D_2_height, Padding2D_2_width, MemBank_B);

	depthwise_conv2d_float32(Padding2D_2_depth, Padding2D_2_height, Padding2D_2_width, MemBank_B,
	SeparableConv2D_2_depth, SeparableConv2D_2_height, SeparableConv2D_2_width, MemBank_A,
	SeparableConv2D_2_b_d,
	3, 3, SeparableConv2D_2_w_d, 0);

	pointwise_conv2d_float32(Padding2D_2_depth, SeparableConv2D_2_height, SeparableConv2D_2_width, MemBank_A,
	SeparableConv2D_2_depth, SeparableConv2D_2_height, SeparableConv2D_2_width, MemBank_B,
	SeparableConv2D_2_b_p,
	1, 1, SeparableConv2D_2_w_p, 1);

	up_sampling2d_float32(2,
	SeparableConv2D_2_depth, SeparableConv2D_2_height, SeparableConv2D_2_width, MemBank_B,
	UpSampling2D_0_depth, UpSampling2D_0_height, UpSampling2D_0_width, MemBank_A);

	padding2d_float32(1, 1,
	UpSampling2D_0_depth, UpSampling2D_0_height, UpSampling2D_0_width, MemBank_A,
	Padding2D_3_height, Padding2D_3_width, MemBank_B);

	depthwise_conv2d_float32(Padding2D_3_depth, Padding2D_3_height, Padding2D_3_width, MemBank_B,
	SeparableConv2D_3_depth, SeparableConv2D_3_height, SeparableConv2D_3_width, MemBank_A,
	SeparableConv2D_3_b_d,
	3, 3, SeparableConv2D_3_w_d, 0);

	pointwise_conv2d_float32(Padding2D_3_depth, SeparableConv2D_3_height, SeparableConv2D_3_width, MemBank_A,
	SeparableConv2D_3_depth, SeparableConv2D_3_height, SeparableConv2D_3_width, MemBank_B,
	SeparableConv2D_3_b_p,
	1, 1, SeparableConv2D_3_w_p, 1);

	up_sampling2d_float32(2,
	SeparableConv2D_3_depth, SeparableConv2D_3_height, SeparableConv2D_3_width, MemBank_B,
	UpSampling2D_1_depth, UpSampling2D_1_height, UpSampling2D_1_width, MemBank_A);

	padding2d_float32(1, 1,
	UpSampling2D_1_depth, UpSampling2D_1_height, UpSampling2D_1_width, MemBank_A,
	Padding2D_4_height, Padding2D_4_width, MemBank_B);

	depthwise_conv2d_float32(Padding2D_4_depth, Padding2D_4_height, Padding2D_4_width, MemBank_B,
	SeparableConv2D_4_depth, SeparableConv2D_4_height, SeparableConv2D_4_width, MemBank_A,
	SeparableConv2D_4_b_d,
	3, 3, SeparableConv2D_4_w_d, 0);

	pointwise_conv2d_float32(Padding2D_4_depth, SeparableConv2D_4_height, SeparableConv2D_4_width, MemBank_A,
	SeparableConv2D_4_depth, SeparableConv2D_4_height, SeparableConv2D_4_width, MemBank_B,
	SeparableConv2D_4_b_p,
	1, 1, SeparableConv2D_4_w_p, 1);

	i = 0;
	for(int depth = 0; depth < SeparableConv2D_4_depth; depth++){
		for(int height = 0; height < SeparableConv2D_4_height; height++){
			for(int width = 0; width < SeparableConv2D_4_width; width++){
				output_data[i] = MemBank_B[depth][height][width];
				i += 1;
			}
		}
	}
}

void main(void){
	float output_buffer[1][28][28];

	network((float*)test_input_fix16, (float*)output_buffer);

	FILE* fp = fopen("template_input_float32.tsv", "w");
	array_fprintf_2D_float32(input_0_height, input_0_width, input_0_array[0], '\t', fp);
	fclose(fp);

	fp = fopen("template_output_float32.tsv", "w");
	array_fprintf_2D_float32(SeparableConv2D_4_height, SeparableConv2D_4_width, output_buffer[0], '\t', fp);
	fclose(fp);

}
