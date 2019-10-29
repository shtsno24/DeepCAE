/*
 * author : shtsno24
 * Date : 2019-10-16 15:53:43.025819
 *
 */

#include <stdint.h>
#include <stdio.h>

#include "./../layers_c/layers.h"
#include "./../test_data/test_data.h"
#include "./../arrays_c/arrays_float32.h"
#include "./../weights_c/weights_float32.h"

#include "./../layers_cpp/array_printf_float32.h"

using namespace std;

int network(float input_data[1*28*28], float output_data[1*28*28]){
	uint16_t input_0_depth = 1, input_0_height = 28, input_0_width = 28;
	float input_0_array[1][28][28];
	vector< vector< vector< float> > > input_img(input_0_depth, vector< vector < float> >(input_0_height, vector< float>(input_0_width)));
	vector< vector< vector< float> > > output_img(input_0_depth, vector< vector < float> >(input_0_height, vector< float>(input_0_width)));

	for(int depth = 0; depth < input_0_depth; depth++){
		for(int height = 0; height < input_0_height; height++){
			for(int width = 0; width < input_0_width; width++){
				input_0_array[depth][height][width] = input_data[depth * 28 * 28 + height * 28 + width];
				input_img[depth][height][width] = input_data[depth * 28 * 28 + height * 28 + width];
			}
		}
	}

	ofstream fp("template_input_float32.tsv");
	array_fprintf_2D_float32(input_0_height, input_0_width, input_img[0], '\t', fp);
	fp.close();

	padding2d_float32(1, 1,
	input_0_depth, input_0_height, input_0_width, (float*) input_0_array,
	Padding2D_0_height, Padding2D_0_width, (float*) Padding2D_0_array);

	separable_conv2d_float32(Padding2D_0_depth, Padding2D_0_height, Padding2D_0_width, (float*) Padding2D_0_array,
	SeparableConv2D_0_depth, SeparableConv2D_0_height, SeparableConv2D_0_width, (float*) SeparableConv2D_0_array,
	(float*) SeparableConv2D_0_b_d, (float*) SeparableConv2D_0_b_p,
	3, 3, (float*) SeparableConv2D_0_w_d, (float*) SeparableConv2D_0_w_p, 1);

	max_pooling2d_float32(2,
	SeparableConv2D_0_depth, SeparableConv2D_0_height, SeparableConv2D_0_width, (float*) SeparableConv2D_0_array,
	MaxPooling2D_0_depth, MaxPooling2D_0_height, MaxPooling2D_0_width, (float*) MaxPooling2D_0_array);

	padding2d_float32(1, 1,
	MaxPooling2D_0_depth, MaxPooling2D_0_height, MaxPooling2D_0_width, (float*) MaxPooling2D_0_array,
	Padding2D_1_height, Padding2D_1_width, (float*) Padding2D_1_array);

	separable_conv2d_float32(Padding2D_1_depth, Padding2D_1_height, Padding2D_1_width, (float*) Padding2D_1_array,
	SeparableConv2D_1_depth, SeparableConv2D_1_height, SeparableConv2D_1_width, (float*) SeparableConv2D_1_array,
	(float*) SeparableConv2D_1_b_d, (float*) SeparableConv2D_1_b_p,
	3, 3, (float*) SeparableConv2D_1_w_d, (float*) SeparableConv2D_1_w_p, 1);

	max_pooling2d_float32(2,
	SeparableConv2D_1_depth, SeparableConv2D_1_height, SeparableConv2D_1_width, (float*) SeparableConv2D_1_array,
	MaxPooling2D_1_depth, MaxPooling2D_1_height, MaxPooling2D_1_width, (float*) MaxPooling2D_1_array);

	padding2d_float32(1, 1,
	MaxPooling2D_1_depth, MaxPooling2D_1_height, MaxPooling2D_1_width, (float*) MaxPooling2D_1_array,
	Padding2D_2_height, Padding2D_2_width, (float*) Padding2D_2_array);

	separable_conv2d_float32(Padding2D_2_depth, Padding2D_2_height, Padding2D_2_width, (float*) Padding2D_2_array,
	SeparableConv2D_2_depth, SeparableConv2D_2_height, SeparableConv2D_2_width, (float*) SeparableConv2D_2_array,
	(float*) SeparableConv2D_2_b_d, (float*) SeparableConv2D_2_b_p,
	3, 3, (float*) SeparableConv2D_2_w_d, (float*) SeparableConv2D_2_w_p, 1);

	up_sampling2d_float32(2,
	SeparableConv2D_2_depth, SeparableConv2D_2_height, SeparableConv2D_2_width, (float*) SeparableConv2D_2_array,
	UpSampling2D_0_depth, UpSampling2D_0_height, UpSampling2D_0_width, (float*) UpSampling2D_0_array);

	padding2d_float32(1, 1,
	UpSampling2D_0_depth, UpSampling2D_0_height, UpSampling2D_0_width, (float*) UpSampling2D_0_array,
	Padding2D_3_height, Padding2D_3_width, (float*) Padding2D_3_array);

	separable_conv2d_float32(Padding2D_3_depth, Padding2D_3_height, Padding2D_3_width, (float*) Padding2D_3_array,
	SeparableConv2D_3_depth, SeparableConv2D_3_height, SeparableConv2D_3_width, (float*) SeparableConv2D_3_array,
	(float*) SeparableConv2D_3_b_d, (float*) SeparableConv2D_3_b_p,
	3, 3, (float*) SeparableConv2D_3_w_d, (float*) SeparableConv2D_3_w_p, 1);

	up_sampling2d_float32(2,
	SeparableConv2D_3_depth, SeparableConv2D_3_height, SeparableConv2D_3_width, (float*) SeparableConv2D_3_array,
	UpSampling2D_1_depth, UpSampling2D_1_height, UpSampling2D_1_width, (float*) UpSampling2D_1_array);

	padding2d_float32(1, 1,
	UpSampling2D_1_depth, UpSampling2D_1_height, UpSampling2D_1_width, (float*) UpSampling2D_1_array,
	Padding2D_4_height, Padding2D_4_width, (float*) Padding2D_4_array);

	separable_conv2d_float32(Padding2D_4_depth, Padding2D_4_height, Padding2D_4_width, (float*) Padding2D_4_array,
	SeparableConv2D_4_depth, SeparableConv2D_4_height, SeparableConv2D_4_width, (float*) SeparableConv2D_4_array,
	(float*) SeparableConv2D_4_b_d, (float*) SeparableConv2D_4_b_p,
	3, 3, (float*) SeparableConv2D_4_w_d, (float*) SeparableConv2D_4_w_p, 1);

	for(int depth = 0; depth < input_0_depth; depth++){
		for(int height = 0; height < input_0_height; height++){
			for(int width = 0; width < input_0_width; width++){
				output_img[depth][height][width] = SeparableConv2D_4_array[depth][height][width];
			}
		}
	}

	fp.open("template_output_float32_cpp.tsv");
	array_fprintf_2D_float32(SeparableConv2D_4_height, SeparableConv2D_4_width, output_img[0], '\t', fp);
	fp.close();

	return(0);
}

int main(void){
	float output_buffer[1*28*28];
	float input_buffer[1*28*28];

	for(int depth = 0; depth < 1; depth++){
		for(int height = 0; height < 28; height++){
			for(int width = 0; width < 28; width++){
				input_buffer[depth * 28 * 28 + height * 28 + width] = test_input_float32[depth][height][width];
			}
		}
	}

	network(input_buffer, output_buffer);
}