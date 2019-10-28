/*
 * author : shtsno24
 * Date : 2019-10-28 17:41:47.335093
 *
 */
#include <stdint.h>
#include <stdio.h>

#include "test_data/test_data.h"
#include "layers_c/array_printf_fix16.h"
#include "arrays_c/arrays_fix16.h"
#include "layers_c/layers.h"
#include "weights_c/weights_fix16.h"

int network(int16_t input_data[1*28*28], int16_t output_data[1*28*28]){
	uint16_t input_0_depth = 1, input_0_height = 28, input_0_width = 28;
	int16_t input_0_array[1][28][28];

	for(int depth = 0; depth < input_0_depth; depth++){
		for(int height = 0; height < input_0_height; height++){
			for(int width = 0; width < input_0_width; width++){
				input_0_array[depth][height][width] = input_data[depth * input_0_height * input_0_width + height * input_0_width + width];
			}
		}
	}
	FILE* fp = fopen("template_input_fix16.tsv", "w");
	array_fprintf_2D_fix16(input_0_height, input_0_width, input_0_array[0], '\t', fp, fractal_width_Conv2D_0);
	fclose(fp);

	padding2d_fix16(1, 1,
	input_0_depth, input_0_height, input_0_width, (int16_t*) input_0_array,
	Padding2D_0_height, Padding2D_0_width, (int16_t*) Padding2D_0_array);

	conv2d_fix16(Padding2D_0_depth, Padding2D_0_height, Padding2D_0_width, (int16_t*) Padding2D_0_array,
	Conv2D_0_depth, Conv2D_0_height, Conv2D_0_width, (int16_t*) Conv2D_0_array,
	(int16_t*) Conv2D_0_b,
	3, 3, (int16_t*) Conv2D_0_w, 1, fractal_width_Conv2D_0);

	max_pooling2d_fix16(2,
	Conv2D_0_depth, Conv2D_0_height, Conv2D_0_width, (int16_t*) Conv2D_0_array,
	MaxPooling2D_0_depth, MaxPooling2D_0_height, MaxPooling2D_0_width, (int16_t*) MaxPooling2D_0_array);

	padding2d_fix16(1, 1,
	MaxPooling2D_0_depth, MaxPooling2D_0_height, MaxPooling2D_0_width, (int16_t*) MaxPooling2D_0_array,
	Padding2D_1_height, Padding2D_1_width, (int16_t*) Padding2D_1_array);

	conv2d_fix16(Padding2D_1_depth, Padding2D_1_height, Padding2D_1_width, (int16_t*) Padding2D_1_array,
	Conv2D_1_depth, Conv2D_1_height, Conv2D_1_width, (int16_t*) Conv2D_1_array,
	(int16_t*) Conv2D_1_b,
	3, 3, (int16_t*) Conv2D_1_w, 1, fractal_width_Conv2D_1);

	max_pooling2d_fix16(2,
	Conv2D_1_depth, Conv2D_1_height, Conv2D_1_width, (int16_t*) Conv2D_1_array,
	MaxPooling2D_1_depth, MaxPooling2D_1_height, MaxPooling2D_1_width, (int16_t*) MaxPooling2D_1_array);

	padding2d_fix16(1, 1,
	MaxPooling2D_1_depth, MaxPooling2D_1_height, MaxPooling2D_1_width, (int16_t*) MaxPooling2D_1_array,
	Padding2D_2_height, Padding2D_2_width, (int16_t*) Padding2D_2_array);

	conv2d_fix16(Padding2D_2_depth, Padding2D_2_height, Padding2D_2_width, (int16_t*) Padding2D_2_array,
	Conv2D_2_depth, Conv2D_2_height, Conv2D_2_width, (int16_t*) Conv2D_2_array,
	(int16_t*) Conv2D_2_b,
	3, 3, (int16_t*) Conv2D_2_w, 1, fractal_width_Conv2D_2);

	up_sampling2d_fix16(2,
	Conv2D_2_depth, Conv2D_2_height, Conv2D_2_width, (int16_t*) Conv2D_2_array,
	UpSampling2D_0_depth, UpSampling2D_0_height, UpSampling2D_0_width, (int16_t*) UpSampling2D_0_array);

	padding2d_fix16(1, 1,
	UpSampling2D_0_depth, UpSampling2D_0_height, UpSampling2D_0_width, (int16_t*) UpSampling2D_0_array,
	Padding2D_3_height, Padding2D_3_width, (int16_t*) Padding2D_3_array);

	conv2d_fix16(Padding2D_3_depth, Padding2D_3_height, Padding2D_3_width, (int16_t*) Padding2D_3_array,
	Conv2D_3_depth, Conv2D_3_height, Conv2D_3_width, (int16_t*) Conv2D_3_array,
	(int16_t*) Conv2D_3_b,
	3, 3, (int16_t*) Conv2D_3_w, 1, fractal_width_Conv2D_3);

	up_sampling2d_fix16(2,
	Conv2D_3_depth, Conv2D_3_height, Conv2D_3_width, (int16_t*) Conv2D_3_array,
	UpSampling2D_1_depth, UpSampling2D_1_height, UpSampling2D_1_width, (int16_t*) UpSampling2D_1_array);

	padding2d_fix16(1, 1,
	UpSampling2D_1_depth, UpSampling2D_1_height, UpSampling2D_1_width, (int16_t*) UpSampling2D_1_array,
	Padding2D_4_height, Padding2D_4_width, (int16_t*) Padding2D_4_array);

	conv2d_fix16(Padding2D_4_depth, Padding2D_4_height, Padding2D_4_width, (int16_t*) Padding2D_4_array,
	Conv2D_4_depth, Conv2D_4_height, Conv2D_4_width, (int16_t*) Conv2D_4_array,
	(int16_t*) Conv2D_4_b,
	3, 3, (int16_t*) Conv2D_4_w, 1, fractal_width_Conv2D_4);

	fp = fopen("template_output_fix16.tsv", "w");
	array_fprintf_2D_fix16(Conv2D_4_height, Conv2D_4_width, Conv2D_4_array[0], '\t', fp, fractal_width_Conv2D_4);
	fclose(fp);

	return(0);
}
