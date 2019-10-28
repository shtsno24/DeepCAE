/*
 * author : shtsno24
 * Date : 2019-10-28 17:41:47.335093
 *
 */
#include <cstdint>
#include <vector>

#include "./../test_data/test_data.h"
#include "./../layers_cpp/array_printf_float32.h"
#include "./../arrays_cpp/arrays_float32.h"
#include "./../layers_cpp/layers.h"
#include "./../weights_cpp/weights_float32.h"

using namespace std;

int network(float input_data[1*28*28], float output_data[1*28*28]){
	uint16_t input_0_depth = 1, input_0_height = 28, input_0_width = 28;
	vector< vector< vector< float> > > input_0_array(input_0_depth, vector< vector < float> >(input_0_height, vector< float>(input_0_width)));

	for(int depth = 0; depth < input_0_depth; depth++){
		for(int height = 0; height < input_0_height; height++){
			for(int width = 0; width < input_0_width; width++){
				input_0_array[depth][height][width] = input_data[depth * input_0_height * input_0_width + height * input_0_width + width];
			}
		}
	}
	ofstream fp("template_input_float32.tsv");
	array_fprintf_2D_float32(input_0_height, input_0_width, input_0_array[0], '\t', fp);
	fp.close();

	padding2d_float32(1, 1,
	input_0_depth, input_0_height, input_0_width, (float*) input_0_array,
	Padding2D_0_height, Padding2D_0_width, (float*) Padding2D_0_array);

	conv2d_float32(Padding2D_0_depth, Padding2D_0_height, Padding2D_0_width, (float*) Padding2D_0_array,
	Conv2D_0_depth, Conv2D_0_height, Conv2D_0_width, (float*) Conv2D_0_array,
	(float*) Conv2D_0_b,
	3, 3, (float*) Conv2D_0_w, 1);

	max_pooling2d_float32(2,
	Conv2D_0_depth, Conv2D_0_height, Conv2D_0_width, (float*) Conv2D_0_array,
	MaxPooling2D_0_depth, MaxPooling2D_0_height, MaxPooling2D_0_width, (float*) MaxPooling2D_0_array);

	padding2d_float32(1, 1,
	MaxPooling2D_0_depth, MaxPooling2D_0_height, MaxPooling2D_0_width, (float*) MaxPooling2D_0_array,
	Padding2D_1_height, Padding2D_1_width, (float*) Padding2D_1_array);

	conv2d_float32(Padding2D_1_depth, Padding2D_1_height, Padding2D_1_width, (float*) Padding2D_1_array,
	Conv2D_1_depth, Conv2D_1_height, Conv2D_1_width, (float*) Conv2D_1_array,
	(float*) Conv2D_1_b,
	3, 3, (float*) Conv2D_1_w, 1);

	max_pooling2d_float32(2,
	Conv2D_1_depth, Conv2D_1_height, Conv2D_1_width, (float*) Conv2D_1_array,
	MaxPooling2D_1_depth, MaxPooling2D_1_height, MaxPooling2D_1_width, (float*) MaxPooling2D_1_array);

	padding2d_float32(1, 1,
	MaxPooling2D_1_depth, MaxPooling2D_1_height, MaxPooling2D_1_width, (float*) MaxPooling2D_1_array,
	Padding2D_2_height, Padding2D_2_width, (float*) Padding2D_2_array);

	conv2d_float32(Padding2D_2_depth, Padding2D_2_height, Padding2D_2_width, (float*) Padding2D_2_array,
	Conv2D_2_depth, Conv2D_2_height, Conv2D_2_width, (float*) Conv2D_2_array,
	(float*) Conv2D_2_b,
	3, 3, (float*) Conv2D_2_w, 1);

	up_sampling2d_float32(2,
	Conv2D_2_depth, Conv2D_2_height, Conv2D_2_width, (float*) Conv2D_2_array,
	UpSampling2D_0_depth, UpSampling2D_0_height, UpSampling2D_0_width, (float*) UpSampling2D_0_array);

	padding2d_float32(1, 1,
	UpSampling2D_0_depth, UpSampling2D_0_height, UpSampling2D_0_width, (float*) UpSampling2D_0_array,
	Padding2D_3_height, Padding2D_3_width, (float*) Padding2D_3_array);

	conv2d_float32(Padding2D_3_depth, Padding2D_3_height, Padding2D_3_width, (float*) Padding2D_3_array,
	Conv2D_3_depth, Conv2D_3_height, Conv2D_3_width, (float*) Conv2D_3_array,
	(float*) Conv2D_3_b,
	3, 3, (float*) Conv2D_3_w, 1);

	up_sampling2d_float32(2,
	Conv2D_3_depth, Conv2D_3_height, Conv2D_3_width, (float*) Conv2D_3_array,
	UpSampling2D_1_depth, UpSampling2D_1_height, UpSampling2D_1_width, (float*) UpSampling2D_1_array);

	padding2d_float32(1, 1,
	UpSampling2D_1_depth, UpSampling2D_1_height, UpSampling2D_1_width, (float*) UpSampling2D_1_array,
	Padding2D_4_height, Padding2D_4_width, (float*) Padding2D_4_array);

	conv2d_float32(Padding2D_4_depth, Padding2D_4_height, Padding2D_4_width, (float*) Padding2D_4_array,
	Conv2D_4_depth, Conv2D_4_height, Conv2D_4_width, (float*) Conv2D_4_array,
	(float*) Conv2D_4_b,
	3, 3, (float*) Conv2D_4_w, 1);

	fp.open("template_output_float32.tsv");
	array_fprintf_2D_float32(Conv2D_4_height, Conv2D_4_width, Conv2D_4_array[0], '\t', fp);
	fp.close();

	return(0);
}
