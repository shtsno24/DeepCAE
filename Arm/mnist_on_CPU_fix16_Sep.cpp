/*
 * author : shtsno24
 * Date : 2019-10-16 15:53:43.025819
 *
 */

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <sys/time.h>

#include "./../layers_c/layers.h"
#include "./../test_data/test_data.h"
#include "./../arrays_c/arrays_fix16.h"
#include "./../weights_c/weights_fix16.h"

#include "./../layers_cpp/array_printf_fix16.h"

using namespace std;

double gettimeofday_sec(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int network(int16_t* input_data, int16_t* output_data){
	int16_t MemBank_A[max_array_size], MemBank_B[max_array_size];
	for(int i = 0; i < input_0_depth * input_0_height * input_0_width; i++){
		MemBank_A[i] = input_data[i];
	}
	padding2d_fix16(1, 1,
	input_0_depth, input_0_height, input_0_width, (int16_t*)MemBank_A,
	Padding2D_0_height, Padding2D_0_width, (int16_t*)MemBank_B);

	depthwise_conv2d_fix16(Padding2D_0_depth, Padding2D_0_height, Padding2D_0_width, (int16_t*)MemBank_B,
	Padding2D_0_depth, SeparableConv2D_0_height, SeparableConv2D_0_width, (int16_t*)MemBank_A,
	(int16_t*) SeparableConv2D_0_b_d,
	3, 3, (int16_t*) SeparableConv2D_0_w_d, 0, fractal_width_SeparableConv2D_0);

	pointwise_conv2d_fix16(Padding2D_0_depth, SeparableConv2D_0_height, SeparableConv2D_0_width, (int16_t*)MemBank_A,
	SeparableConv2D_0_depth, SeparableConv2D_0_height, SeparableConv2D_0_width, (int16_t*)MemBank_B,
	(int16_t*)SeparableConv2D_0_b_p,
	1, 1, (int16_t*)SeparableConv2D_0_w_p, 1, fractal_width_SeparableConv2D_0);

	max_pooling2d_fix16(2,
	SeparableConv2D_0_depth, SeparableConv2D_0_height, SeparableConv2D_0_width, (int16_t*)MemBank_B,
	MaxPooling2D_0_depth, MaxPooling2D_0_height, MaxPooling2D_0_width, (int16_t*)MemBank_A);

	padding2d_fix16(1, 1,
	MaxPooling2D_0_depth, MaxPooling2D_0_height, MaxPooling2D_0_width, (int16_t*)MemBank_A,
	Padding2D_1_height, Padding2D_1_width, (int16_t*)MemBank_B);

	depthwise_conv2d_fix16(Padding2D_1_depth, Padding2D_1_height, Padding2D_1_width, (int16_t*)MemBank_B,
	Padding2D_1_depth, SeparableConv2D_1_height, SeparableConv2D_1_width, (int16_t*)MemBank_A,
	(int16_t*) SeparableConv2D_1_b_d,
	3, 3, (int16_t*) SeparableConv2D_1_w_d, 0, fractal_width_SeparableConv2D_1);

	pointwise_conv2d_fix16(Padding2D_1_depth, SeparableConv2D_1_height, SeparableConv2D_1_width, (int16_t*)MemBank_A,
	SeparableConv2D_1_depth, SeparableConv2D_1_height, SeparableConv2D_1_width, (int16_t*)MemBank_B,
	(int16_t*)SeparableConv2D_1_b_p,
	1, 1, (int16_t*)SeparableConv2D_1_w_p, 1, fractal_width_SeparableConv2D_1);

	max_pooling2d_fix16(2,
	SeparableConv2D_1_depth, SeparableConv2D_1_height, SeparableConv2D_1_width, (int16_t*)MemBank_B,
	MaxPooling2D_1_depth, MaxPooling2D_1_height, MaxPooling2D_1_width, (int16_t*)MemBank_A);

	padding2d_fix16(1, 1,
	MaxPooling2D_1_depth, MaxPooling2D_1_height, MaxPooling2D_1_width, (int16_t*)MemBank_A,
	Padding2D_2_height, Padding2D_2_width, (int16_t*)MemBank_B);

	depthwise_conv2d_fix16(Padding2D_2_depth, Padding2D_2_height, Padding2D_2_width, (int16_t*)MemBank_B,
	Padding2D_2_depth, SeparableConv2D_2_height, SeparableConv2D_2_width, (int16_t*)MemBank_A,
	(int16_t*) SeparableConv2D_2_b_d,
	3, 3, (int16_t*) SeparableConv2D_2_w_d, 0, fractal_width_SeparableConv2D_2);

	pointwise_conv2d_fix16(Padding2D_2_depth, SeparableConv2D_2_height, SeparableConv2D_2_width, (int16_t*)MemBank_A,
	SeparableConv2D_2_depth, SeparableConv2D_2_height, SeparableConv2D_2_width, (int16_t*)MemBank_B,
	(int16_t*)SeparableConv2D_2_b_p,
	1, 1, (int16_t*)SeparableConv2D_2_w_p, 1, fractal_width_SeparableConv2D_2);

	up_sampling2d_fix16(2,
	SeparableConv2D_2_depth, SeparableConv2D_2_height, SeparableConv2D_2_width, (int16_t*)MemBank_B,
	UpSampling2D_0_depth, UpSampling2D_0_height, UpSampling2D_0_width, (int16_t*)MemBank_A);

	padding2d_fix16(1, 1,
	UpSampling2D_0_depth, UpSampling2D_0_height, UpSampling2D_0_width, (int16_t*)MemBank_A,
	Padding2D_3_height, Padding2D_3_width, (int16_t*)MemBank_B);

	depthwise_conv2d_fix16(Padding2D_3_depth, Padding2D_3_height, Padding2D_3_width, (int16_t*)MemBank_B,
	Padding2D_3_depth, SeparableConv2D_3_height, SeparableConv2D_3_width, (int16_t*)MemBank_A,
	(int16_t*) SeparableConv2D_3_b_d,
	3, 3, (int16_t*) SeparableConv2D_3_w_d, 0, fractal_width_SeparableConv2D_3);

	pointwise_conv2d_fix16(Padding2D_3_depth, SeparableConv2D_3_height, SeparableConv2D_3_width, (int16_t*)MemBank_A,
	SeparableConv2D_3_depth, SeparableConv2D_3_height, SeparableConv2D_3_width, (int16_t*)MemBank_B,
	(int16_t*)SeparableConv2D_3_b_p,
	1, 1, (int16_t*)SeparableConv2D_3_w_p, 1, fractal_width_SeparableConv2D_3);

	up_sampling2d_fix16(2,
	SeparableConv2D_3_depth, SeparableConv2D_3_height, SeparableConv2D_3_width, (int16_t*)MemBank_B,
	UpSampling2D_1_depth, UpSampling2D_1_height, UpSampling2D_1_width, (int16_t*)MemBank_A);

	padding2d_fix16(1, 1,
	UpSampling2D_1_depth, UpSampling2D_1_height, UpSampling2D_1_width, (int16_t*)MemBank_A,
	Padding2D_4_height, Padding2D_4_width, (int16_t*)MemBank_B);

	depthwise_conv2d_fix16(Padding2D_4_depth, Padding2D_4_height, Padding2D_4_width, (int16_t*)MemBank_B,
	Padding2D_4_depth, SeparableConv2D_4_height, SeparableConv2D_4_width, (int16_t*)MemBank_A,
	(int16_t*) SeparableConv2D_4_b_d,
	3, 3, (int16_t*) SeparableConv2D_4_w_d, 0, fractal_width_SeparableConv2D_4);

	pointwise_conv2d_fix16(Padding2D_4_depth, SeparableConv2D_4_height, SeparableConv2D_4_width, (int16_t*)MemBank_A,
	SeparableConv2D_4_depth, SeparableConv2D_4_height, SeparableConv2D_4_width, (int16_t*)MemBank_B,
	(int16_t*)SeparableConv2D_4_b_p,
	1, 1, (int16_t*)SeparableConv2D_4_w_p, 1, fractal_width_SeparableConv2D_4);

	for(int i = 0; i < SeparableConv2D_4_depth * SeparableConv2D_4_height * SeparableConv2D_4_width; i++){
		output_data[i] = MemBank_B[i];
	}

	return(0);

}

int main(void){
	double start, end, sum_time = 0;
	int32_t times = 5000;
	int16_t output_buffer[1][28][28];
	vector< vector< vector< int16_t> > > input_img(1, vector< vector< int16_t> >(28, vector< int16_t>(28)));
	vector< vector< vector< int16_t> > > output_img(1, vector< vector< int16_t> >(28, vector< int16_t>(28)));

	for(int depth = 0; depth < input_0_depth; depth++){
		for(int height = 0; height < input_0_height; height++){
			for(int width = 0; width < input_0_width; width++){
				input_img[depth][height][width] = test_input_fix16[depth][height][width];
			}
		}
	}

	ofstream time_file("time_output_fix16_cpp_Sep.tsv");
	for(int32_t i = 0; i < times; i++){
		start = gettimeofday_sec();
		network((int16_t*)test_input_fix16, (int16_t*)output_buffer);
		end = gettimeofday_sec();
		time_file << end - start << endl;
		sum_time += end - start;
	}
	cout << "end_time_fix16_cpp_Sep : " << sum_time / (double)times << " [s]" << endl;
	time_file.close();	

	for(int depth = 0; depth < SeparableConv2D_4_depth; depth++){
		for(int height = 0; height < SeparableConv2D_4_height; height++){
			for(int width = 0; width < SeparableConv2D_4_width; width++){
				output_img[depth][height][width] = output_buffer[depth][height][width];
			}
		}
	}
	ofstream fp("template_input_fix16_Sep.tsv");
	array_fprintf_2D_fix16(input_0_height, input_0_width, input_img[0], '\t', fp, fractal_width_input_0);
	fp.close();

	fp.open("template_output_fix16_cpp_Sep.tsv");
	array_fprintf_2D_fix16(SeparableConv2D_4_height, SeparableConv2D_4_width, output_img[0], '\t', fp, fractal_width_SeparableConv2D_4);
	fp.close();
	return(0);
}