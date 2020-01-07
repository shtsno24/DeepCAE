/*
 * author : shtsno24
 * Date : 2019-11-26 09:31:49.625580
 * Language : c
 * Precision : float32
 *
 */
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>

#include "./../test_data/test_data.h"
#include "./../layers_c/array_printf_float32.h"
#include "./../arrays_c/arrays_float32.h"
#include "./../layers_c/layers.h"
#include "./../weights_c/weights_float32.h"

double gettimeofday_sec(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int network(float* input_data, float* output_data){

	float MemBank_A[max_array_size], MemBank_B[max_array_size];
	for(int i = 0; i < input_0_depth * input_0_height * input_0_width; i++){
		MemBank_A[i] = input_data[i];
	}
	padding2d_float32(1, 1,
	input_0_depth, input_0_height, input_0_width, (float*)MemBank_A,
	Padding2D_0_height, Padding2D_0_width, (float*)MemBank_B);

	depthwise_conv2d_float32(Padding2D_0_depth, Padding2D_0_height, Padding2D_0_width, (float*)MemBank_B,
	Padding2D_0_depth, SeparableConv2D_0_height, SeparableConv2D_0_width, (float*)MemBank_A,
	(float*) SeparableConv2D_0_b_d,
	3, 3, (float*) SeparableConv2D_0_w_d, 0);

	pointwise_conv2d_float32(Padding2D_0_depth, SeparableConv2D_0_height, SeparableConv2D_0_width, (float*)MemBank_A,
	SeparableConv2D_0_depth, SeparableConv2D_0_height, SeparableConv2D_0_width, (float*)MemBank_B,
	(float*)SeparableConv2D_0_b_p,
	1, 1, (float*)SeparableConv2D_0_w_p, 1);

	max_pooling2d_float32(2,
	SeparableConv2D_0_depth, SeparableConv2D_0_height, SeparableConv2D_0_width, (float*)MemBank_B,
	MaxPooling2D_0_depth, MaxPooling2D_0_height, MaxPooling2D_0_width, (float*)MemBank_A);

	padding2d_float32(1, 1,
	MaxPooling2D_0_depth, MaxPooling2D_0_height, MaxPooling2D_0_width, (float*)MemBank_A,
	Padding2D_1_height, Padding2D_1_width, (float*)MemBank_B);

	depthwise_conv2d_float32(Padding2D_1_depth, Padding2D_1_height, Padding2D_1_width, (float*)MemBank_B,
	Padding2D_1_depth, SeparableConv2D_1_height, SeparableConv2D_1_width, (float*)MemBank_A,
	(float*) SeparableConv2D_1_b_d,
	3, 3, (float*) SeparableConv2D_1_w_d, 0);

	pointwise_conv2d_float32(Padding2D_1_depth, SeparableConv2D_1_height, SeparableConv2D_1_width, (float*)MemBank_A,
	SeparableConv2D_1_depth, SeparableConv2D_1_height, SeparableConv2D_1_width, (float*)MemBank_B,
	(float*)SeparableConv2D_1_b_p,
	1, 1, (float*)SeparableConv2D_1_w_p, 1);

	max_pooling2d_float32(2,
	SeparableConv2D_1_depth, SeparableConv2D_1_height, SeparableConv2D_1_width, (float*)MemBank_B,
	MaxPooling2D_1_depth, MaxPooling2D_1_height, MaxPooling2D_1_width, (float*)MemBank_A);

	padding2d_float32(1, 1,
	MaxPooling2D_1_depth, MaxPooling2D_1_height, MaxPooling2D_1_width, (float*)MemBank_A,
	Padding2D_2_height, Padding2D_2_width, (float*)MemBank_B);

	depthwise_conv2d_float32(Padding2D_2_depth, Padding2D_2_height, Padding2D_2_width, (float*)MemBank_B,
	Padding2D_2_depth, SeparableConv2D_2_height, SeparableConv2D_2_width, (float*)MemBank_A,
	(float*) SeparableConv2D_2_b_d,
	3, 3, (float*) SeparableConv2D_2_w_d, 0);

	pointwise_conv2d_float32(Padding2D_2_depth, SeparableConv2D_2_height, SeparableConv2D_2_width, (float*)MemBank_A,
	SeparableConv2D_2_depth, SeparableConv2D_2_height, SeparableConv2D_2_width, (float*)MemBank_B,
	(float*)SeparableConv2D_2_b_p,
	1, 1, (float*)SeparableConv2D_2_w_p, 1);

	up_sampling2d_float32(2,
	SeparableConv2D_2_depth, SeparableConv2D_2_height, SeparableConv2D_2_width, (float*)MemBank_B,
	UpSampling2D_0_depth, UpSampling2D_0_height, UpSampling2D_0_width, (float*)MemBank_A);

	padding2d_float32(1, 1,
	UpSampling2D_0_depth, UpSampling2D_0_height, UpSampling2D_0_width, (float*)MemBank_A,
	Padding2D_3_height, Padding2D_3_width, (float*)MemBank_B);

	depthwise_conv2d_float32(Padding2D_3_depth, Padding2D_3_height, Padding2D_3_width, (float*)MemBank_B,
	Padding2D_3_depth, SeparableConv2D_3_height, SeparableConv2D_3_width, (float*)MemBank_A,
	(float*) SeparableConv2D_3_b_d,
	3, 3, (float*) SeparableConv2D_3_w_d, 0);

	pointwise_conv2d_float32(Padding2D_3_depth, SeparableConv2D_3_height, SeparableConv2D_3_width, (float*)MemBank_A,
	SeparableConv2D_3_depth, SeparableConv2D_3_height, SeparableConv2D_3_width, (float*)MemBank_B,
	(float*)SeparableConv2D_3_b_p,
	1, 1, (float*)SeparableConv2D_3_w_p, 1);

	up_sampling2d_float32(2,
	SeparableConv2D_3_depth, SeparableConv2D_3_height, SeparableConv2D_3_width, (float*)MemBank_B,
	UpSampling2D_1_depth, UpSampling2D_1_height, UpSampling2D_1_width, (float*)MemBank_A);

	padding2d_float32(1, 1,
	UpSampling2D_1_depth, UpSampling2D_1_height, UpSampling2D_1_width, (float*)MemBank_A,
	Padding2D_4_height, Padding2D_4_width, (float*)MemBank_B);

	depthwise_conv2d_float32(Padding2D_4_depth, Padding2D_4_height, Padding2D_4_width, (float*)MemBank_B,
	Padding2D_4_depth, SeparableConv2D_4_height, SeparableConv2D_4_width, (float*)MemBank_A,
	(float*) SeparableConv2D_4_b_d,
	3, 3, (float*) SeparableConv2D_4_w_d, 0);

	pointwise_conv2d_float32(Padding2D_4_depth, SeparableConv2D_4_height, SeparableConv2D_4_width, (float*)MemBank_A,
	SeparableConv2D_4_depth, SeparableConv2D_4_height, SeparableConv2D_4_width, (float*)MemBank_B,
	(float*)SeparableConv2D_4_b_p,
	1, 1, (float*)SeparableConv2D_4_w_p, 1);

	for(int i = 0; i < SeparableConv2D_4_depth * SeparableConv2D_4_height * SeparableConv2D_4_width; i++){
		output_data[i] = MemBank_B[i];
	}

	return(0);

}

int main(void){
	double start, end, sum_time = 0;
	int32_t times = 5000;
	FILE* fp;

	float output_buffer[1][28][28];


	fp = fopen("time_output_float32_c_Sep.tsv", "w");
	for(int32_t i = 0; i < times; i++){

		start = gettimeofday_sec();
		network((float*)test_input_float32, (float*)output_buffer);
		end = gettimeofday_sec();

		fprintf(fp, "%lf\t\n", end - start);
		sum_time += end - start;
	}

	printf("end_time_float32_c_Sep : %lf [s]\r\n", sum_time / (double)times);
	fclose(fp);


	fp = fopen("template_input_float32_Sep.tsv", "w");
	array_fprintf_2D_float32(input_0_height, input_0_width, test_input_float32[0], '\t', fp);
	fclose(fp);

	fp = fopen("template_output_float32_c_Sep.tsv", "w");
	array_fprintf_2D_float32(SeparableConv2D_4_height, SeparableConv2D_4_width, output_buffer[0], '\t', fp);
	fclose(fp);
	return(0);

}
