/*
 * author : shtsno24
 * Date : 2019-10-18 23:26:40.302012
 *
 */
#include <cstdint>
#include <vector>

#include "hls_stream.h"
#include "ap_int.h"
#include "ap_axi_sdata.h"


#include "./arrays_cpp/arrays_fix16.h"
#include "./layers_cpp/layers.h"
#include "./weights_cpp/weights_fix16.h"
#include "mnist_fix16_AXI_Stream.h"

using namespace std;


void network(axis &input_data, axis &output_data){
	#pragma HLS INTERFACE s_axilite register port=return
	#pragma HLS INTERFACE axis register both port=input_data
	#pragma HLS INTERFACE axis register both port=output_data

	uint16_t input_0_depth = 1, input_0_height = 28, input_0_width = 28;
	vector< vector< vector< int16_t> > > input_0_array(input_0_depth, vector< vector < int16_t> >(input_0_height, vector< int16_t>(input_0_width)));

	ap_axis<32, 1, 1, 1> tmp;
	ap_axis<32, 1, 1, 1> out;

	do {
		input_data >> tmp;
	} while(tmp.user == 0);

	for(int depth = 0; depth < input_0_depth; depth++){
		for(int height = 0; height < input_0_height; height++){
			for(int width = 0; width < input_0_width; width++){
				tmp = input_data.read();
				input_0_array[depth][height][width] = (int16_t)tmp.data;
			}
		}
	}

	padding2d_fix16(1, 1,
	input_0_depth ,input_0_height ,input_0_width ,input_0_array,
	Padding2D_0_height ,Padding2D_0_width ,Padding2D_0_array);

	conv2d_fix16(Padding2D_0_depth ,Padding2D_0_height ,Padding2D_0_width ,Padding2D_0_array,
	Conv2D_0_depth ,Conv2D_0_height ,Conv2D_0_width ,Conv2D_0_array,
	Conv2D_0_b,
	3, 3, Conv2D_0_w, 1, fractal_width_Conv2D_0);

	max_pooling2d_fix16(2,
	Conv2D_0_depth ,Conv2D_0_height ,Conv2D_0_width ,Conv2D_0_array,
	MaxPooling2D_0_depth ,MaxPooling2D_0_height ,MaxPooling2D_0_width ,MaxPooling2D_0_array);

	padding2d_fix16(1, 1,
	MaxPooling2D_0_depth ,MaxPooling2D_0_height ,MaxPooling2D_0_width ,MaxPooling2D_0_array,
	Padding2D_1_height ,Padding2D_1_width ,Padding2D_1_array);

	conv2d_fix16(Padding2D_1_depth ,Padding2D_1_height ,Padding2D_1_width ,Padding2D_1_array,
	Conv2D_1_depth ,Conv2D_1_height ,Conv2D_1_width ,Conv2D_1_array,
	Conv2D_1_b,
	3, 3, Conv2D_1_w, 1, fractal_width_Conv2D_1);

	max_pooling2d_fix16(2,
	Conv2D_1_depth ,Conv2D_1_height ,Conv2D_1_width ,Conv2D_1_array,
	MaxPooling2D_1_depth ,MaxPooling2D_1_height ,MaxPooling2D_1_width ,MaxPooling2D_1_array);

	padding2d_fix16(1, 1,
	MaxPooling2D_1_depth ,MaxPooling2D_1_height ,MaxPooling2D_1_width ,MaxPooling2D_1_array,
	Padding2D_2_height ,Padding2D_2_width ,Padding2D_2_array);

	conv2d_fix16(Padding2D_2_depth ,Padding2D_2_height ,Padding2D_2_width ,Padding2D_2_array,
	Conv2D_2_depth ,Conv2D_2_height ,Conv2D_2_width ,Conv2D_2_array,
	Conv2D_2_b,
	3, 3, Conv2D_2_w, 1, fractal_width_Conv2D_2);

	up_sampling2d_fix16(2,
	Conv2D_2_depth ,Conv2D_2_height ,Conv2D_2_width ,Conv2D_2_array,
	UpSampling2D_0_depth ,UpSampling2D_0_height ,UpSampling2D_0_width ,UpSampling2D_0_array);

	padding2d_fix16(1, 1,
	UpSampling2D_0_depth ,UpSampling2D_0_height ,UpSampling2D_0_width ,UpSampling2D_0_array,
	Padding2D_3_height ,Padding2D_3_width ,Padding2D_3_array);

	conv2d_fix16(Padding2D_3_depth ,Padding2D_3_height ,Padding2D_3_width ,Padding2D_3_array,
	Conv2D_3_depth ,Conv2D_3_height ,Conv2D_3_width ,Conv2D_3_array,
	Conv2D_3_b,
	3, 3, Conv2D_3_w, 1, fractal_width_Conv2D_3);

	up_sampling2d_fix16(2,
	Conv2D_3_depth ,Conv2D_3_height ,Conv2D_3_width ,Conv2D_3_array,
	UpSampling2D_1_depth ,UpSampling2D_1_height ,UpSampling2D_1_width ,UpSampling2D_1_array);

	padding2d_fix16(1, 1,
	UpSampling2D_1_depth ,UpSampling2D_1_height ,UpSampling2D_1_width ,UpSampling2D_1_array,
	Padding2D_4_height ,Padding2D_4_width ,Padding2D_4_array);

	conv2d_fix16(Padding2D_4_depth ,Padding2D_4_height ,Padding2D_4_width ,Padding2D_4_array,
	Conv2D_4_depth ,Conv2D_4_height ,Conv2D_4_width ,Conv2D_4_array,
	Conv2D_4_b,
	3, 3, Conv2D_4_w, 1, fractal_width_Conv2D_4);

	for(int depth = 0; depth < Conv2D_4_depth; depth++){
		for(int height = 0; height < Conv2D_4_height; height++){
			for(int width = 0; width < Conv2D_4_width; width++){

				out.data = Conv2D_4_array[depth][height][width];

				if(depth == 0 && height == 0 && width == 0){
					out.user = 1;
				} else {
					out.user = 0;
				}

				if(depth == Conv2D_4_depth -1 && height == Conv2D_4_height -1 && width == Conv2D_4_width-1){
					out.last = 1;
				} else{
					out.last = 0;
				}
				output_data.write(out);

			}
		}
	}

	return;
}
