#include <stdio.h>
#include <stdint.h>

#include "array_printf_fix16.h"
#include "layers/layers.h"
#include "keras_mnist_DCAE/keras_mnist_DCAE_params_fixed.h"



int main(void){
    // Input_0 layer
    uint16_t input_0_depth = 1, input_0_height = 28, input_0_width = 28;
    int16_t input_0_array[input_0_depth][input_0_height][input_0_width];
    // init input_0_array
    for(uint16_t depth=0; depth<input_0_depth; depth++){
        for(uint16_t height=0; height<input_0_height; height++){
            for(uint16_t width=0; width<input_0_width; width++){
                input_0_array[depth][height][width] = float2fixed((float)width / (float)input_0_width, fractal_width_Conv2D_0);
                // printf("%d ", float2fixed((float)width, fractal_width_Conv2D_0));
            }
            // printf("\r\n");
        }
    }

    // Conv2D_0 layer
    uint16_t Padding_0_depth = 1, Padding_0_height = 30, Padding_0_width = 30;
    int16_t Padding_0_array[Padding_0_depth][Padding_0_height][Padding_0_width];

    uint16_t Conv2d_0_depth = 16, Conv2D_0_height = 28, Conv2D_0_width = 28;
    int16_t Conv2D_0_array[Conv2d_0_depth][Conv2D_0_height][Conv2D_0_width];


    // MaxPool_0 layer
    uint16_t MaxPool_0_depth = 16, MaxPool_0_height = 14, MaxPool_0_width = 14;
    int16_t MaxPool_0_array[MaxPool_0_depth][MaxPool_0_height][MaxPool_0_width];


    // Conv2D_1 layer
    uint16_t Padding_1_depth = 16, Padding_1_height = 16, Padding_1_width = 16;
    int16_t Padding_1_array[Padding_1_depth][Padding_1_height][Padding_1_width];

    uint16_t Conv2d_1_depth = 16, Conv2D_1_height = 14, Conv2D_1_width = 14;
    int16_t Conv2D_1_array[Conv2d_1_depth][Conv2D_1_height][Conv2D_1_width];


    // MaxPool_1 layer
    uint16_t MaxPool_1_depth = 8, MaxPool_1_height = 7, MaxPool_1_width = 7;
    int16_t MaxPool_1_array[MaxPool_1_depth][MaxPool_1_height][MaxPool_1_width];


    // Upsampling_0 layer
    uint16_t UpSampling_0_depth = 8, UpSampling_0_height = 14, UpSampling_0_width = 14;
    int16_t UpSampling_0_array[UpSampling_0_depth][UpSampling_0_height][UpSampling_0_width];


    // Conv2D_2 layer
    uint16_t Padding_2_depth = 8, Padding_2_height = 16, Padding_2_width = 16;
    int16_t Padding_2_array[Padding_2_depth][Padding_2_height][Padding_2_width];

    uint16_t Conv2d_2_depth = 16, Conv2D_2_height = 14, Conv2D_2_width = 14;
    int16_t Conv2D_2_array[Conv2d_2_depth][Conv2D_2_height][Conv2D_2_width];


    // Upsampling_0 layer
    uint16_t UpSampling_1_depth = 16, UpSampling_1_height = 28, UpSampling_1_width = 28;
    int16_t UpSampling_1_array[UpSampling_1_depth][UpSampling_1_height][UpSampling_1_width];


    // Conv2D_3 layer
    uint16_t Padding_3_depth = 16, Padding_3_height = 30, Padding_3_width = 30;
    int16_t Padding_3_array[Padding_3_depth][Padding_3_height][Padding_3_width];

    uint16_t Conv2d_3_depth = 1, Conv2D_3_height = 28, Conv2D_3_width = 28;
    int16_t Conv2D_3_array[Conv2d_3_depth][Conv2D_3_height][Conv2D_3_width];


    // Output_0 layer
    uint16_t output_0_depth = 1, output_0_height = 28, output_0_width = 28;
    int16_t output_0_array[output_0_depth][output_0_height][output_0_width];


    // array_printf_3D_int16(input_0_depth, input_0_height, input_0_width, input_0_array);

    FILE* fp = fopen("DCAE_input_fix16.tsv", "w");
    array_fprintf_2D_fix16(input_0_height, input_0_width, input_0_array[0], '\t', fp, fractal_width_Conv2D_0);
    fclose(fp);

    padding2d_fix16(1, 1, 
    input_0_depth, input_0_height, input_0_width, input_0_array,
    Padding_0_height, Padding_0_width, Padding_0_array);

    // array_printf_3D_int16_t32(Padding_0_depth, Padding_0_height, Padding_0_width, Padding_0_array);

    conv2d_fix16(Padding_0_depth, Padding_0_height, Padding_0_width, Padding_0_array,
    Conv2d_0_depth, Conv2D_0_height, Conv2D_0_width, Conv2D_0_array,
    Conv2D_0_b,
    3, 3, Conv2D_0_w, 1, fractal_width_Conv2D_0);

    // array_printf_3D_int16(Conv2d_0_depth, Conv2D_0_height, Conv2D_0_width, Conv2D_0_array);

    max_pooling2d_fix16(2,
    Conv2d_0_depth, Conv2D_0_height, Conv2D_0_width, Conv2D_0_array,
    MaxPool_0_depth, MaxPool_0_height, MaxPool_0_width, MaxPool_0_array);

    // array_printf_3D_int16(MaxPool_0_depth, MaxPool_0_height, MaxPool_0_width, MaxPool_0_array);

    padding2d_fix16(1, 1,
    MaxPool_0_depth, MaxPool_0_height, MaxPool_0_width, MaxPool_0_array,
    Padding_1_height, Padding_1_width, Padding_1_array);

    // array_printf_3D_int16(Padding_1_depth, Padding_1_height, Padding_1_width, Padding_1_array);

    conv2d_fix16(Padding_1_depth, Padding_1_height, Padding_1_width, Padding_1_array,
    Conv2d_1_depth, Conv2D_1_height, Conv2D_1_width, Conv2D_1_array,
    Conv2D_1_b,
    3, 3, Conv2D_1_w, 1, fractal_width_Conv2D_1);

    // array_printf_3D_int16(Conv2d_1_depth, Conv2D_1_height, Conv2D_1_width, Conv2D_1_array);

    max_pooling2d_fix16(2,
    Conv2d_1_depth, Conv2D_1_height, Conv2D_1_width, Conv2D_1_array,
    MaxPool_1_depth, MaxPool_1_height, MaxPool_1_width, MaxPool_1_array);

    // array_printf_3D_int16(MaxPool_1_depth, MaxPool_1_height, MaxPool_1_width, MaxPool_1_array);

    up_sampling2d_fix16(2,
    MaxPool_1_depth, MaxPool_1_height, MaxPool_1_width, MaxPool_1_array,
    UpSampling_0_depth, UpSampling_0_height, UpSampling_0_width, UpSampling_0_array);

    // array_printf_3D_int16(MaxPool_1_depth, MaxPool_1_height, MaxPool_1_width, MaxPool_1_array);

    padding2d_fix16(1, 1,
    UpSampling_0_depth, UpSampling_0_height, UpSampling_0_width, UpSampling_0_array,
    Padding_2_height, Padding_2_width, Padding_2_array);

    // array_printf_3D_int16(Padding_2_depth, Padding_2_height, Padding_2_width, Padding_2_array);

    conv2d_fix16(Padding_2_depth, Padding_2_height, Padding_2_width, Padding_2_array,
    Conv2d_2_depth, Conv2D_2_height, Conv2D_2_width, Conv2D_2_array,
    Conv2D_2_b,
    3, 3, Conv2D_2_w, 1, fractal_width_Conv2D_2);

    // array_printf_3D_int16(Conv2d_2_depth, Conv2D_2_height, Conv2D_2_width, Conv2D_2_array);

    up_sampling2d_fix16(2,
    Conv2d_2_depth, Conv2D_2_height, Conv2D_2_width, Conv2D_2_array,
    UpSampling_1_depth, UpSampling_1_height, UpSampling_1_width, UpSampling_1_array);

    // array_printf_3D_int16(UpSampling_1_depth, UpSampling_1_height, UpSampling_1_width, UpSampling_1_array);

    padding2d_fix16(1, 1,
    UpSampling_1_depth, UpSampling_1_height, UpSampling_1_width, UpSampling_1_array,
    Padding_3_height, Padding_3_width, Padding_3_array);

    // array_printf_3D_int16(Padding_3_depth, Padding_3_height, Padding_3_width, Padding_3_array);

    conv2d_fix16(Padding_3_depth, Padding_3_height, Padding_3_width, Padding_3_array,
    Conv2d_3_depth, Conv2D_3_height, Conv2D_3_width, Conv2D_3_array,
    Conv2D_3_b,
    3, 3, Conv2D_3_w, 1, fractal_width_Conv2D_3);

    // array_printf_3D_int16(Conv2d_3_depth, Conv2D_3_height, Conv2D_3_width, Conv2D_3_array);

    fp = fopen("DCAE_output_fix16.tsv", "w");
    array_fprintf_2D_fix16(Conv2D_3_height, Conv2D_3_width, Conv2D_3_array[0], '\t', fp, fractal_width_Conv2D_3);
    fclose(fp);

}