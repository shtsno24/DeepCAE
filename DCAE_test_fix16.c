#include <stdio.h>
#include <stdint.h>

#include "array_printf_fix16.h"
#include "arrays/arrays_fix16.h"
#include "layers/layers.h"
#include "weights/weights_fix16.h"

int main(void){
    // Input_0 layer
    uint16_t input_0_depth = 1, input_0_height = 28, input_0_width = 28;
    int16_t input_0_array[input_0_depth][input_0_height][input_0_width];
    
    // init input_0_array
    for(uint16_t depth=0; depth<input_0_depth; depth++){
        for(uint16_t height=0; height<input_0_height; height++){
            for(uint16_t width=0; width<input_0_width; width++){
                input_0_array[depth][height][width] = float2fixed((float)width / (float)input_0_width, fractal_width_Conv2D_0);
            }
        }
    }

    // Output_0 layer
    uint16_t output_0_depth = 1, output_0_height = 28, output_0_width = 28;
    int16_t output_0_array[output_0_depth][output_0_height][output_0_width];


    FILE* fp = fopen("DCAE_input_fix16.tsv", "w");
    array_fprintf_2D_fix16(input_0_height, input_0_width, input_0_array[0], '\t', fp, fractal_width_Conv2D_0);
    fclose(fp);

    padding2d_fix16(1, 1, 
    input_0_depth, input_0_height, input_0_width, input_0_array,
    Padding2D_0_height, Padding2D_0_width, Padding2D_0_array);

    conv2d_fix16(Padding2D_0_depth, Padding2D_0_depth, Padding2D_0_height, Padding2D_0_array,
    Conv2D_0_depth, Conv2D_0_height, Conv2D_0_width, Conv2D_0_array,
    Conv2D_0_b,
    3, 3, Conv2D_0_w, 1, fractal_width_Conv2D_0);


    max_pooling2d_fix16(2,
    Conv2D_0_depth, Conv2D_0_height, Conv2D_0_width, Conv2D_0_array,
    MaxPooling2D_0_depth, MaxPooling2D_0_height, MaxPooling2D_0_width, MaxPooling2D_0_array);


    padding2d_fix16(1, 1,
    MaxPooling2D_0_depth, MaxPooling2D_0_height, MaxPooling2D_0_width, MaxPooling2D_0_array,
    Padding2D_1_height, Padding2D_1_width, Padding2D_1_array);

    conv2d_fix16(Padding2D_1_depth, Padding2D_1_height, Padding2D_1_width, Padding2D_1_array,
    Conv2D_1_depth, Conv2D_1_height, Conv2D_1_width, Conv2D_1_array,
    Conv2D_1_b,
    3, 3, Conv2D_1_w, 1, fractal_width_Conv2D_1);


    max_pooling2d_fix16(2,
    Conv2D_1_depth, Conv2D_1_height, Conv2D_1_width, Conv2D_1_array,
    MaxPooling2D_1_depth, MaxPooling2D_1_height, MaxPooling2D_1_width, MaxPooling2D_1_array);


    up_sampling2d_fix16(2,
    MaxPooling2D_1_depth, MaxPooling2D_1_height, MaxPooling2D_1_width, MaxPooling2D_1_array,
    UpSampling2D_0_depth, UpSampling2D_0_height, UpSampling2D_0_width, UpSampling2D_0_array);


    padding2d_fix16(1, 1,
    UpSampling2D_0_depth, UpSampling2D_0_height, UpSampling2D_0_width, UpSampling2D_0_array,
    Padding2D_2_height, Padding2D_2_width, Padding2D_2_array);

    conv2d_fix16(Padding2D_2_depth, Padding2D_2_height, Padding2D_2_width, Padding2D_2_array,
    Conv2D_2_depth, Conv2D_2_height, Conv2D_2_width, Conv2D_2_array,
    Conv2D_2_b,
    3, 3, Conv2D_2_w, 1, fractal_width_Conv2D_2);


    up_sampling2d_fix16(2,
    Conv2D_2_depth, Conv2D_2_height, Conv2D_2_width, Conv2D_2_array,
    UpSampling2D_1_depth, UpSampling2D_1_height, UpSampling2D_1_width, UpSampling2D_1_array);


    padding2d_fix16(1, 1,
    UpSampling2D_1_depth, UpSampling2D_1_height, UpSampling2D_1_width, UpSampling2D_1_array,
    Padding2D_3_height, Padding2D_3_width, Padding2D_3_array);

    conv2d_fix16(Padding2D_3_depth, Padding2D_3_height, Padding2D_3_width, Padding2D_3_array,
    Conv2D_3_depth, Conv2D_3_height, Conv2D_3_width, Conv2D_3_array,
    Conv2D_3_b,
    3, 3, Conv2D_3_w, 1, fractal_width_Conv2D_3);


    fp = fopen("DCAE_output_fix16.tsv", "w");
    array_fprintf_2D_fix16(Conv2D_3_height, Conv2D_3_width, Conv2D_3_array[0], '\t', fp, fractal_width_Conv2D_3);
    fclose(fp);

}