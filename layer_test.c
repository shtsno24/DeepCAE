#include <stdio.h>
#include <stdint.h>

#include "array_printf.h"
#include "layers/padding2d.h"
#include "layers/up_sampling2d.h"
#include "layers/max_pooling2d.h"
#include "layers/conv2d.h"

#define array_size_h 8
#define array_size_w 6
#define pad_h 1
#define pad_w 1
#define depth 3
#define kernel_s 2
#define kernel_s_conv 3

int16_t input_array[depth][array_size_h][array_size_w];
int16_t output_array_pad[depth][array_size_h + 2 * pad_h][array_size_w + 2 * pad_w];
int16_t output_array[depth][array_size_h * 2][array_size_w * 2];

int16_t input_array_conv[depth][array_size_h][array_size_w];
int16_t input_array_conv_pad[depth][array_size_h + pad_h * 2][array_size_w + pad_w * 2];
int16_t kernel_array_conv[depth * 2][depth][kernel_s_conv][kernel_s_conv];
int16_t bias_array_conv[depth * 2];
int16_t output_array_conv[depth * 2][array_size_h][array_size_w];


void padding2d_fix16_test(uint16_t padding_height, uint16_t padding_width,
uint16_t input_depth, uint16_t input_height, uint16_t input_width, int16_t input[input_depth][input_height][input_width],
uint16_t output_height, uint16_t output_width, int16_t output[input_depth][output_height][output_width]){
    
    padding2d_fix16(padding_height, padding_width,
            input_depth, input_height, input_width, input,
            output_height, output_width, output);
    
    printf("\r\n=== padding2d_test ===\r\n=== input ===\r\n");
    array_printf_3D(input_depth, input_height, input_width, input);
    printf("\r\n\r\n=== output ===\r\n");
    array_printf_3D(input_depth, output_height, output_width, output);
    printf("\r\n\r\n");
}

void up_sampling2d_fix16_test(uint16_t kernel_size,
uint16_t input_depth, uint16_t input_height, uint16_t input_width, int16_t input[input_depth][input_height][input_width],
uint16_t output_depth, uint16_t output_height, uint16_t output_width, int16_t output[output_depth][output_height][output_width]){
    up_sampling2d_fix16(kernel_size,
    input_depth, input_height, input_width, input,
    output_depth, output_height, output_width, output);

    printf("\r\n=== up_sampling2d_test ===\r\n=== input ===\r\n");
    array_printf_3D(input_depth, input_height, input_width, input);
    printf("\r\n\r\n=== output ===\r\n");
    array_printf_3D(output_depth, output_height, output_width, output);
    printf("\r\n\r\n");
}

void max_pooling2d_fix16_test(uint16_t kernel_size,
uint16_t input_depth, uint16_t input_height, uint16_t input_width, int16_t input[input_depth][input_height][input_width],
uint16_t output_depth, uint16_t output_height, uint16_t output_width, int16_t output[output_depth][output_height][output_width]){

    max_pooling2d_fix16(kernel_size,
    input_depth, input_height, input_width, input,
    output_depth, output_height, output_width, output);

    printf("\r\n=== max_pooling2d_test ===\r\n=== input ===\r\n");
    array_printf_3D(input_depth, input_height, input_width, input);
    printf("\r\n\r\n=== output ===\r\n");
    array_printf_3D(output_depth, output_height, output_width, output);
    printf("\r\n\r\n");

}

void conv2d_fix16_test(uint16_t input_depth, uint16_t input_height, uint16_t input_width, int16_t input[input_depth][input_height][input_width],
uint16_t output_depth, uint16_t output_height, uint16_t output_width, int16_t output[output_depth][output_height][output_width],
int16_t bias[output_depth],
uint16_t kernel_height, uint16_t kernel_width, int16_t kernel[output_depth][input_depth][kernel_height][kernel_width],
uint8_t relu, uint8_t fractal_width){
    conv2d_fix16(input_depth, input_height, input_width, input,
            output_depth, output_height, output_width, output,
            bias,
            kernel_height, kernel_width, kernel,
            relu, fractal_width);
    
    printf("\r\n=== conv2d_test ===\r\n=== kernel ===\r\n");
    array_printf_4D(output_depth, input_depth, kernel_height, kernel_width, kernel);
    printf("\r\n\r\n=== input ===\r\n");
    array_printf_3D(input_depth, input_height, input_width, input);
    printf("\r\n\r\n=== output ===\r\n");
    array_printf_3D(output_depth, output_height, output_width, output);
    printf("\r\n\r\n");
}

int main(void){
    /*
     *  test conv2d
     */

    for(int i = 0; i < depth; i++){
        for(int j = 0; j < array_size_h; j++){
            for(int k = 0; k < array_size_w; k++){
                input_array_conv[i][j][k] = (j + 1) * (k + 1);
            }
        }
    }

    for(int l = 0; l < depth * 2; l++){
        for(int i = 0; i < depth; i++){
            for(int j = 0; j < kernel_s_conv; j++){
                for(int k = 0; k < kernel_s_conv; k++){
                    kernel_array_conv[l][i][j][k] = (i + 1);
                }
            }
        }
    }

    for(int i = 0; i < depth * 2; i++){
        for(int j = 0; j < array_size_h; j++){
            for(int k = 0; k < array_size_w; k++){
                output_array_conv[i][j][k] = 0;
            }
        }
    }

    for(int i = 0; i < depth * 2; i++){
        bias_array_conv[i] = 0;
    }

    padding2d_fix16_test(pad_h, pad_w,
    depth, array_size_h, array_size_w, input_array_conv,
    array_size_h + 2 * pad_h, array_size_w + 2 * pad_w, input_array_conv_pad);


    conv2d_fix16_test(depth, array_size_h + 2 * pad_h, array_size_w + 2 * pad_w, input_array_conv_pad,
    2 * depth, array_size_h, array_size_w, output_array_conv,
    bias_array_conv,
    kernel_s_conv, kernel_s_conv, kernel_array_conv,
    0, 0);

    /*
     *  test padding2d
     */

    for(int i = 0; i < depth; i++){
        for(int j = 0; j < array_size_h; j++){
            for(int k = 0; k < array_size_w; k++){
                input_array[i][j][k] = (j + 1) * (k + 1);
            }
        }
    }

    for(int i = 0; i < depth; i++){
        for(int j = 0; j < array_size_h + 2 * pad_h; j++){
            for(int k = 0; k < array_size_w + 2 * pad_w; k++){
                output_array_pad[i][j][k] = j + 1;
            }
        }
    }

    padding2d_fix16_test(pad_h, pad_w,
    depth, array_size_h, array_size_w, input_array,
    array_size_h + 2 * pad_h, array_size_w + 2 * pad_w, output_array_pad);

    /*
     *  test up_sampling2d and max_pooling2d
     */

    for(int i = 0; i < depth; i++){
        for(int j = 0; j < array_size_h; j++){
            for(int k = 0; k < array_size_w; k++){
                input_array[i][j][k] = (j + 1) * (k + 1);
            }
        }
    }

    for(int i = 0; i < depth; i++){
        for(int j = 0; j < array_size_h * 2; j++){
            for(int k = 0; k < array_size_w * 2; k++){
                output_array[i][j][k] = j + 1;
            }
        }
    }

    up_sampling2d_fix16_test(kernel_s,
    depth, array_size_h, array_size_w, input_array,
    depth, array_size_h * 2, array_size_w * 2, output_array);

    for(int i = 0; i < depth; i++){
        for(int j = 0; j < array_size_h; j++){
            for(int k = 0; k < array_size_w; k++){
                input_array[i][j][k] = 0;
            }
        }
    }

    max_pooling2d_fix16_test(kernel_s,
    depth, array_size_h * 2, array_size_w * 2, output_array,
    depth, array_size_h, array_size_w, input_array);

    return(0);
}