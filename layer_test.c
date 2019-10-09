#include <stdio.h>
#include <stdint.h>

#include "array_printf.h"
#include "layers/padding2d.h"
#include "layers/up_sampling2d.h"

#define array_size_h 6
#define array_size_w 6
#define pad_h 1
#define pad_w 1
#define depth 3
#define kernel 2

int16_t input_array[depth][array_size_h][array_size_w];
// int16_t output_array[depth][array_size_h + 2 * pad_h][array_size_w + 2 * pad_w];
int16_t output_array[depth][array_size_h * 2][array_size_w * 2];

void padding2d_test(uint16_t padding_height, uint16_t padding_width,
uint16_t input_depth, uint16_t input_height, uint16_t input_width, int16_t input[input_depth][input_height][input_width],
uint16_t output_height, uint16_t output_width, int16_t output[input_depth][output_height][output_width]){
    
    padding2d(padding_height, padding_width,
            input_depth, input_height, input_width, input,
            output_height, output_width, output);
    
    array_printf_3D(input_depth, input_height, input_width, input);
    printf("\r\n\r\n");
    array_printf_3D(input_depth, output_height, output_width, output);
    printf("\r\n\r\n");
}

void up_sampling2d_test(uint16_t kernel_size,
uint16_t input_depth, uint16_t input_height, uint16_t input_width, int16_t input[input_depth][input_height][input_width],
uint16_t output_depth, uint16_t output_height, uint16_t output_width, int16_t output[output_depth][output_height][output_width]){
    up_sampling2d(kernel_size,
    input_depth, input_height, input_width, input,
    output_depth, output_height, output_width, output);

    array_printf_3D(input_depth, input_height, input_width, input);
    printf("\r\n\r\n");
    array_printf_3D(input_depth, output_height, output_width, output);
    printf("\r\n\r\n");
}

int main(void){

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
                output_array[i][j][k] = j + 1;
            }
        }
    }

    padding2d_test(pad_h, pad_w,
    depth, array_size_h, array_size_w, input_array,
    array_size_h + 2 * pad_h, array_size_w + 2 * pad_w, output_array);

    up_sampling2d_test(kernel,
    depth, array_size_h, array_size_w, input_array,
    depth, array_size_h * 2, array_size_w * 2, output_array);

    return(0);
}