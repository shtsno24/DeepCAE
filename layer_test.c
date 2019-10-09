#include <stdio.h>
#include <stdint.h>

#include "array_printf.h"
#include "layers/padding2d.h"

#define array_size_h 4
#define array_size_w 2
#define pad_h 1
#define pad_w 1
#define depth 3

int16_t input_array[depth][array_size_h][array_size_w];
int16_t output_array[depth][array_size_h + 2 * pad_h][array_size_w + 2 * pad_w];

void padding2d_test(uint16_t padding_height, uint16_t padding_width,
uint16_t input_depth, uint16_t input_height, uint16_t input_width, int16_t input[input_depth][input_height][input_width],
uint16_t output_height, uint16_t output_width, int16_t output[input_depth][output_height][output_width]){
    
    padding2d(padding_height, padding_width,
            input_depth, input_height, input_width, input,
            output_height, output_width, output);
    
    array_printf_3D(input_depth, input_height, input_width, input_array);
    printf("\r\n\r\n");
    array_printf_3D(input_depth, output_height, output_width, output_array);
    printf("\r\n\r\n");
}

int main(void){
    padding2d_test(pad_h, pad_w,
    depth, array_size_h, array_size_w, input_array,
    array_size_h + 2 * pad_h, array_size_w + 2 * pad_w, output_array);

    return(0);
}