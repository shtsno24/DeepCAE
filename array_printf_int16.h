#include <stdint.h>

void array_printf_1D_int16(uint16_t input_length, 
int16_t input[input_length]);

void array_printf_2D_int16(uint16_t input_height, uint16_t input_width, 
int16_t input[input_height][input_width]);

void array_printf_3D_int16(uint16_t input_depth, uint16_t input_height, uint16_t input_width, 
int16_t input[input_depth][input_height][input_width]);

void array_printf_4D_int16(uint16_t output_depth, uint16_t input_depth, uint16_t input_height, uint16_t input_width, 
int16_t input[output_depth][input_depth][input_height][input_width]);