#include <stdint.h>

uint8_t padding2d_fix16(uint16_t padding_height, uint16_t padding_width,
uint16_t input_depth, uint16_t input_height, uint16_t input_width, int16_t input[input_depth][input_height][input_width],
uint16_t output_height, uint16_t output_width, int16_t output[input_depth][output_height][output_width]);

uint8_t padding2d_float32(uint16_t padding_height, uint16_t padding_width,
uint16_t input_depth, uint16_t input_height, uint16_t input_width, float input[input_depth][input_height][input_width],
uint16_t output_height, uint16_t output_width, float output[input_depth][output_height][output_width]);