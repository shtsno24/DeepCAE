#include <stdint.h>
#include "conv2d.h"

uint8_t max_pooling2d(uint16_t kernel_size,
uint16_t input_depth, uint16_t input_height, uint16_t input_width, int16_t input[input_depth][input_height][input_width],
uint16_t output_depth, uint16_t output_height, uint16_t output_width, int16_t output[output_depth][output_height][output_width]);