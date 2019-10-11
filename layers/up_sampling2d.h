
uint8_t up_sampling2d_fix16(uint16_t kernel_size,
uint16_t input_depth, uint16_t input_height, uint16_t input_width, int16_t input[input_depth][input_height][input_width],
uint16_t output_depth, uint16_t output_height, uint16_t output_width, int16_t output[output_depth][output_height][output_width]);

uint8_t up_sampling2d_float(uint16_t kernel_size,
uint16_t input_depth, uint16_t input_height, uint16_t input_width, float input[input_depth][input_height][input_width],
uint16_t output_depth, uint16_t output_height, uint16_t output_width, float output[output_depth][output_height][output_width]);