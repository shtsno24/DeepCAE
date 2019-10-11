uint8_t conv2d_fix16(uint16_t input_depth, uint16_t input_height, uint16_t input_width, int16_t input[input_depth][input_height][input_width],
uint16_t output_depth, uint16_t output_height, uint16_t output_width, int16_t output[output_depth][output_height][output_width],
int16_t bias[output_depth],
uint16_t kernel_height, uint16_t kernel_width, int16_t kernel[output_depth][input_depth][kernel_height][kernel_width],
uint8_t relu);

uint8_t conv2d_float32(uint16_t input_depth, uint16_t input_height, uint16_t input_width, float input[input_depth][input_height][input_width],
uint16_t output_depth, uint16_t output_height, uint16_t output_width, float output[output_depth][output_height][output_width],
float bias[output_depth],
uint16_t kernel_height, uint16_t kernel_width, float kernel[output_depth][input_depth][kernel_height][kernel_width],
uint8_t relu);