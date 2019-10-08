
typedef struct{
    uint16_t height, width, depth;
}ARRAY_PARAMS_3D;

uint8_t conv2d(uint16_t kernel_depth, uint16_t kernel_height, uint16_t kernel_width, int16_t kernel[kernel_depth][kernel_height][kernel_depth],
uint16_t input_depth, uint16_t input_height, uint16_t input_width, int16_t input[input_depth][input_height][input_depth],
uint16_t bias_depth, int16_t bias[bias_depth],
uint16_t output_depth, uint16_t output_height, uint16_t output_width, int16_t output[output_depth][output_height][output_depth],
uint8_t relu);
