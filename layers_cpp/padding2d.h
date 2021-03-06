#include <cstdint>
#include <vector>
#include "conv2d.h"

using namespace std;

uint8_t padding2d_fix16(uint16_t padding_height, uint16_t padding_width,
uint16_t input_depth, uint16_t input_height, uint16_t input_width, vector< vector< vector< int16_t> > >& input,
uint16_t output_height, uint16_t output_width, vector< vector< vector< int16_t> > >& output);

uint8_t padding2d_float32(uint16_t padding_height, uint16_t padding_width,
uint16_t input_depth, uint16_t input_height, uint16_t input_width, vector< vector< vector< float> > >& input,
uint16_t output_height, uint16_t output_width, vector< vector< vector< float> > >& output);