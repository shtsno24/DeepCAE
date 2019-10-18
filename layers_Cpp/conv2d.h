#include <cstdint>
#include <vector>

using namespace std;
// typedef vector<int16_t> fix16_1D;
// typedef vector<fix16_1D> fix16_2D;
// typedef vector<fix16_2D> fix16_3D;
// typedef vector<fix16_3D> fix16_4D;


uint8_t conv2d_fix16(uint16_t input_depth, uint16_t input_height, uint16_t input_width, vector< vector< vector< int16_t> > >& input,
uint16_t output_depth, uint16_t output_height, uint16_t output_width, vector< vector< vector< int16_t> > >& output,
const vector< int16_t >& bias,
uint16_t kernel_height, uint16_t kernel_width, const vector< vector< vector< vector< int16_t> > > >& kernel,
uint8_t relu, uint8_t fractal_width);

uint8_t conv2d_float32(uint16_t input_depth, uint16_t input_height, uint16_t input_width,  vector< vector< vector< float> > >& input,
uint16_t output_depth, uint16_t output_height, uint16_t output_width, vector< vector< vector< float> > >& output,
const vector< int16_t >& bias,
uint16_t kernel_height, uint16_t kernel_width, const vector< vector< vector< vector< int16_t> > > >& kernel,
uint8_t relu);

