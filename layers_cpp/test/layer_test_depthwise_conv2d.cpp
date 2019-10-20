#include <vector>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <fstream>
#include "./../array_printf_fix16.h"
#include "./../array_printf_float32.h"
#include "./../depthwise_conv2d.h"

using namespace std;

int main(void){
    vector< vector <vector< int16_t> > > input_array(3, vector< vector< int16_t>>(7, vector <int16_t>(7, 0)));
    vector< vector <vector< int16_t> > > output_array(3, vector< vector< int16_t>>(7, vector <int16_t>(7, 0)));
    vector< vector <vector< vector< int16_t> > > > kernel_array(1, vector< vector< vector< int16_t> > >(3, vector< vector< int16_t> >(3, vector< int16_t>(3, 0))));
    vector< int16_t> bias_array(3, 0);


    for(uint16_t d = 0; d < 3; d++){
        for(uint16_t h = 0; h < 7; h++){
            for(uint16_t w = 0; w < 7; w++){
                input_array[d][h][w] = w + 1;
            }
        }
    }

    for(uint16_t d = 0; d < 3; d++){
        for(uint16_t h = 0; h < 3; h++){
            for(uint16_t w = 0; w < 3; w++){
                kernel_array[1][d][h][w] = (h * 1) + (w + 1);
            }
        }
    }

    array_printf_3D_fix16(3, 7, 7, input_array, 0);

    depthwise_conv2d_fix16(3, 7, 7, input_array,
    3, 7, 7, output_array,
    bias_array,
    3, 3, kernel_array, 1, 0);

    array_printf_3D_fix16(3, 7, 7, output_array, 0);
}