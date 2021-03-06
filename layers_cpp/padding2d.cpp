#include <cstdint>
#include <vector>
#include "conv2d.h"

using namespace std;

uint8_t padding2d_fix16(uint16_t padding_height, uint16_t padding_width,
uint16_t input_depth, uint16_t input_height, uint16_t input_width, vector< vector< vector< int16_t> > >& input,
uint16_t output_height, uint16_t output_width, vector< vector< vector< int16_t> > >& output){

    for(uint16_t depth = 0; depth < input_depth; depth++){
        for(uint16_t height = 0; height < padding_height * 2 + input_height; height++){
            if(height < padding_height){
                for(uint16_t width = 0; width < padding_width * 2 + input_width; width++){
                    output[depth][height][width] = 0;
                }
            } else if (height>=padding_height && height < padding_height + input_height){
                for(uint16_t width = 0; width < padding_width * 2 + input_width; width++){
                    if(width < padding_width){
                        output[depth][height][width] = 0;
                    }else if(width>=padding_width && width < padding_width + input_width){
                        output[depth][height][width] = input[depth][height-padding_height][width - padding_width];
                    }else{
                        output[depth][height][width] = 0;
                    }
                }
            } else {
                for(uint16_t width = 0; width < padding_width * 2 + input_width; width++){
                    output[depth][height][width] = 0;
                }
            }

        }
    }
    return(0);
}

uint8_t padding2d_float32(uint16_t padding_height, uint16_t padding_width,
uint16_t input_depth, uint16_t input_height, uint16_t input_width, vector< vector< vector< float> > >& input,
uint16_t output_height, uint16_t output_width, vector< vector< vector< float> > >& output){

    for(uint16_t depth = 0; depth < input_depth; depth++){
        for(uint16_t height = 0; height < padding_height * 2 + input_height; height++){
            if(height < padding_height){
                for(uint16_t width = 0; width < padding_width * 2 + input_width; width++){
                    output[depth][height][width] = 0;
                }
            } else if (height>=padding_height && height < padding_height + input_height){
                for(uint16_t width = 0; width < padding_width * 2 + input_width; width++){
                    if(width < padding_width){
                        output[depth][height][width] = 0;
                    }else if(width>=padding_width && width < padding_width + input_width){
                        output[depth][height][width] = input[depth][height-padding_height][width - padding_width];
                    }else{
                        output[depth][height][width] = 0;
                    }
                }
            } else {
                for(uint16_t width = 0; width < padding_width * 2 + input_width; width++){
                    output[depth][height][width] = 0;
                }
            }

        }
    }
    return(0);
}