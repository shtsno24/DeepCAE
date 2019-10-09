#include <stdint.h>
#include "padding2d.h"


uint8_t padding2d(uint16_t padding,
uint16_t input_depth, uint16_t input_height, uint16_t input_width, int16_t input[input_depth][input_height][input_width],
uint16_t output_height, uint16_t output_width, int16_t output[input_depth][output_height][output_width]){

    for(uint16_t depth = 0; depth < input_depth; depth++){
        for(uint16_t height = 0; height < padding * 2 + input_height; height++){
            if(height < padding){
                for(uint16_t width = 0; width < padding * 2 + input_width; width++){
                    output[depth][height][width] = 0;
                }
            } else if (height>=padding && height < padding + input_height){
                for(uint16_t width = 0; width < padding * 2 + input_width; width++){
                    if(width < padding){
                        output[depth][height][width] = 0;
                    }else if(width>=padding && width < padding + input_width){
                        output[depth][height][width] = input[depth][height-padding][width - padding];
                    }else{
                        output[depth][height][width] = 0;
                    }
                }
            } else {
                for(uint16_t width = 0; width < padding * 2 + input_width; width++){
                    output[depth][height][width] = 0;
                }
            }

        }
    }
}