#include <stdio.h>
#include <stdint.h>

#include "array_printf.h"

#define array_size 8

int16_t test_array[2 * array_size][array_size][array_size];


int main(void){
    for(uint16_t k = 0; k < 2 * array_size; k++){
        for(uint16_t i = 0; i < array_size; i++){
            for(uint16_t j = 0; j < array_size; j++){
                test_array[k][i][j] = k;
            }
        }
    }
    array_printf_3D(2 * array_size, array_size, array_size, test_array);
}