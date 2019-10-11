/*
 * author : shtsno24
 * Date : 2019-10-11 15:30:13.756109
 * array_type : int16
 * fractal_width : 8 bit
 * bit_width : 16 bit
 *
 */
#pragma once
#include <stdint.h>

#define data_width_Conv2D_3 16
#define fractal_width_Conv2D_3 8

const uint16_t shape_Conv2D_3_w[] = {1, 16, 3, 3};
const int16_t Conv2D_3_w[1][16][3][3] =
{{{{12, 46, -24},
{32, 21, -24},
{19, -31, -38}},

{{-10, -39, -27},
{41, 43, 44},
{-1, 4, 41}},

{{-51, 39, 7},
{-46, -23, 9},
{-44, -31, 46}},

{{6, -46, -9},
{-10, -46, -44},
{47, -44, 20}},

{{33, 17, -39},
{-7, 25, -42},
{43, 44, 41}},

{{42, 0, 28},
{41, 16, 4},
{-20, 47, -16}},

{{-11, 10, -13},
{39, 38, -30},
{46, -20, -45}},

{{-16, -38, -11},
{-2, -19, -12},
{-7, -48, 39}},

{{43, -50, -16},
{7, -36, -10},
{-40, -4, -44}},

{{-8, 31, -26},
{17, -20, -11},
{-7, 5, 19}},

{{-36, -22, -12},
{-26, -24, 23},
{-41, -16, 14}},

{{-16, -8, -5},
{-4, 6, 21},
{10, 17, 24}},

{{22, -45, 5},
{17, -12, 19},
{-32, 32, -35}},

{{-35, -22, -18},
{-22, -50, 31},
{-19, 22, -43}},

{{0, -35, 30},
{6, -39, -50},
{-33, 49, -17}},

{{-39, 39, -38},
{23, -19, -40},
{3, 35, 39}}}};

const uint16_t shape_Conv2D_3_b = 1;
const int16_t Conv2D_3_b[1] = {-1};
