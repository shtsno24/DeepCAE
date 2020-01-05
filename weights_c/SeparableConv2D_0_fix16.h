/*
 * author : shtsno24
 * Date : 2020-01-05 17:26:08.370546
 * array_type : int16
 * fractal_width : 13 bit
 * bit_width : 16 bit
 *
 */
#pragma once
#include <stdint.h>

#define data_width_SeparableConv2D_0 16
#define fractal_width_SeparableConv2D_0 13

const uint16_t shape_SeparableConv2D_0_w_d[] = {1, 1, 3, 3};
const int16_t SeparableConv2D_0_w_d[1][1][3][3] =
{{{{  504,  1016,   -25},
{ 1523,  3820,  1130},
{  478,  2078,   803}}}};

const uint16_t shape_SeparableConv2D_0_w_p[] = {16, 1, 1, 1};
const int16_t SeparableConv2D_0_w_p[16] =
{ 1529, -3315, -2145, -1851,  4671,  3148,  3768, -4060,  2451,  3116,  4782, -3673, -2333,    25,  2437,   637};

const uint16_t shape_SeparableConv2D_0_b_d = 1;
const int16_t SeparableConv2D_0_b_d[1] = {    0};
const uint16_t shape_SeparableConv2D_0_b_p = 16;
const int16_t SeparableConv2D_0_b_p[16] = {   42,  -126,  2975,  2544,   -25,  3715,    36,  -324,  5030,     0,   907,   -70,  3224,  -222,  3298,  3923};
