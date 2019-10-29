/*
 * author : shtsno24
 * Date : 2019-10-30 01:16:29.380363
 * array_type : int16
 * fractal_width : 14 bit
 * bit_width : 16 bit
 *
 */
#pragma once
#include <stdint.h>

#define data_width_SeparableConv2D_0 16
#define fractal_width_SeparableConv2D_0 14

const uint16_t shape_SeparableConv2D_0_w_d[] = {1, 1, 3, 3};
const int16_t SeparableConv2D_0_w_d[1][1][3][3] =
{{{{-5214,  6212, -7478},
{-10523, -10162,  2914},
{ 4405, -6503,   333}}}};

const uint16_t shape_SeparableConv2D_0_w_p[] = {16, 1, 1, 1};
const int16_t SeparableConv2D_0_w_p[16][1][1][1] =
{{{{ 5735}}},

{{{-7103}}},

{{{ 2850}}},

{{{ 5836}}},

{{{-8881}}},

{{{ 8178}}},

{{{-6228}}},

{{{  247}}},

{{{ 3075}}},

{{{ 3237}}},

{{{-8844}}},

{{{ 2446}}},

{{{ 1543}}},

{{{ 4386}}},

{{{ 4107}}},

{{{-2047}}}};

const uint16_t shape_SeparableConv2D_0_b_d = 16;
const int16_t SeparableConv2D_0_b_d[16] = {    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0};
const uint16_t shape_SeparableConv2D_0_b_p = 16;
const int16_t SeparableConv2D_0_b_p[16] = {  169,     0,   -66,  -309,    24,  -144,     0,   -38,   -48,  -206,     5,    29,     0,   790,   -71,  -331};
