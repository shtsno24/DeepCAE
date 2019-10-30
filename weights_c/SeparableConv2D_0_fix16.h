/*
 * author : shtsno24
 * Date : 2019-10-30 12:48:43.481130
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
{{{{-7363,  8225, -8260},
{-2992,  5014, -6059},
{-5551,  6054, -3703}}}};

const uint16_t shape_SeparableConv2D_0_w_p[] = {16, 1, 1, 1};
const int16_t SeparableConv2D_0_w_p[16][1][1][1] =
{{{{  280}}},

{{{ 2842}}},

{{{ 2913}}},

{{{ 8109}}},

{{{ 4184}}},

{{{-8437}}},

{{{-9180}}},

{{{ 1272}}},

{{{-9284}}},

{{{ 4926}}},

{{{ 8799}}},

{{{ 4325}}},

{{{ 7421}}},

{{{-1696}}},

{{{ 6741}}},

{{{ 5313}}}};

const uint16_t shape_SeparableConv2D_0_b_d = 16;
const int16_t SeparableConv2D_0_b_d[16] = {    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0};
const uint16_t shape_SeparableConv2D_0_b_p = 16;
const int16_t SeparableConv2D_0_b_p[16] = {    0,     0,     0,     0,     8,    -7,    -1,     2,    -2,     3,     0,     0,     1,    -7,     0,     1};
