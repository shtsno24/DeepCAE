/*
 * author : shtsno24
 * Date : 2019-11-29 22:24:28.974207
 * array_type : int16
 * fractal_width : 14 bit
 * bit_width : 16 bit
 *
 */
#pragma once
#include <cstdint>
#include <vector>

using namespace std;

#define data_width_SeparableConv2D_0 16
#define fractal_width_SeparableConv2D_0 14

const vector< uint16_t> shape_SeparableConv2D_0_w_d = {1, 1, 3, 3};
const vector< vector< vector< vector< int16_t> > > > SeparableConv2D_0_w_d =
{{{{-12554, -12685,  1933},
{-5322, -7602, -1037},
{ -783,  3015, -3653}}}};

const vector< uint16_t> shape_SeparableConv2D_0_w_p = {16, 1, 1, 1};
const vector< vector< vector< vector< int16_t> > > > SeparableConv2D_0_w_p =
{{{{-1770}}},

{{{ 6639}}},

{{{ 8093}}},

{{{ -628}}},

{{{ 4920}}},

{{{ 9618}}},

{{{-9253}}},

{{{ 1545}}},

{{{-10856}}},

{{{ 2733}}},

{{{ 1194}}},

{{{ 6035}}},

{{{ 3956}}},

{{{-6935}}},

{{{ 5414}}},

{{{ 6639}}}};

const uint16_t shape_SeparableConv2D_0_b_d = 1;
const vector< int16_t> SeparableConv2D_0_b_d = {    0};
const uint16_t shape_SeparableConv2D_0_b_p = 16;
const vector< int16_t> SeparableConv2D_0_b_p = { 1542, -1048,  -333,   783,  -788,  -804,   758,  -342,  1083,  2512,   441,     0,  -963,   103,  -697,  -852};
