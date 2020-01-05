/*
 * author : shtsno24
 * Date : 2020-01-05 16:38:33.306018
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
{{{{ 1008,  2033,   -51},
{ 3047,  7641,  2261},
{  956,  4156,  1606}}}};

const vector< uint16_t> shape_SeparableConv2D_0_w_p = {16, 1, 1, 1};
const vector< vector< vector< vector< int16_t> > > > SeparableConv2D_0_w_p =
{{{{ 3059}}},

{{{-6632}}},

{{{-4290}}},

{{{-3703}}},

{{{ 9343}}},

{{{ 6298}}},

{{{ 7537}}},

{{{-8120}}},

{{{ 4902}}},

{{{ 6232}}},

{{{ 9565}}},

{{{-7346}}},

{{{-4666}}},

{{{   50}}},

{{{ 4875}}},

{{{ 1274}}}};

const uint16_t shape_SeparableConv2D_0_b_d = 1;
const vector< int16_t> SeparableConv2D_0_b_d = {    0};
const uint16_t shape_SeparableConv2D_0_b_p = 16;
const vector< int16_t> SeparableConv2D_0_b_p = {   84,  -253,  5950,  5088,   -51,  7430,    73,  -648, 10061,     0,  1815,  -141,  6449,  -444,  6597,  7846};
