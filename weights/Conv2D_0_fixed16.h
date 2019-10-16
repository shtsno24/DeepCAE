/*
 * author : shtsno24
 * Date : 2019-10-16 16:29:26.723440
 * array_type : int16
 * fractal_width : 14 bit
 * bit_width : 16 bit
 *
 */
#pragma once
#include <stdint.h>

#define data_width_Conv2D_0 16
#define fractal_width_Conv2D_0 14

const uint16_t shape_Conv2D_0_w[] = {16, 1, 3, 3};
const int16_t Conv2D_0_w[16][1][3][3] =
{{{{  453,  2078,  3302},
{  525,  3853, -2291},
{ 2010,  2172,  -152}}},

{{{-3275,  1640,   200},
{ -800,  1449,  2411},
{ 1494, -2177, -1795}}},

{{{ -743,  -106,    75},
{ 2659, -2450, -1567},
{ 2422,   542,  2639}}},

{{{ 3925,  3187,  3568},
{ 1690,  1687, -2504},
{-2031, -2459,  2024}}},

{{{ 1177,  2766,  1593},
{-2594,  1942,  1892},
{-1054, -1633,  3115}}},

{{{ -277, -2328, -1893},
{  554,   -20,  1604},
{ 3310,  3272, -1912}}},

{{{  745,   606, -2537},
{-1531, -2097,   693},
{  843, -1998, -2593}}},

{{{ -306,  2514,  1975},
{-2246,  4269,  1697},
{ 3741, -1325,  3254}}},

{{{ 1613,  1579,  -273},
{-2903,  3128,   370},
{-2764,  3229,  3744}}},

{{{-1927,  2342,   564},
{-2163, -1494, -1122},
{  425,   539,   466}}},

{{{ 3171, -3008,  2667},
{   32,  2826, -2235},
{ 2298,  1907,  2273}}},

{{{ 3966, -1565,  -672},
{ 1958,  4016,  1964},
{  880, -2101, -1305}}},

{{{-3137, -2527, -1723},
{-2360,    82,  1187},
{ 1337,  1550,    28}}},

{{{ -227,   758,   -71},
{  786, -2980,     3},
{-2845, -2365, -2787}}},

{{{ 1662,  1086,  1186},
{  -86,  -644, -2427},
{ 2652,   325,  2451}}},

{{{-1060,  1176, -2777},
{-1132, -1798,   703},
{-2151, -2326,  2128}}}};

const uint16_t shape_Conv2D_0_b = 16;
const int16_t Conv2D_0_b[16] = {  349,   406,  -132,    11,   124,   193,  -427,   162,     0,  -120,   148,     0,   -64,     0,    16,  -577};
