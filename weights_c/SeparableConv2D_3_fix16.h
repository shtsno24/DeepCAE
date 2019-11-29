/*
 * author : shtsno24
 * Date : 2019-11-29 22:24:29.075230
 * array_type : int16
 * fractal_width : 14 bit
 * bit_width : 16 bit
 *
 */
#pragma once
#include <stdint.h>

#define data_width_SeparableConv2D_3 16
#define fractal_width_SeparableConv2D_3 14

const uint16_t shape_SeparableConv2D_3_w_d[] = {1, 8, 3, 3};
const int16_t SeparableConv2D_3_w_d[1][8][3][3] =
{{{{ 6483,  6800,  1793},
{ 1754, -2268,  1790},
{-5240,  1297,  4003}},

{{ 4126,   478, -1728},
{ -500, -3686,   206},
{   70,  1164,  3027}},

{{-2759,  -475, -3147},
{-1026,  1453,  1979},
{  -30,   699,  2001}},

{{ -931, -1964,    88},
{ 3610,  3068,  2235},
{ 2735,  5515,  2644}},

{{ 2054, -1082,  -282},
{-2935, -2559,  -806},
{ 5151,  -202,   459}},

{{ 1691,  6153,  3979},
{ 2879,  8324,  5482},
{  270,  1706, -1204}},

{{  587,  -445, -1591},
{ 5546,  2445, -3794},
{ 5844,   305,  1431}},

{{ 4402, -2220,  1032},
{-2557, -3565,  4049},
{ 3118,  2826, -2191}}}};

const uint16_t shape_SeparableConv2D_3_w_p[] = {16, 8, 1, 1};
const int16_t SeparableConv2D_3_w_p[128] =
{-2651,  2073,  1156,  3349,   604, -8156, -5028,  6387, -10026,  5920,  1322, -7540,  3993,  8863, 10066,  6497, -5839,  4141, -5493,  7347, -1488, -2148, -8023,  6361,  2258,  -967,  -946, -7367,  6947, -7301,  1582, -5886,  3762, -5238,  4505, -7039, -1746,   953,  5991, -7318, -4192,  7946, -6801, -6934,  6251, -7325, -4293, -7732,  6991, -2587,  1501, -4347, -6826,   535,  7485,   -99, -4407,  6362, -4718, -4734,  2770, -1312,  2669, -2859, -4441, -5484,  6676, -4288, -7260,   669, -6517, -2572, -6350,  2136,  4793,  7198, -6744,  8508, -5838,  6660,  8779,  7883, -1167, -1243,  3744,  2181,  4585, -2179,   738, -7926,  4738,  2494,  2931, -7453,  7364,  -767, -4640, -6124, -4374, -1684,  9249,  9727,   -67,  1304, -6983,   788,  5068, -2722, -5044, -8187, -1037,  7848, -2065, -7293,  2987, -5070,  -260, -7630,  -228, -2840, -2855, -8214, -2709, -5582, -4899,  4232,  5145,  1740};

const uint16_t shape_SeparableConv2D_3_b_d = 8;
const int16_t SeparableConv2D_3_b_d[8] = {    0,     0,     0,     0,     0,     0,     0,     0};
const uint16_t shape_SeparableConv2D_3_b_p = 16;
const int16_t SeparableConv2D_3_b_p[16] = {   14, -3325,   262,     1,   -56,     0,    -5,   402,  -323, -3385,  4336,     0, -4652,    59,    76, -1950};
