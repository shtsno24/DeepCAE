/*
 * author : shtsno24
 * Date : 2020-01-06 13:05:29.758609
 * array_type : int16
 * fractal_width : 13 bit
 * bit_width : 16 bit
 *
 */
#pragma once
#include <cstdint>
#include <vector>

using namespace std;

#define data_width_SeparableConv2D_4 16
#define fractal_width_SeparableConv2D_4 13

const vector< uint16_t> shape_SeparableConv2D_4_w_d = {1, 16, 3, 3};
const vector< vector< vector< vector< int16_t> > > > SeparableConv2D_4_w_d =
{{{{-4715, -1366, -2767},
{ -816,  3055,  3317},
{-2644,  1361, -1130}},

{{ 4505,  3985,  -144},
{ 4410, -1096, -2029},
{   70, -2445,  -926}},

{{ 1660, -1307,  1797},
{-1099, -6824, -4007},
{ 1900, -2380,    48}},

{{ -896,  1612,  -353},
{ 1833,  3511,  1836},
{ -335,  1941, -1096}},

{{ 6302, 10461,  2818},
{ 2066, -3739,  1710},
{ 3152,  7991,  2402}},

{{ 4246,  2551,  4200},
{ 3819, -1659, -10760},
{ -753, -9437, -5126}},

{{-2933,  3187,   631},
{ 3670,  7734,  4271},
{  235,  3527, -1879}},

{{ 3050,  4007,  2611},
{ 4037,  -684, -1631},
{ 3355,  -509, -1179}},

{{ 1978, -2401,   348},
{-3090, -9810, -4008},
{ -251, -2388,  2082}},

{{ 8327,  4211,  3087},
{ 3356,   752, -2489},
{ 1164,  -782,  -357}},

{{-2944, -2914, -1428},
{  -89, -4330,   104},
{ -379, -2829, -1243}},

{{  271, -4208, -1059},
{-5121, -7630, -5936},
{  311, -5056,  1833}},

{{ 4140, -1505, -1278},
{-2116, -5438,  -329},
{-2246,  -776,  4213}},

{{-2397, -1677, -3577},
{ 1141,   693,  3845},
{-3508,    67,   307}},

{{ 5517,  3017,  4460},
{ 3636, -9134,  5940},
{ 2718,  5989,  6247}},

{{  880, -1035,   904},
{-3687, -3983, -3312},
{ 1197,  -659,   510}}}};

const vector< uint16_t> shape_SeparableConv2D_4_w_p = {1, 16, 1, 1};
const vector< vector< vector< vector< int16_t> > > > SeparableConv2D_4_w_p =
{{{{-10450}},

{{-2469}},

{{-6603}},

{{-5012}},

{{-7282}},

{{ 4203}},

{{-4648}},

{{-5960}},

{{ 8633}},

{{-3476}},

{{-5839}},

{{-8678}},

{{-5088}},

{{ 5693}},

{{-8689}},

{{ 6196}}}};

const uint16_t shape_SeparableConv2D_4_b_d = 16;
const vector< int16_t> SeparableConv2D_4_b_d = {    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0};
const uint16_t shape_SeparableConv2D_4_b_p = 1;
const vector< int16_t> SeparableConv2D_4_b_p = {-6470};
