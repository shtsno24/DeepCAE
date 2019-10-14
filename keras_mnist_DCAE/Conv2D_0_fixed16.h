/*
 * author : shtsno24
 * Date : 2019-10-14 11:51:21.800380
 * array_type : int16
 * fractal_width : 8 bit
 * bit_width : 16 bit
 *
 */
#pragma once
#include <stdint.h>

#define data_width_Conv2D_0 16
#define fractal_width_Conv2D_0 8

const uint16_t shape_Conv2D_0_w[] = {16, 1, 3, 3};
const int16_t Conv2D_0_w[16][1][3][3] =
{{{{50, -15, 31},
{-15, -16, 42},
{29, 50, 50}}},

{{{-22, -24, -6},
{-11, 15, 21},
{-27, -33, -11}}},

{{{-27, -27, -29},
{-28, 40, 3},
{-36, -1, 20}}},

{{{34, -9, 47},
{-6, -36, -49},
{-36, 28, 4}}},

{{{47, 33, 49},
{-25, -8, -15},
{48, -37, -16}}},

{{{42, -20, -43},
{-26, -14, -27},
{-6, 25, -16}}},

{{{-21, -36, 3},
{21, 21, 17},
{37, -37, -42}}},

{{{-19, 40, 27},
{-48, -10, 41},
{-47, 40, -44}}},

{{{-19, -7, 50},
{30, 44, -8},
{-9, 7, 27}}},

{{{-29, 32, -45},
{-32, -17, -20},
{8, -10, -2}}},

{{{-16, -2, -22},
{-38, -45, -14},
{22, 8, -3}}},

{{{-40, 7, -2},
{-22, 17, 43},
{47, -27, -33}}},

{{{49, 8, 8},
{8, 22, -48},
{44, 23, 20}}},

{{{28, 37, 32},
{26, 49, 8},
{19, -30, -41}}},

{{{45, 39, -41},
{35, -48, 29},
{51, 32, 32}}},

{{{-49, -39, -7},
{10, 39, 18},
{-4, 37, 13}}}};

const uint16_t shape_Conv2D_0_b = 16;
const int16_t Conv2D_0_b[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
