/*
 * author : shtsno24
 * Date : 2019-10-30 10:38:44.234825
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
{{{{ 3076, -3049,    40},
{-1831, -1714,   701},
{-2970, -2159, -2321}}},

{{{ 1170,  2334,   953},
{-2352, -2478,   125},
{ 3202,   484,  -346}}},

{{{ 1725,   753,  2066},
{ 2633,  1545,  -243},
{-2061,  2738,  2459}}},

{{{ 1640,  -634,  -120},
{ -198,  -821,  3122},
{-1047,  1560,   920}}},

{{{ 2111,  1524,   547},
{-1022,  3104, -2660},
{-1475,  -609, -1687}}},

{{{ 1786,  2341, -3010},
{ -346,  -670, -2446},
{ 2438,  2297,  1366}}},

{{{ -756,  1971, -1440},
{ 1478,  -332, -2611},
{ 1752, -2985,   127}}},

{{{ 2574,   872,  3148},
{ -802,  2011,  1867},
{-1063,  1715,  2218}}},

{{{  733,  2898,  2460},
{ 1216,  2105, -2653},
{ -775,   986,  2293}}},

{{{  -62,  1272,   427},
{  554,  1597, -1644},
{-2928,  -344,  2030}}},

{{{ 2262,  -417,  3170},
{ 1557,   625,  1489},
{ 1177, -1169,  2945}}},

{{{  843, -2079,  -598},
{ 1400,  2926, -2097},
{  202, -1555,  2982}}},

{{{-2622,  1903,  2254},
{ 1232,  2855,  2769},
{  758,   677,  2476}}},

{{{  696,  1411,  2725},
{ 1310, -1713, -2341},
{-2425,  -106,  1379}}},

{{{ 2985,  2624,  -321},
{  927, -1054, -1973},
{  374,  1203,  2744}}},

{{{ -412,    87, -1487},
{-3101, -3135, -1097},
{-1507, -1633, -3030}}}};

const uint16_t shape_Conv2D_0_b = 16;
const int16_t Conv2D_0_b[16] = {   92,    54,   -60,    40,    -5,   -38,    21,   -36,    23,   -51,   -17,   -52,    14,     7,    33,     3};
