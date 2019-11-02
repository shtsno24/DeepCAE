/*
 * author : shtsno24
 * Date : 2019-10-30 10:38:44.567233
 * array_type : int16
 * fractal_width : 14 bit
 * bit_width : 16 bit
 *
 */
#pragma once
#include <stdint.h>

#define data_width_Conv2D_2 16
#define fractal_width_Conv2D_2 14

const uint16_t shape_Conv2D_2_w[] = {8, 8, 3, 3};
const int16_t Conv2D_2_w[8][8][3][3] =
{{{{-2378, -1084, -1191},
{-1861, -1715,  -851},
{  700,  2240,  3018}},

{{  387, -1535, -1189},
{ 2833,  2973,  3056},
{ 1684, -2428,  1739}},

{{-3157, -1283,  -936},
{  484,    48,   622},
{ -367,  2098, -3091}},

{{-1110,  2805, -2013},
{   39, -2327,   987},
{ 3064,  1560,   875}},

{{ 1475, -1747, -3336},
{  -38, -2154, -1529},
{ 2005, -2559,  1305}},

{{-1362,  3159,   504},
{  801, -1234,   914},
{ -250, -3013, -3281}},

{{ 1801,  1330,   677},
{  464,   171,   446},
{-1238,  3308,  1147}},

{{  561, -1738,   976},
{ 2493,  1013, -3292},
{-1775, -2445,  -283}}},

{{{-2359, -2950, -1771},
{ 1348,  2524,   690},
{  759,  1365, -3235}},

{{ 1677,  -987,  1042},
{-3252, -2258, -2932},
{ -930,  1924, -1162}},

{{ 1003,   803, -1128},
{-1206,  1228,  -532},
{ 1201,  1990,  1666}},

{{  561, -2891,  1645},
{-2955,  2307,  2213},
{  482,  2818,  3247}},

{{ -656,   924, -1717},
{ 2661,  -704, -1975},
{ 2129,  1734, -3123}},

{{-1600,  1224,  2606},
{ 3311,  3233,  -346},
{ -749,  1548,  -180}},

{{ -216,    28, -1251},
{ -557,   391, -1371},
{-1946,  2869, -2484}},

{{  702,   323,  2689},
{ -634,  3296,  2099},
{ 1135, -3089, -2136}}},

{{{ 1164, -2762,  2563},
{-1135,  2716,  2698},
{  337,   882,  3219}},

{{  -30, -3195,  1575},
{ 1300,  3131, -3114},
{  222, -1018, -2250}},

{{ 2073,   568, -1864},
{    1, -1333,  3117},
{ -366,   944,   692}},

{{ 2024, -1397,  2238},
{ -229,   686, -3365},
{ -371,  -228, -1961}},

{{-2104,  2750,  2165},
{ 1586,  1050,  1314},
{-1478, -1041, -1114}},

{{ 1338, -1061,   860},
{-2260,  1088, -2764},
{  927,  2229,   914}},

{{ 1035,  -615, -3307},
{-1632,  2035,  2630},
{ -667, -1033,  2362}},

{{  -99,  1539, -1992},
{  252,  -531,  3322},
{  499,   325,  -576}}},

{{{ 2193,  1658, -1399},
{ -434,  2317,  1261},
{  802, -1553,  -790}},

{{-2792,  -391,  -227},
{ -818,  2711, -2562},
{  860,  2394,   -51}},

{{-2865,  3229,  2233},
{ 1053,  2522, -2974},
{ 1356, -2602,  2619}},

{{-1293,  1419, -1793},
{ 3192,     6, -2554},
{ 1537,  -533, -1460}},

{{ 1746,  1405, -2662},
{-2134, -1066,  3217},
{ 1513,  2327,  2881}},

{{-1694, -1363,   808},
{ -159,  -166, -2090},
{ 2508,   633, -1298}},

{{  631,  1519, -2838},
{  490,   901,  -940},
{-3083, -1893,  1575}},

{{-1349,  2147,  2554},
{  116,   773, -1602},
{  -98,  2042,  1093}}},

{{{ 3150,    75, -1408},
{  636,  -922, -2847},
{ 1653, -1917, -1631}},

{{-2330,   626, -1818},
{-1297,   373,   833},
{ 2504,   422,  1532}},

{{-1507, -2122,  3218},
{  162, -3239,   237},
{ 1168,   731,  1566}},

{{-1734,   -46,   774},
{-2912, -2553,  2372},
{-2135,  -417,  1404}},

{{-1475,    41,  1357},
{ 1089,  -116,  2074},
{  302,    50,  2662}},

{{ -573,  2637, -2620},
{  117,  2640, -1936},
{-1715,  3117, -2644}},

{{ 1741, -2264, -2697},
{    9,  1101,   991},
{-2310,  -626,   565}},

{{-2718,  -814,   288},
{ 1911,   -86,  1084},
{ -147, -2037,  2631}}},

{{{-1923,   370, -1617},
{-3167, -3314, -3043},
{ 1991,  -379,  -470}},

{{ 1045,   912,  2441},
{ 2133,  2678, -2035},
{-2818, -1761, -2487}},

{{ 1243, -1714,  -885},
{ 1063,  3016,   690},
{ -345, -2654,  3122}},

{{  510,  -555,  -845},
{ 2084, -3103, -1289},
{  665,   474,   621}},

{{  850,  1646,  1953},
{ 2664,  2586,    11},
{  705,   285,  -568}},

{{  911,  1418,   609},
{-2894, -1944,  1077},
{  266,  2365,  2426}},

{{ 2742,  3128,  -487},
{  496,  -173,  1242},
{ 1629, -2397,  1145}},

{{  917,  -923,  2160},
{ 1672,  2697,  -107},
{  -68,  2323, -3168}}},

{{{ -173, -2497,  -474},
{ 1245,  1745, -2195},
{ -620,   475,  -601}},

{{ 1225,  2140, -1912},
{ -803, -3004, -1496},
{-1009,  -744,  2168}},

{{ 1302, -1018,   481},
{  179,   404,  1473},
{-1025,  1750,  3025}},

{{-1053,  1885, -2751},
{-2743,   509,   152},
{ 1413, -3112,  3014}},

{{ 2246,  -157, -3211},
{   86,  -379,  2176},
{ -301,  -965, -1816}},

{{ -357,   534,  2623},
{ -889,   558, -2526},
{  -47, -1206,    96}},

{{ 2677,  2481,  -141},
{-2896,  1564,  2762},
{   67,  -306,  -394}},

{{ 1077,  -146,  1217},
{ -757,   608,   954},
{-1647, -1586, -3060}}},

{{{-1026,  1719, -2840},
{-1370,    73,   256},
{-1531, -3163,  3072}},

{{  133,  -233,   927},
{-1802,   941,  2489},
{-1401,  1624,  2945}},

{{ 2772,  -642,  1984},
{ -265,  3271,  -269},
{ 1478,  2967,  2763}},

{{ 2843,  3118, -1527},
{ 1688,  -505, -1172},
{ 2621,   569,  1007}},

{{ -910,  2958,  -460},
{   59, -3160,  -242},
{-1537,  -267,  2609}},

{{   -9,  2165,   -35},
{-1096,  2082,  1480},
{ -900, -2728,   552}},

{{ 2446,  1176,  1047},
{-2153, -1749,  -884},
{ -880, -1058,  -754}},

{{ -757,   814, -2232},
{-2666,  -481,  1282},
{ 2624, -2283,  1370}}}};

const uint16_t shape_Conv2D_2_b = 8;
const int16_t Conv2D_2_b[8] = {  -14,     9,   -21,     4,    62,   -79,    -2,   -20};