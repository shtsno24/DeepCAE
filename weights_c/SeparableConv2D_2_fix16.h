/*
 * author : shtsno24
 * Date : 2019-11-13 13:45:38.553172
 * array_type : int16
 * fractal_width : 14 bit
 * bit_width : 16 bit
 *
 */
#pragma once
#include <stdint.h>

#define data_width_SeparableConv2D_2 16
#define fractal_width_SeparableConv2D_2 14

const uint16_t shape_SeparableConv2D_2_w_d[] = {1, 8, 3, 3};
const int16_t SeparableConv2D_2_w_d[1][8][3][3] =
{{{{ 2536, -6851,  2756},
{-1423,    63,  9249},
{ -101, -3198, -5076}},

{{  843,   864,  3946},
{ 2454,  -240,  4605},
{  635,  5447,   940}},

{{ 3744,  3585, -4107},
{  394,  -275,  3632},
{ 4332,   915, -3506}},

{{  410, -1860, -3467},
{-2970,  1565, -5061},
{-2905,  -580, -3411}},

{{-4123, -2067,  1639},
{-4528, -2755,  2186},
{-4038, -1126,  1206}},

{{  349, -1179,  2428},
{-1867,  9229,  2740},
{-1920,  2935,   667}},

{{-2151,  1705,  -645},
{ 3635,  3630, -2133},
{  495,   287,  1052}},

{{   33,  2247,  -537},
{ 1212, -2449,  3377},
{  733,  1342, -1115}}}};

const uint16_t shape_SeparableConv2D_2_w_p[] = {8, 8, 1, 1};
const int16_t SeparableConv2D_2_w_p[8][8][1][1] =
{{{{10898}},

{{ 6361}},

{{-3520}},

{{ 3573}},

{{-7765}},

{{ -153}},

{{10043}},

{{ 6976}}},

{{{  573}},

{{-7188}},

{{ 6348}},

{{-9376}},

{{-9190}},

{{-5413}},

{{ -876}},

{{  -40}}},

{{{-8324}},

{{ 2097}},

{{-9487}},

{{-6635}},

{{  607}},

{{-8081}},

{{ 7867}},

{{-5466}}},

{{{11547}},

{{ 8413}},

{{-1735}},

{{ 8485}},

{{ 2665}},

{{ 3423}},

{{ -914}},

{{ 4560}}},

{{{-6048}},

{{ 7028}},

{{-5818}},

{{ 9405}},

{{ 6586}},

{{ 4944}},

{{ 7540}},

{{ 8629}}},

{{{ 6564}},

{{-10620}},

{{-1783}},

{{ 5899}},

{{-8276}},

{{12149}},

{{ 2883}},

{{-4728}}},

{{{ 9628}},

{{ 8093}},

{{  733}},

{{ 9047}},

{{-4274}},

{{-1225}},

{{-6631}},

{{-3726}}},

{{{ 1720}},

{{ 2151}},

{{-1930}},

{{ 5081}},

{{ -409}},

{{-5892}},

{{-4037}},

{{ 1956}}}};

const uint16_t shape_SeparableConv2D_2_b_d = 8;
const int16_t SeparableConv2D_2_b_d[8] = {    0,     0,     0,     0,     0,     0,     0,     0};
const uint16_t shape_SeparableConv2D_2_b_p = 8;
const int16_t SeparableConv2D_2_b_p[8] = {  414,     4,    42, -3159,    11,   159,  2276, -1477};
