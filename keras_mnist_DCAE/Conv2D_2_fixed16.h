/*
 * author : shtsno24
 * Date : 2019-10-11 15:30:13.753559
 * array_type : int16
 * fractal_width : 8 bit
 * bit_width : 16 bit
 *
 */
#pragma once
#include <stdint.h>

#define data_width_Conv2D_2 16
#define fractal_width_Conv2D_2 8

const uint16_t shape_Conv2D_2_w[] = {16, 8, 3, 3};
const int16_t Conv2D_2_w[16][8][3][3] =
{{{{21, 11, 15},
{-37, -28, 17},
{-14, 21, 0}},

{{1, -3, -9},
{-42, -34, -25},
{0, -27, -36}},

{{11, -1, 22},
{38, 26, -31},
{-34, -1, 17}},

{{-4, -5, -17},
{31, -37, 5},
{-9, -28, 41}},

{{24, 10, -21},
{-43, -2, -8},
{29, 4, 3}},

{{-32, 6, -15},
{18, 41, 29},
{6, 9, 20}},

{{-41, -5, 23},
{35, -10, -3},
{-8, -32, -35}},

{{21, 34, -14},
{0, 1, -24},
{22, 13, 18}}},

{{{18, 14, 18},
{23, 27, 26},
{13, 11, 33}},

{{-18, -37, -35},
{-26, 28, 1},
{31, -38, -14}},

{{37, 17, -21},
{-26, -2, 38},
{-18, 10, -12}},

{{-36, 36, 39},
{34, -20, -8},
{26, -30, -36}},

{{18, -21, -38},
{-30, 2, -31},
{-10, -38, 0}},

{{0, -23, -27},
{-20, -15, -9},
{21, 15, 24}},

{{-30, -32, -3},
{7, 12, 20},
{-40, -28, -10}},

{{-21, 25, -26},
{37, -30, 2},
{-25, 31, 22}}},

{{{23, -21, 21},
{-3, 5, 2},
{-34, 36, 29}},

{{-35, -6, 20},
{12, -18, -15},
{15, -2, 1}},

{{13, 37, 40},
{32, -29, 6},
{26, -5, -33}},

{{2, -30, 4},
{22, 21, -2},
{33, 13, 6}},

{{-36, -17, -6},
{34, 0, -26},
{43, -21, -26}},

{{-3, 3, 15},
{20, -24, 41},
{-17, 10, -15}},

{{22, 8, -30},
{35, -6, 24},
{-17, -23, 7}},

{{-40, -18, -42},
{29, 5, 11},
{13, 11, 3}}},

{{{21, 14, 21},
{15, 30, 9},
{-6, -41, -11}},

{{25, -5, -12},
{0, -12, -31},
{8, 4, 38}},

{{-1, -19, -36},
{-25, -17, -30},
{-31, 8, 20}},

{{4, 16, -28},
{21, -16, -5},
{38, -11, -30}},

{{-2, -34, 14},
{42, -18, -6},
{28, 33, 18}},

{{0, 0, -5},
{-41, 27, -33},
{-23, -4, 40}},

{{-27, -33, -21},
{42, -29, -24},
{-11, 3, -17}},

{{15, -9, -37},
{-20, -39, -10},
{-28, 3, -37}}},

{{{9, -34, 23},
{15, -28, -25},
{-38, 13, -38}},

{{-40, -25, -13},
{-1, -1, -16},
{21, 13, 15}},

{{-15, -3, -15},
{27, 32, 31},
{-16, 1, 41}},

{{35, 8, -40},
{-8, 16, -15},
{-17, -4, 0}},

{{15, 16, -3},
{14, 11, 21},
{8, -7, 12}},

{{5, 33, 21},
{38, 21, -28},
{-36, -31, -39}},

{{32, -13, 13},
{-12, 17, -41},
{6, 12, 15}},

{{3, -19, -8},
{28, -13, 35},
{-22, 0, -10}}},

{{{41, 39, -21},
{-18, 15, -39},
{-8, -26, 9}},

{{-34, -38, -35},
{-2, 23, -40},
{-11, 13, 40}},

{{-13, -22, 34},
{4, 1, -6},
{-6, -37, 28}},

{{0, -19, -37},
{9, -4, -13},
{-16, 37, -13}},

{{-15, -9, 36},
{-33, -9, -5},
{38, 36, -17}},

{{-20, -15, -14},
{11, 8, -11},
{33, -25, -20}},

{{32, 25, -30},
{-39, -6, 28},
{-29, 34, -40}},

{{12, -2, -17},
{26, -10, 10},
{-32, 0, -10}}},

{{{-36, -21, -28},
{41, 17, 9},
{0, -27, -20}},

{{-13, 5, -25},
{11, -4, -6},
{-16, -39, 28}},

{{-38, -14, -2},
{-24, -17, 0},
{-39, 18, 33}},

{{-36, 12, 32},
{-2, -27, -7},
{-34, 30, -13}},

{{20, 34, 25},
{-28, 21, -37},
{-32, -35, -34}},

{{-23, -34, -8},
{7, 19, 11},
{41, 16, -2}},

{{21, -24, 12},
{12, 40, -10},
{-38, 19, -1}},

{{31, 28, -33},
{9, 14, 7},
{-3, 21, 34}}},

{{{-18, 31, -23},
{1, -8, 38},
{-35, -34, -34}},

{{27, 17, 22},
{-41, 4, 38},
{35, -41, -27}},

{{28, 28, 4},
{-36, -35, -39},
{-27, -40, -16}},

{{-28, -31, -12},
{34, -25, -22},
{6, 19, -19}},

{{-18, 33, -6},
{-31, -17, -31},
{-14, 9, 26}},

{{-24, 23, 23},
{-6, 32, 2},
{-9, -26, -1}},

{{13, 29, 19},
{35, 7, 6},
{32, -35, 30}},

{{-7, -7, 25},
{7, -41, 9},
{7, -12, 0}}},

{{{-12, -42, -25},
{38, 0, 15},
{21, 32, 21}},

{{-4, 7, 34},
{38, -5, 33},
{31, -35, 32}},

{{-12, 9, -5},
{2, 3, -4},
{17, -12, 3}},

{{5, 24, 41},
{14, -35, 25},
{-38, -6, 33}},

{{-34, -10, -1},
{-2, 27, 0},
{26, 10, 12}},

{{-32, -30, -18},
{13, -30, 32},
{24, -12, 5}},

{{9, 10, 4},
{-39, 41, 34},
{-37, 24, 22}},

{{-28, -23, 31},
{29, 34, -41},
{37, -16, -11}}},

{{{25, 14, 5},
{-14, -15, 16},
{-27, -18, 10}},

{{12, 11, -13},
{34, 14, 0},
{0, -8, 23}},

{{3, -26, -32},
{40, -23, 8},
{41, -35, 27}},

{{19, 37, -15},
{31, -30, 4},
{-32, -31, -11}},

{{-4, -14, 41},
{3, 0, -23},
{30, 24, 36}},

{{-40, 5, 22},
{8, 3, -20},
{4, 11, 7}},

{{6, -31, 33},
{39, -42, -32},
{-14, -29, 39}},

{{-3, 19, 6},
{0, -12, -19},
{36, 31, 9}}},

{{{-15, 38, 5},
{7, 39, 6},
{17, 36, 25}},

{{-20, 20, -5},
{-3, 2, -35},
{1, -5, -27}},

{{-9, 22, -28},
{-36, 11, -2},
{24, -29, -35}},

{{20, 6, -23},
{-26, -25, 18},
{42, 0, -29}},

{{7, 0, -20},
{26, -23, -32},
{31, 0, -9}},

{{-32, -19, -12},
{-40, -14, -2},
{-13, -5, 5}},

{{40, -29, 15},
{12, 27, 7},
{42, 30, -14}},

{{5, -13, -13},
{-25, 5, -26},
{8, 34, 6}}},

{{{-3, -15, 31},
{-14, 34, 9},
{-16, 31, 39}},

{{43, -32, 33},
{-5, 16, 36},
{-37, 25, -27}},

{{22, -34, 13},
{21, 6, -12},
{-15, 7, 3}},

{{7, 23, -2},
{38, 41, -20},
{25, 26, -17}},

{{-28, -15, -1},
{-39, -12, -30},
{5, -28, -26}},

{{-1, -6, 11},
{-24, -31, -22},
{12, -23, -4}},

{{38, -19, 33},
{-4, 5, 5},
{28, 17, -42}},

{{-22, -35, -12},
{16, 8, -9},
{35, 3, 13}}},

{{{-8, -10, 25},
{21, 36, -21},
{6, -2, 15}},

{{3, 7, -36},
{-6, 40, 24},
{26, 1, -4}},

{{25, 16, 16},
{-29, 15, 16},
{-11, 34, 24}},

{{6, 16, -29},
{23, -37, -26},
{-5, -41, -18}},

{{-11, 24, 8},
{-41, -11, 18},
{-16, 25, -12}},

{{14, -13, 11},
{-21, 29, -30},
{-27, 11, 26}},

{{-5, 1, 15},
{14, -17, 23},
{0, -23, -11}},

{{-13, -7, 23},
{-5, -12, -4},
{29, 32, 7}}},

{{{0, -22, 25},
{19, -25, 27},
{22, 19, 42}},

{{12, 35, 14},
{30, 17, 40},
{41, 0, 10}},

{{-8, -13, 37},
{-40, 0, 27},
{-13, -2, -2}},

{{13, 25, -25},
{23, 11, 7},
{41, -13, 31}},

{{2, 25, 30},
{-40, 30, 29},
{-26, 38, 34}},

{{39, -32, 34},
{31, -24, -5},
{2, 12, -12}},

{{-28, -10, -22},
{18, 28, 31},
{7, 37, -18}},

{{-28, 30, -14},
{-31, -20, 39},
{6, 0, 34}}},

{{{43, -34, -30},
{0, -18, -31},
{-2, 25, 15}},

{{4, -19, 8},
{12, -20, -10},
{-39, 22, 11}},

{{-35, -31, 30},
{41, -29, 26},
{3, 39, 23}},

{{36, 12, -6},
{37, -10, 28},
{-1, 0, -13}},

{{31, 41, 33},
{7, -28, -32},
{-15, 28, 13}},

{{-5, -27, -16},
{41, 22, 40},
{26, -37, -6}},

{{35, 18, -30},
{6, 9, 17},
{-25, 0, -41}},

{{-27, 5, -19},
{30, -27, 8},
{-10, 31, 31}}},

{{{-39, -29, -3},
{-24, -10, 29},
{-23, -13, -1}},

{{-37, 4, -25},
{-39, 40, 14},
{-14, 0, 19}},

{{-30, -33, -41},
{21, -13, 1},
{5, 15, -32}},

{{-25, 4, 22},
{39, 39, -27},
{15, -10, 25}},

{{0, 17, -5},
{35, 6, -18},
{-33, -9, -41}},

{{-10, 39, -22},
{-15, -15, -13},
{-11, -22, 11}},

{{-29, -1, -36},
{9, 8, -20},
{-29, -20, -19}},

{{-12, -24, -30},
{-35, 17, 20},
{-23, -29, -15}}}};

const uint16_t shape_Conv2D_2_b = 16;
const int16_t Conv2D_2_b[16] = {-1, -1, 1, 1, -1, -1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0};
