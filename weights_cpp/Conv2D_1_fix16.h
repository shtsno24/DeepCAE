/*
 * author : shtsno24
 * Date : 2019-10-18 21:42:43.464432
 * array_type : int16
 * fractal_width : 14 bit
 * bit_width : 16 bit
 *
 */
#pragma once
#include <cstdint>
#include <vector>

using namespace std;

#define data_width_Conv2D_1 16
#define fractal_width_Conv2D_1 14

const vector< uint16_t> shape_Conv2D_1_w = {8, 16, 3, 3};
const vector< vector< vector< vector< int16_t> > > > Conv2D_1_w =
{{{{  -25,  1010,  -607},
{ 2277, -1566,   871},
{ 1861,  1843,  2009}},

{{-2044, -2003, -2507},
{ -339, -1644,  1857},
{ -483, -1714,  1346}},

{{-1257, -2647, -1764},
{ -816,  -578,   915},
{  493,  2399,   302}},

{{ 2892,  1897, -2492},
{ -992, -1973,  -410},
{-1471,  1780, -2739}},

{{  391,   346,  1809},
{ 2523,   -52, -2073},
{-2159,  2218,  1360}},

{{-1750, -2629, -1767},
{-1835,   486, -2064},
{ -109,   491,  1528}},

{{ 2182, -2706,  1750},
{-1455,  1989, -2025},
{-2609,  1178,    11}},

{{  646,  2023,  -666},
{ -706, -1247,   584},
{ 2084, -1688, -1971}},

{{-1690,  1964,  1634},
{ 2401,   552,  -104},
{ 1501,   762,  -435}},

{{ -177,  1380,  -482},
{ 2257,   731,  1832},
{  658,   996,  2126}},

{{ -440,   -98,   925},
{ 1050,   196, -1268},
{   75,  1491,  2753}},

{{  741,   709, -1665},
{ 1514, -1390,  1643},
{  754,  1921,  2183}},

{{-1431,  1530, -2564},
{-1300,   986,  2539},
{ 2658,   -77,    49}},

{{ 1586, -2625,  1076},
{ 1687, -2681,   -71},
{ 1118,  1839,  1715}},

{{ -777,  2220,  -649},
{ -720, -2184, -2341},
{ 2101, -1315,  -891}},

{{ 2650,  -262,    48},
{  -93,  -640,   322},
{-1377,   227, -1389}}},

{{{-1871,   222,  2736},
{ 1439, -1054, -2125},
{ 2468,  2217, -2218}},

{{-1817, -1747,   136},
{ 1544,   724,  2148},
{-1458,  1344, -1133}},

{{ 2214, -1837,  1441},
{  793,  2879,  -209},
{  341,   216,  2905}},

{{  947,  1754,  2183},
{ -798,  2182,   919},
{ -869,  2504,  2300}},

{{ -851,   -50,  -834},
{ 1692, -1422,   369},
{ 3539,  -143,  1409}},

{{-1216,  1057, -1468},
{-2681,   379,  1816},
{   83,    18, -1122}},

{{  496,  1700,   916},
{  375, -2562,  -222},
{-1341, -2653,   -36}},

{{ -794,  1802,  1665},
{  575,   426,  -781},
{ 3139,  1298,  1573}},

{{-2477,  2209, -2408},
{ -371,  -802,  -731},
{  498,  3109, -1858}},

{{-2559, -2538,  -599},
{ 1902, -2156,   228},
{-2364,   433,  -349}},

{{ -281,   240,   610},
{  851,    39,  2080},
{ 2337,  1404,  1419}},

{{-2352, -1867, -2293},
{  453,  1181,   487},
{ -634,  3134,  2390}},

{{ -173, -1895, -2553},
{-1295,  -774, -1779},
{ -134,   909, -1977}},

{{-1645, -1159, -1628},
{ -103, -2149, -2637},
{  788,   877,   430}},

{{ 2029,   806, -1256},
{ 2084, -1329, -1964},
{ 2744,  1221,  2738}},

{{-1451,  1106, -1480},
{-2114,  -426, -2244},
{ 2005,   517,   166}}},

{{{  751,  2131,  -916},
{  873, -1694, -2390},
{ 2184, -1359,  1852}},

{{  574,  1301,   326},
{ 2581, -1852,   -87},
{ 1974,   698,   400}},

{{ -727, -1375,   324},
{-1914,  1128, -2038},
{  533, -2194, -1630}},

{{-1505,  1510, -1139},
{  203, -1207, -2290},
{ 1868,  1786, -1005}},

{{ 1413,  -395,  -247},
{-2753,  -850,  1261},
{ 1305,  2454,  2420}},

{{ 1776, -1002,   998},
{ -174,   490,  1808},
{ 1854,  -889,  -515}},

{{ 1180,  1696, -2604},
{   97,  -586,  -435},
{-2668,   355,  -310}},

{{-2573,  2191,  -376},
{-1801,  -405,  2025},
{  197,  2099,  -862}},

{{ -289,  1905, -2666},
{ -847,  1344,   355},
{-2078,   820,   913}},

{{  536, -2690,  -168},
{ 2336, -2707,  1996},
{-2566,  2111,   262}},

{{ 2170, -1611,   494},
{ -244,  1355,  -983},
{  -98,  2101,  1452}},

{{-2708,  -194,   502},
{-1849, -1274,  -541},
{ -184,  2414, -2310}},

{{ 2415, -2472,  -812},
{-1477,  1372,   639},
{-2051,  -115, -1115}},

{{ -277,   930,   -49},
{  641, -2113,  -402},
{ -355,   721, -2258}},

{{  613,  1005,   544},
{  680,  1212,  -803},
{ 1503,  2457,   844}},

{{-1925,   373, -1865},
{  662,   127, -2483},
{ 2586,   328,   775}}},

{{{ -346, -1357, -1863},
{ 1634,  2394, -2582},
{ 2496, -2087,   707}},

{{ -280,   255, -2298},
{-2260, -1412, -1344},
{ -109,  -749, -1710}},

{{-2228,  1579,   736},
{ -854, -2582, -1847},
{-2394,  2310, -2075}},

{{-2362, -2187, -1737},
{-2308,  1004,   531},
{  232,  1384, -2307}},

{{  789,  1400,  2014},
{ -408, -1604,  2337},
{ 1852, -1595,  1199}},

{{-1851, -1374,  1278},
{ 1007,  1912,    77},
{  245,   491,   742}},

{{ 2080, -1233, -1828},
{-2131, -2003, -1678},
{ 2344, -2253, -2438}},

{{-1223,  2792,  2182},
{ 2787,   825,  -770},
{-1877, -1365, -2406}},

{{ 2472, -1073, -2444},
{-1674,  -963,  1051},
{ -886,    63,  2357}},

{{  270, -2385,  1644},
{-1732, -1814,   358},
{ 1447,   889, -1933}},

{{  871,  1547, -1282},
{-1955,  1152, -1747},
{ 1446,  2397, -1224}},

{{  662,  1680, -2437},
{ -764, -1480,  2568},
{ -439,  1455,   643}},

{{ 2044,  1042, -1075},
{  349,   992,  1789},
{  765,   916, -1756}},

{{ -743, -1190,  1729},
{  878,  2469, -2369},
{-2625,   440, -1364}},

{{-1879,  2646,  -289},
{ 1360,  1789,   160},
{ 1216,  -453,  2231}},

{{-2459,  2115, -2220},
{-1653, -1028,  -977},
{  357, -1876,  2395}}},

{{{-3051,  -216, -1552},
{ -366, -1001,   733},
{-1515,  2408,   -99}},

{{ 1504,   946, -1902},
{ -233,  2299, -1504},
{ -935, -1765,  -170}},

{{  848, -1249,   124},
{ 1259, -1953,  -813},
{-1039, -2402,  -591}},

{{ 2085, -1292,   382},
{-2287,  2639,  -593},
{-1773,  1282, -2504}},

{{ 1418,    19, -1430},
{-1516,  1238,  -461},
{ 2392, -2526,  1801}},

{{ 1563, -2190,   702},
{ -580,  2717,  -808},
{ 1675, -1833, -2025}},

{{ 1489,   994,  1988},
{-1450, -1692,  -809},
{  -78, -1343,   378}},

{{-2287, -1246,  2047},
{ -491,  1167,  2601},
{ -430,  1531,  1091}},

{{ 1018,  -808,  1704},
{ 2051, -1101,  -786},
{-1684, -2177,  2583}},

{{-1727, -2330,  1814},
{ 1473,  1994, -2304},
{ -753,  -451,  -379}},

{{ 1154,  2212,  1269},
{ 1197,  -149, -2627},
{ -655, -1283, -1059}},

{{ -446, -1226,   840},
{ 2187,   845, -1839},
{ 1506, -2642, -1936}},

{{-1731,   179,  -808},
{  405,  2494, -2068},
{-2439,  -850,  -796}},

{{  344,  1236, -2294},
{ -310,  1971, -1157},
{-2402,  -348,  -132}},

{{ 1177,  1599, -1364},
{ -210,  1492,  -964},
{  477, -2279,    -2}},

{{-1944, -2526,   992},
{-1547, -2108,    49},
{ -942,  1051, -2711}}},

{{{ 2404,   382,  -982},
{ 2372,    39,  -362},
{-2284, -2927,  1525}},

{{  617,  2678,   756},
{  422, -2357,  2284},
{-2679,  2840,  1019}},

{{  299, -1013,  2375},
{  778,  -514,  -968},
{ -290,   367, -1705}},

{{-2140,  1264, -1669},
{-1325,  1465,  -638},
{ 1683,   169,  1896}},

{{  880, -1708,  2685},
{-1752,   661, -1588},
{ 1568,  -223,  1464}},

{{ 1586, -1386,  1999},
{-1819,  2337,   385},
{  423,  -594,  1617}},

{{  874, -2480,  1884},
{-1047,  -290,  2571},
{ 2397,  -142,  2212}},

{{  454,  -471,   229},
{-2446,  2203, -1134},
{    8,  1155,   504}},

{{ -422, -2089,  2840},
{ 1247,  2116,  2641},
{-1694,   579,  2852}},

{{ -206,  -653,  1070},
{ 1589,  -187,  1578},
{-1849, -1803,  2271}},

{{  666,  -498,   856},
{ 1245,   -47,  -157},
{-1980,   472, -1868}},

{{-1107, -2764,  2106},
{ 1104,  1677, -1643},
{-2049, -1419,  2829}},

{{ 2529,   341,   260},
{-1560,  1337,   803},
{ -469, -1325, -2533}},

{{ 2659,  1436,   439},
{ -442,  -957,  -471},
{-1295,  1944, -1638}},

{{  496,  2323,  2094},
{  857,  -785, -1372},
{-2542,   791,  -835}},

{{-2735,  1096,  2730},
{ 2395, -1174, -1984},
{ 1081,  -875, -2502}}},

{{{  394,  1500,  -804},
{  928,  -635,  1040},
{ 1363, -2988,  1205}},

{{-1587,  1285,  1106},
{-1463,   269,  2258},
{ -188,  2243, -2440}},

{{ -589, -2221,   -24},
{-1500, -2415, -1178},
{ 1700,   894,  1743}},

{{ 1396,  2826,     1},
{ 1719,   388,   907},
{ 1077, -2250,   697}},

{{ -251, -2046,  -502},
{-2328,   118, -2623},
{  761,    22,  1167}},

{{-1296,   570,    34},
{ 1842, -2575,  1621},
{ -370, -1542,  -121}},

{{ 1451,  2148,  -731},
{-1806, -2580, -2633},
{ 2448,   915,   787}},

{{  184,   711,  -230},
{ 2130, -1232,  -104},
{-1714,  1944,   630}},

{{-2526,   408,   736},
{ 1899, -1969, -1748},
{ -545, -1470, -1396}},

{{ -640,   334,  1733},
{ 1670, -1578,  1377},
{ 1788, -2061,  -465}},

{{ -145,  1735,  2012},
{ 1817,  2464, -2678},
{ 1750, -2916,  1582}},

{{-1462, -2291, -1722},
{-2417, -2300, -2677},
{ -496,   408, -2461}},

{{ 1547, -2638, -1751},
{-1020, -1252,  1489},
{-2208, -2739, -1517}},

{{-1864,  1753,   306},
{ 2045, -2416,  -744},
{ -691,  2051,    70}},

{{  650, -2386, -1056},
{-1299, -2131, -1433},
{ 2148,  1845,   578}},

{{  503, -1676,  1405},
{ 1108, -1861, -1952},
{-1873,  -888, -1806}}},

{{{  392,   379,  2029},
{-2568,  2166, -1054},
{ 1077, -1574,  1939}},

{{ 1779, -2148,   941},
{-1539, -1803,  -509},
{-1948,   459,  1470}},

{{ -466, -2261, -1793},
{-2273,  -230,  1007},
{ 2330, -1035, -1286}},

{{ 1191, -2011, -2910},
{  751,   596,  3101},
{ 2422,  2982,  -255}},

{{-1317,   344, -2746},
{  995,  2356,  -269},
{  545, -1379,  2325}},

{{-2648, -1225,  2402},
{ -475,  -474,   126},
{ 2536,   667, -1030}},

{{ 1500,  1112, -1214},
{ -353,   590,  -277},
{ 1833, -1222,  -648}},

{{ 1945,  1904, -1019},
{ 1842,  3543,  1130},
{   33,  1484, -1860}},

{{-1031,  2102,    32},
{ -898,   872, -1678},
{-2695,  -674,  -587}},

{{ -519, -1200,  -576},
{ 2045, -1199,  2020},
{ 1118,   605,  1044}},

{{ -502, -1862,  2098},
{  986,  -740,  2481},
{ 1480, -1551,  1435}},

{{-2344, -1357,    73},
{  544,  2634,   656},
{-1657,  1874,  2287}},

{{  139, -1710,  -344},
{-2422,  1602,  -354},
{-1921,  2137,   -39}},

{{-1107,  1593,   787},
{ 1515,  1936,   543},
{-1405, -1811, -1338}},

{{  -85,  2663,  2362},
{-1575,  -212, -1343},
{-1560,  1750,  2703}},

{{ -842, -1459,   451},
{-1791,  1320,  2205},
{  903, -1769, -1743}}}};

const uint16_t shape_Conv2D_1_b = 8;
const vector< int16_t> Conv2D_1_b = {  111,   -33,   854,   177,  -853,   232,  1063,  -256};
