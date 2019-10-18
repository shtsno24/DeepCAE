/*
 * author : shtsno24
 * Date : 2019-10-18 23:26:07.527662
 * array_type : int16
 * fractal_width : 14 bit
 * bit_width : 16 bit
 *
 */
#pragma once
#include <cstdint>
#include <vector>

using namespace std;

#define data_width_Conv2D_3 16
#define fractal_width_Conv2D_3 14

const vector< uint16_t> shape_Conv2D_3_w = {16, 8, 3, 3};
const vector< vector< vector< vector< int16_t> > > > Conv2D_3_w =
{{{{ 2034, -2507,   386},
{  952,  -446,  1162},
{-1554,   275,  2390}},

{{  270,   838,  1596},
{  675,   966,  1976},
{ -930,  1122,  1377}},

{{-1305,  -176,  2065},
{-2018,   856,  1088},
{-1049,  1800,   882}},

{{-1531,  1461,   532},
{-1377,  1320, -1824},
{   99,  1057,  2489}},

{{ 1678,   711,  2860},
{-1353,  1845,  1515},
{  528,   837,   715}},

{{-1465,  2335,  -844},
{ 3178,  1379,   943},
{-1747, -2539,  1079}},

{{-1110,  1242, -2322},
{ 2318,  1376, -1830},
{  859, -1619,   106}},

{{  930, -1865, -1060},
{ 1607,  -849,  1241},
{  524,  -957,   201}}},

{{{-1923,  -661, -2223},
{ 1556, -2516,  -710},
{ 1478,  1777,  -432}},

{{ 1614, -2718,  1859},
{  128,  1448,   874},
{ 2147,   556,   142}},

{{ -876,  2485, -2682},
{ 2182,   391, -2499},
{  239, -2625,  1200}},

{{ 1161, -2662,   758},
{ 1138, -2279, -2606},
{ 1769,  1527,   183}},

{{ 1644,  1062, -2713},
{   43,  2264, -1508},
{ 2497,  2384,   731}},

{{ 2296,   113,  2292},
{-1614,   -58, -1559},
{ 3041,   691,   330}},

{{-2084,   416, -1419},
{ -667, -1761,   346},
{-1380,   694,  1551}},

{{-1530,   588,  1930},
{ 1180,  -110, -1490},
{  153,  1497,  2009}}},

{{{-1212,   344, -2447},
{ 1552, -1844,  1382},
{ 1276,    16,  -519}},

{{-1568,   579,   190},
{-1000,  1373,  2223},
{-2276,   888,  2312}},

{{-1526,  1689, -1954},
{ 1315,  1863, -1779},
{ 1443, -2569, -1975}},

{{-1813,   946, -1343},
{ -602,  2335,  -613},
{-2447,  -916, -2653}},

{{-2607,  -522,  1911},
{ 1489, -1236, -1213},
{ 1511, -2015, -1643}},

{{ 2460,   -27, -1392},
{-2526,   466, -1540},
{  431,  2332,   605}},

{{-1556,  1230, -2096},
{ -361, -2365,  -337},
{-2011,  1036,   -28}},

{{ -372,   -47, -1070},
{-1363, -1395, -2216},
{ -257,  1822,    32}}},

{{{-1658,  1575,   716},
{-2612, -2404,   529},
{-1441,  -791, -1158}},

{{-1317,  1099,  1694},
{-2546, -1467,  1479},
{ 1384, -1921,    10}},

{{  475,  1653,   939},
{ 2119,  1857,  -286},
{ 1272,  1742, -1974}},

{{ -252,  1380, -1828},
{ 1495, -2099,  -386},
{ 2383,  -579, -2418}},

{{-2412,  1289,  -321},
{ 1054, -1262,   489},
{ -903,  2238,  1755}},

{{ 1060,   417,   295},
{ -497,  1942, -1431},
{ -887, -1784, -1631}},

{{ 1753,   451,  -410},
{-2509,  2500,  -542},
{-1902,   -98, -2235}},

{{-1332, -2574, -1197},
{-2488,  -818,  2657},
{ 2229,  1249, -1068}}},

{{{-2286,  1115,  1465},
{ -686,  1724,  1791},
{-1069,  1454,  2547}},

{{  753,   958,  -893},
{-2653,   782,   920},
{-1056,  -406,   323}},

{{ -492,   -14, -1188},
{ 2495,  2824,  -327},
{ -901, -1263,  2091}},

{{  329,   782,  2141},
{-1387,   506,  2131},
{-2519, -1932, -1653}},

{{-2144,   661,   590},
{-1424,   443,  -572},
{ 1464,  1885,   276}},

{{ -210,   247,  2200},
{ -947,   762,  2738},
{-1295, -1150, -1307}},

{{ 1654, -2030,  2834},
{ -985,  1395,  -354},
{  442,  2421,  1180}},

{{ 2106,   580,   915},
{ -962,  1939,  -747},
{-1000,  1281,  2387}}},

{{{ -921,  -928,  -520},
{ 2151,  -274,   -65},
{ 2154,   956,  1295}},

{{ 1910,  1308,  -409},
{-1551,  -787,   999},
{ -576,  2106, -2571}},

{{-1348,  2130,  2439},
{ 1174,  2695,  2492},
{-1137, -1821,  1514}},

{{ 1567,  -132,   252},
{  103, -2499, -1364},
{-2094,  -745,   230}},

{{-1581,  -752,   339},
{ -739, -1223, -2718},
{  386,  -495,   730}},

{{ 1831, -1668,  1844},
{ -852,  1415, -1186},
{  399, -2262,  1767}},

{{ 1366,  -448, -1179},
{-2205, -2971,  -149},
{ 2388,  1845, -1334}},

{{ 2244,  -892, -2486},
{ 2431,  2662, -2329},
{-1418,  -821,  -925}}},

{{{ 1684, -2338,  1311},
{  326, -1057,  2428},
{-1655, -1903, -2380}},

{{-1846,  1634,  1334},
{ 1481, -1800,   249},
{-1943,     0, -1650}},

{{-3421,  1236,  1692},
{ -476,  2558,  2075},
{ 1695, -1149,  -451}},

{{-1932, -1264,  1107},
{  803, -2691,  2547},
{ -950,   785,  1104}},

{{   -6,   941, -1824},
{ 1815,   805,  -704},
{ -735,  2465, -2196}},

{{-2115,  2385,  1531},
{ 2277,   152,  1436},
{ 1241, -1900, -1092}},

{{-1400,  1173, -2500},
{ 1743,  1860,  1647},
{ 1457,  1339, -2853}},

{{  218,  1943,   862},
{ 1369, -1818,  2211},
{ 1796,  -773,  1720}}},

{{{-1985, -1658,   998},
{  987,  -395,   129},
{ 1454, -1191,   313}},

{{ 1662, -2030,  1063},
{ -328,  2455,    36},
{-2690, -1199,  1623}},

{{-2307,   793, -2308},
{ 1046, -1091,  -201},
{ 2739,  1894,   659}},

{{ -336,   713,   893},
{ 1882, -1945, -3101},
{-2309,   531,  2065}},

{{-1559, -1212,  1913},
{ 1166, -1914, -2340},
{ 1037,  1183,  1412}},

{{  330,  1673,  1446},
{ 1911,  1407, -1122},
{ 3204,   339, -1373}},

{{  861, -2544,   354},
{   53, -1466, -1343},
{-2056,  1165,  -969}},

{{ 1076,   563,  2398},
{  740,  2134, -1452},
{ 2914,   589,   591}}},

{{{-1706,  -168,  -290},
{ 2057,  2061,  1654},
{-1266,  1071,  -782}},

{{ 2504, -1882,  -986},
{ 2066,  1317,   559},
{ 1042,  -352,  -210}},

{{ -990,   738,  -689},
{-1484,  2620,  1928},
{ 2142,  1164,  1461}},

{{-1568,  2146, -1479},
{  -59,  -590,  1085},
{-1570,    50, -2379}},

{{   64,  2640,   710},
{ 2181, -1413, -2255},
{ 1106,  -750,   614}},

{{  601,  2092,  -214},
{ 2835,   732,  -960},
{  586,  2010,  1601}},

{{ 2477,   903, -2706},
{ 2506, -1444,   870},
{ 2347,  -961,  2210}},

{{ -956,  -644, -1943},
{  549,  2519, -2421},
{ 1436,  1719, -1954}}},

{{{ 1498,  1237,  -154},
{  987,   212,   539},
{ -362,  -833, -1040}},

{{-1171, -1257, -2678},
{ -639, -1932,  -239},
{ 1666, -2347,  1818}},

{{ 1871, -1353,   826},
{ 2718,  2172,  3161},
{-1696,  2880, -2323}},

{{ 1184, -1843,  1354},
{-2548,   886,   331},
{ -612,  1854,   791}},

{{ 2699,   244, -2318},
{ 1139, -1277,  2257},
{ -817,  -872,  -787}},

{{  480, -1876, -1122},
{ 1546, -1678,  1509},
{ 1494, -2015,   600}},

{{-2296, -2822,   556},
{ 1985,  2905, -1499},
{ 2764,  2144,  1107}},

{{-2267,  2450, -2634},
{  251,  1719,  1436},
{ -668, -1609,  -458}}},

{{{ -494,  1006,  1974},
{  303,  1385, -1759},
{-2201, -1601,  2282}},

{{  -14,  2253, -1396},
{-2711, -1075,  -788},
{-1980,  1339, -1590}},

{{ -254, -2183,  2439},
{ -801,  1158, -1738},
{  831,  2167,  -186}},

{{-2180,  2394, -1112},
{   54,  2112, -1069},
{ -562,   -89,  -167}},

{{  280, -1970, -2545},
{ 2024,   698,  1375},
{ 1876, -2189, -1932}},

{{ -911,   233, -2886},
{  387,  -767,   486},
{ 2594,  2445, -1811}},

{{ 1643, -1783,   931},
{ 2080,  -332,  -595},
{  386,  1528,  -229}},

{{ 1665,  1571,   773},
{ -685,  1676,   759},
{-1401,  1086,  -749}}},

{{{-1805,  2489,   734},
{  -82,  1958,  1900},
{ 1496,   522,  1247}},

{{-2531, -2343, -1874},
{-1846,  -425, -1436},
{-1672,  2106,  1829}},

{{-2285,  2100,  2362},
{  458,  -201, -1357},
{  263,   941,   921}},

{{-1561,  2077, -2820},
{  701, -2466,  1507},
{ -549, -2239, -1113}},

{{ -600,   194,   394},
{-2657, -2006,  2277},
{-2946,  1897,   791}},

{{ -428, -1931, -1924},
{  219,  -296,  2897},
{ 2560,  2072,  2134}},

{{ -367,  2150, -1487},
{  257,  1387, -1692},
{ 2259, -2126,  1514}},

{{ -535,  -146,   233},
{  624,  2390,   760},
{-2307,  1917,  1675}}},

{{{  107,  -412,  -902},
{  664, -1670, -1335},
{ 2750, -1672, -1843}},

{{-2243, -1921,  1411},
{-1288, -2235,  -171},
{ -555,  -504,  1513}},

{{ 1261, -1798,   966},
{ 1039,   167,  -565},
{ 1791,  -928,  2538}},

{{  996,   653,   -31},
{ 1185,  2810, -1836},
{-1213,  2538,   365}},

{{  812,   489,  2100},
{  713, -1252,  -453},
{  913,  2027, -2619}},

{{-2423, -1342,  1794},
{-1854,  -117, -2587},
{-2394,  1043,  -344}},

{{-2033, -1886,   585},
{ 2490,  1285,   127},
{  981,  2258,  1481}},

{{-2134, -1795,    48},
{ -988,  2215, -2335},
{  906,  -293,  2650}}},

{{{ -141,  2179, -1464},
{  216,  1800,   282},
{ 2557,  2145,  -286}},

{{ -387,  2188,  -451},
{-2504,  1999,  2074},
{ 2595,  1674, -1532}},

{{ 1538,  1487, -1222},
{ 2840,  1386,  -741},
{-1080,  -430,  -245}},

{{-1998,  -728,  2761},
{-2316,  -398,  2696},
{-2363,  1435, -2129}},

{{ -655,   421, -1695},
{-1281,  1941,   534},
{-1390,  2588, -1280}},

{{ -461, -1115,  -847},
{-1363, -1088, -1575},
{-2492, -1932, -1303}},

{{-1058,  2577,  2305},
{ -755,  1510,  -620},
{ -829,  2252, -1037}},

{{  891,  1522, -1659},
{  893,  -163,   447},
{-1999,   911, -2408}}},

{{{  129, -1892,  2857},
{  -26, -1952,  1056},
{ 1145,  1636, -2559}},

{{  329, -2720,  1325},
{-1936,  -270,  1045},
{-1232,   203,  1317}},

{{ 1607,   196,  1781},
{  468,  -429,  1717},
{-2145,   244, -2005}},

{{-1971,  2541,   -20},
{ -265,   191, -1833},
{-1889,   588,  -878}},

{{ 1593,  1912,  -918},
{ -590,  2335,  1198},
{ 1403,  1821, -1754}},

{{ 2910,  2018,   401},
{-2704,  1058, -1084},
{ -836,  2185,  2351}},

{{ 1000,  2684,   711},
{ 2190, -1525,   332},
{-1602, -2418,  -326}},

{{  985, -2494,  2391},
{ -813,   862,  1635},
{   57,  1888, -1292}}},

{{{ 2921, -1100, -1397},
{   18, -1467,  1783},
{  349,  1809,  1761}},

{{-1832,  -712,  -853},
{ -478, -2505,    66},
{-1125, -1860,  1908}},

{{-2148,  -522, -1053},
{ 1965,   992, -2066},
{-1563,  1386,  2421}},

{{  656,  1988,  2724},
{ -744,  -612,   787},
{ 2581,  3380,  -919}},

{{ 1931, -2735, -1005},
{  417,   903, -1021},
{ -270,  -298, -2571}},

{{-2049, -1927,  -895},
{-2612,  1155,   330},
{ -914,  1571, -1051}},

{{  916,   394,    98},
{ 2338, -1975,  2968},
{-2461, -2558, -1020}},

{{ -990, -1036,  2260},
{-1681,  -810,  1937},
{ -404,  -319,  -361}}}};

const uint16_t shape_Conv2D_3_b = 16;
const vector< int16_t> Conv2D_3_b = { -921,   739,     1,  -301,  -111,  -404, -1034,   729, -1052, -1045,   896,   339,   987,   294,  1203,  1578};
