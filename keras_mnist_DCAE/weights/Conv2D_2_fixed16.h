/*
 * author : shtsno24
 * Date : 2019-10-15 13:43:38.423135
 * array_type : int16
 * fractal_width : 14 bit
 * bit_width : 16 bit
 *
 */
#pragma once
#include <stdint.h>

#define data_width_Conv2D_2 16
#define fractal_width_Conv2D_2 14

const uint16_t shape_Conv2D_2_w[] = {16, 8, 3, 3};
const int16_t Conv2D_2_w[16][8][3][3] =
{{{{2561.00000000000000000000, 2111.00000000000000000000, -1171.00000000000000000000},
{800.00000000000000000000, -1658.00000000000000000000, 1523.00000000000000000000},
{365.00000000000000000000, 1024.00000000000000000000, 1286.00000000000000000000}},

{{1282.00000000000000000000, 451.00000000000000000000, 661.00000000000000000000},
{2623.00000000000000000000, -575.00000000000000000000, 2235.00000000000000000000},
{1171.00000000000000000000, -585.00000000000000000000, -1705.00000000000000000000}},

{{1030.00000000000000000000, 38.00000000000000000000, 1074.00000000000000000000},
{-2478.00000000000000000000, -1677.00000000000000000000, -71.00000000000000000000},
{-185.00000000000000000000, -2651.00000000000000000000, 1465.00000000000000000000}},

{{-929.00000000000000000000, -2548.00000000000000000000, 1347.00000000000000000000},
{1345.00000000000000000000, 1652.00000000000000000000, 1927.00000000000000000000},
{-1110.00000000000000000000, 2209.00000000000000000000, -253.00000000000000000000}},

{{678.00000000000000000000, 2025.00000000000000000000, -409.00000000000000000000},
{-241.00000000000000000000, 2510.00000000000000000000, 923.00000000000000000000},
{-1124.00000000000000000000, 2102.00000000000000000000, 148.00000000000000000000}},

{{-89.00000000000000000000, 1050.00000000000000000000, 2379.00000000000000000000},
{-595.00000000000000000000, -148.00000000000000000000, 802.00000000000000000000},
{-481.00000000000000000000, -1617.00000000000000000000, -1851.00000000000000000000}},

{{-2313.00000000000000000000, 232.00000000000000000000, -1939.00000000000000000000},
{-818.00000000000000000000, 431.00000000000000000000, 2228.00000000000000000000},
{2734.00000000000000000000, -885.00000000000000000000, -1756.00000000000000000000}},

{{1725.00000000000000000000, -1994.00000000000000000000, 1553.00000000000000000000},
{-131.00000000000000000000, -1203.00000000000000000000, -1529.00000000000000000000},
{1736.00000000000000000000, 500.00000000000000000000, 354.00000000000000000000}}},

{{{-10.00000000000000000000, -2685.00000000000000000000, 1357.00000000000000000000},
{-853.00000000000000000000, -655.00000000000000000000, 1345.00000000000000000000},
{1176.00000000000000000000, -1722.00000000000000000000, -2159.00000000000000000000}},

{{1348.00000000000000000000, 92.00000000000000000000, 1557.00000000000000000000},
{1535.00000000000000000000, -1887.00000000000000000000, -2416.00000000000000000000},
{-1724.00000000000000000000, -942.00000000000000000000, -445.00000000000000000000}},

{{1509.00000000000000000000, -278.00000000000000000000, -711.00000000000000000000},
{-910.00000000000000000000, 2729.00000000000000000000, 780.00000000000000000000},
{-1158.00000000000000000000, 2251.00000000000000000000, 1701.00000000000000000000}},

{{375.00000000000000000000, -2251.00000000000000000000, 1678.00000000000000000000},
{1275.00000000000000000000, 1370.00000000000000000000, 2630.00000000000000000000},
{-1816.00000000000000000000, -2234.00000000000000000000, -899.00000000000000000000}},

{{392.00000000000000000000, -210.00000000000000000000, -187.00000000000000000000},
{1340.00000000000000000000, 400.00000000000000000000, 1808.00000000000000000000},
{-1065.00000000000000000000, 900.00000000000000000000, 1236.00000000000000000000}},

{{1737.00000000000000000000, -1130.00000000000000000000, 865.00000000000000000000},
{1913.00000000000000000000, -2553.00000000000000000000, -2628.00000000000000000000},
{807.00000000000000000000, 432.00000000000000000000, -617.00000000000000000000}},

{{1547.00000000000000000000, -411.00000000000000000000, -1802.00000000000000000000},
{-220.00000000000000000000, 641.00000000000000000000, 1183.00000000000000000000},
{-906.00000000000000000000, -2258.00000000000000000000, -1145.00000000000000000000}},

{{2360.00000000000000000000, 2436.00000000000000000000, 2262.00000000000000000000},
{1659.00000000000000000000, -1602.00000000000000000000, -817.00000000000000000000},
{501.00000000000000000000, 1728.00000000000000000000, 2406.00000000000000000000}}},

{{{456.00000000000000000000, 134.00000000000000000000, 1185.00000000000000000000},
{2188.00000000000000000000, -2652.00000000000000000000, -2400.00000000000000000000},
{276.00000000000000000000, 649.00000000000000000000, 2615.00000000000000000000}},

{{177.00000000000000000000, -2541.00000000000000000000, 553.00000000000000000000},
{2145.00000000000000000000, -1553.00000000000000000000, 656.00000000000000000000},
{-55.00000000000000000000, -1477.00000000000000000000, 729.00000000000000000000}},

{{15.00000000000000000000, 2161.00000000000000000000, 1508.00000000000000000000},
{237.00000000000000000000, 634.00000000000000000000, 1757.00000000000000000000},
{2337.00000000000000000000, -2434.00000000000000000000, -2406.00000000000000000000}},

{{2099.00000000000000000000, -938.00000000000000000000, 35.00000000000000000000},
{-1386.00000000000000000000, -2077.00000000000000000000, -1004.00000000000000000000},
{-177.00000000000000000000, -776.00000000000000000000, -1450.00000000000000000000}},

{{-1053.00000000000000000000, -2671.00000000000000000000, -381.00000000000000000000},
{-106.00000000000000000000, -1270.00000000000000000000, -821.00000000000000000000},
{83.00000000000000000000, -1762.00000000000000000000, -779.00000000000000000000}},

{{2482.00000000000000000000, -1344.00000000000000000000, 2635.00000000000000000000},
{-1257.00000000000000000000, -1365.00000000000000000000, 2709.00000000000000000000},
{-1497.00000000000000000000, -817.00000000000000000000, 2246.00000000000000000000}},

{{-188.00000000000000000000, -1521.00000000000000000000, 944.00000000000000000000},
{-2671.00000000000000000000, 1048.00000000000000000000, -948.00000000000000000000},
{865.00000000000000000000, 1943.00000000000000000000, 199.00000000000000000000}},

{{-621.00000000000000000000, -2469.00000000000000000000, 2356.00000000000000000000},
{-1131.00000000000000000000, 1853.00000000000000000000, 687.00000000000000000000},
{-414.00000000000000000000, -343.00000000000000000000, -2114.00000000000000000000}}},

{{{1042.00000000000000000000, 686.00000000000000000000, -1800.00000000000000000000},
{-416.00000000000000000000, -791.00000000000000000000, 2238.00000000000000000000},
{-2583.00000000000000000000, -473.00000000000000000000, 996.00000000000000000000}},

{{-678.00000000000000000000, 2483.00000000000000000000, 1022.00000000000000000000},
{2479.00000000000000000000, -800.00000000000000000000, 93.00000000000000000000},
{-1710.00000000000000000000, 678.00000000000000000000, -1753.00000000000000000000}},

{{2169.00000000000000000000, -1351.00000000000000000000, 1898.00000000000000000000},
{-1739.00000000000000000000, 835.00000000000000000000, 1014.00000000000000000000},
{1781.00000000000000000000, 1371.00000000000000000000, -2364.00000000000000000000}},

{{2355.00000000000000000000, 1861.00000000000000000000, 1346.00000000000000000000},
{-1790.00000000000000000000, 210.00000000000000000000, -743.00000000000000000000},
{-118.00000000000000000000, 2077.00000000000000000000, 2053.00000000000000000000}},

{{2232.00000000000000000000, 1260.00000000000000000000, -527.00000000000000000000},
{-487.00000000000000000000, -2511.00000000000000000000, 1106.00000000000000000000},
{753.00000000000000000000, -2334.00000000000000000000, -116.00000000000000000000}},

{{2413.00000000000000000000, -2010.00000000000000000000, -951.00000000000000000000},
{-1173.00000000000000000000, -1082.00000000000000000000, 860.00000000000000000000},
{-901.00000000000000000000, 659.00000000000000000000, 1190.00000000000000000000}},

{{1444.00000000000000000000, 999.00000000000000000000, -2283.00000000000000000000},
{1109.00000000000000000000, 2034.00000000000000000000, -109.00000000000000000000},
{-1344.00000000000000000000, -2700.00000000000000000000, -1185.00000000000000000000}},

{{-227.00000000000000000000, 1298.00000000000000000000, 626.00000000000000000000},
{-1143.00000000000000000000, 2195.00000000000000000000, 2212.00000000000000000000},
{2230.00000000000000000000, -1962.00000000000000000000, 1467.00000000000000000000}}},

{{{-1998.00000000000000000000, 959.00000000000000000000, 2142.00000000000000000000},
{2160.00000000000000000000, 1491.00000000000000000000, 722.00000000000000000000},
{-2483.00000000000000000000, -2455.00000000000000000000, 893.00000000000000000000}},

{{200.00000000000000000000, 2543.00000000000000000000, -515.00000000000000000000},
{930.00000000000000000000, -2063.00000000000000000000, -633.00000000000000000000},
{-830.00000000000000000000, 2155.00000000000000000000, 948.00000000000000000000}},

{{-923.00000000000000000000, -213.00000000000000000000, -1461.00000000000000000000},
{2504.00000000000000000000, -2462.00000000000000000000, -1778.00000000000000000000},
{-1273.00000000000000000000, -782.00000000000000000000, -371.00000000000000000000}},

{{2170.00000000000000000000, -450.00000000000000000000, -812.00000000000000000000},
{-268.00000000000000000000, 2257.00000000000000000000, -916.00000000000000000000},
{-526.00000000000000000000, -2344.00000000000000000000, -2701.00000000000000000000}},

{{-1296.00000000000000000000, 1310.00000000000000000000, 1.00000000000000000000},
{-1105.00000000000000000000, 1353.00000000000000000000, 1074.00000000000000000000},
{-2042.00000000000000000000, -1427.00000000000000000000, 1727.00000000000000000000}},

{{425.00000000000000000000, -2635.00000000000000000000, -592.00000000000000000000},
{1317.00000000000000000000, -1010.00000000000000000000, 578.00000000000000000000},
{1472.00000000000000000000, -42.00000000000000000000, 1463.00000000000000000000}},

{{-1815.00000000000000000000, 782.00000000000000000000, 1989.00000000000000000000},
{156.00000000000000000000, 86.00000000000000000000, -708.00000000000000000000},
{2180.00000000000000000000, 694.00000000000000000000, -1111.00000000000000000000}},

{{-2213.00000000000000000000, 1934.00000000000000000000, -477.00000000000000000000},
{-136.00000000000000000000, 233.00000000000000000000, 131.00000000000000000000},
{288.00000000000000000000, 2227.00000000000000000000, -1076.00000000000000000000}}},

{{{-2479.00000000000000000000, -681.00000000000000000000, 890.00000000000000000000},
{1688.00000000000000000000, -1706.00000000000000000000, 1830.00000000000000000000},
{-593.00000000000000000000, 1787.00000000000000000000, -2735.00000000000000000000}},

{{1051.00000000000000000000, 251.00000000000000000000, 2702.00000000000000000000},
{1674.00000000000000000000, -2027.00000000000000000000, -101.00000000000000000000},
{-770.00000000000000000000, -1660.00000000000000000000, -2592.00000000000000000000}},

{{1539.00000000000000000000, 2309.00000000000000000000, -961.00000000000000000000},
{-1840.00000000000000000000, 2170.00000000000000000000, -2528.00000000000000000000},
{178.00000000000000000000, 1780.00000000000000000000, -953.00000000000000000000}},

{{-1060.00000000000000000000, 110.00000000000000000000, 2051.00000000000000000000},
{1402.00000000000000000000, 366.00000000000000000000, 990.00000000000000000000},
{-2240.00000000000000000000, 1220.00000000000000000000, -1077.00000000000000000000}},

{{1450.00000000000000000000, 1417.00000000000000000000, 535.00000000000000000000},
{106.00000000000000000000, 511.00000000000000000000, 1715.00000000000000000000},
{1154.00000000000000000000, 1628.00000000000000000000, 1843.00000000000000000000}},

{{806.00000000000000000000, 1151.00000000000000000000, 1753.00000000000000000000},
{-1354.00000000000000000000, -2281.00000000000000000000, -2302.00000000000000000000},
{-2515.00000000000000000000, -704.00000000000000000000, -1018.00000000000000000000}},

{{166.00000000000000000000, 1415.00000000000000000000, 2201.00000000000000000000},
{-1543.00000000000000000000, 740.00000000000000000000, -115.00000000000000000000},
{1806.00000000000000000000, -2655.00000000000000000000, -1437.00000000000000000000}},

{{-773.00000000000000000000, -2347.00000000000000000000, 936.00000000000000000000},
{267.00000000000000000000, -2729.00000000000000000000, -1200.00000000000000000000},
{-2157.00000000000000000000, 861.00000000000000000000, 2652.00000000000000000000}}},

{{{-2287.00000000000000000000, 2335.00000000000000000000, -617.00000000000000000000},
{831.00000000000000000000, -63.00000000000000000000, 2004.00000000000000000000},
{2522.00000000000000000000, -2025.00000000000000000000, 795.00000000000000000000}},

{{-2172.00000000000000000000, 2257.00000000000000000000, 504.00000000000000000000},
{2689.00000000000000000000, -1379.00000000000000000000, -2227.00000000000000000000},
{257.00000000000000000000, 808.00000000000000000000, 905.00000000000000000000}},

{{-540.00000000000000000000, 1355.00000000000000000000, -1691.00000000000000000000},
{2214.00000000000000000000, -589.00000000000000000000, -1592.00000000000000000000},
{2637.00000000000000000000, -598.00000000000000000000, 1073.00000000000000000000}},

{{-965.00000000000000000000, 258.00000000000000000000, 156.00000000000000000000},
{1372.00000000000000000000, -665.00000000000000000000, -236.00000000000000000000},
{1476.00000000000000000000, -2743.00000000000000000000, -449.00000000000000000000}},

{{-169.00000000000000000000, -578.00000000000000000000, 961.00000000000000000000},
{1299.00000000000000000000, 937.00000000000000000000, 365.00000000000000000000},
{1286.00000000000000000000, 43.00000000000000000000, -1508.00000000000000000000}},

{{-1147.00000000000000000000, -1289.00000000000000000000, 68.00000000000000000000},
{398.00000000000000000000, -2571.00000000000000000000, 1964.00000000000000000000},
{665.00000000000000000000, -2085.00000000000000000000, -1388.00000000000000000000}},

{{811.00000000000000000000, 580.00000000000000000000, 2519.00000000000000000000},
{2016.00000000000000000000, -1070.00000000000000000000, -1025.00000000000000000000},
{-2387.00000000000000000000, 1996.00000000000000000000, -90.00000000000000000000}},

{{119.00000000000000000000, -1115.00000000000000000000, 1244.00000000000000000000},
{1799.00000000000000000000, -2408.00000000000000000000, 132.00000000000000000000},
{928.00000000000000000000, 2541.00000000000000000000, 130.00000000000000000000}}},

{{{-906.00000000000000000000, 2387.00000000000000000000, 521.00000000000000000000},
{-2524.00000000000000000000, -1955.00000000000000000000, -1044.00000000000000000000},
{2189.00000000000000000000, 2140.00000000000000000000, 2637.00000000000000000000}},

{{-1355.00000000000000000000, -181.00000000000000000000, -663.00000000000000000000},
{474.00000000000000000000, 984.00000000000000000000, -18.00000000000000000000},
{2249.00000000000000000000, -1061.00000000000000000000, -643.00000000000000000000}},

{{-2657.00000000000000000000, -1759.00000000000000000000, -2491.00000000000000000000},
{-871.00000000000000000000, 1472.00000000000000000000, 2185.00000000000000000000},
{2263.00000000000000000000, 526.00000000000000000000, 1292.00000000000000000000}},

{{-1759.00000000000000000000, 1505.00000000000000000000, -1584.00000000000000000000},
{460.00000000000000000000, 562.00000000000000000000, 1253.00000000000000000000},
{-1495.00000000000000000000, -1005.00000000000000000000, -1108.00000000000000000000}},

{{1229.00000000000000000000, 2565.00000000000000000000, 2322.00000000000000000000},
{-378.00000000000000000000, -2243.00000000000000000000, 376.00000000000000000000},
{1309.00000000000000000000, -1583.00000000000000000000, -2176.00000000000000000000}},

{{2282.00000000000000000000, -1516.00000000000000000000, -1742.00000000000000000000},
{-2019.00000000000000000000, -2451.00000000000000000000, -1109.00000000000000000000},
{2211.00000000000000000000, 555.00000000000000000000, 2653.00000000000000000000}},

{{-1459.00000000000000000000, 781.00000000000000000000, -2078.00000000000000000000},
{2071.00000000000000000000, 491.00000000000000000000, 120.00000000000000000000},
{-1138.00000000000000000000, -884.00000000000000000000, 618.00000000000000000000}},

{{-422.00000000000000000000, -1489.00000000000000000000, -2309.00000000000000000000},
{1312.00000000000000000000, -605.00000000000000000000, 1639.00000000000000000000},
{2636.00000000000000000000, -370.00000000000000000000, 1402.00000000000000000000}}},

{{{-2263.00000000000000000000, 1911.00000000000000000000, -973.00000000000000000000},
{2586.00000000000000000000, 79.00000000000000000000, -585.00000000000000000000},
{2195.00000000000000000000, -383.00000000000000000000, -2574.00000000000000000000}},

{{-1388.00000000000000000000, -1237.00000000000000000000, 846.00000000000000000000},
{-1727.00000000000000000000, -1315.00000000000000000000, 1659.00000000000000000000},
{841.00000000000000000000, 2478.00000000000000000000, 296.00000000000000000000}},

{{-931.00000000000000000000, -1158.00000000000000000000, 1334.00000000000000000000},
{-735.00000000000000000000, 1895.00000000000000000000, -2357.00000000000000000000},
{-378.00000000000000000000, -1096.00000000000000000000, -1050.00000000000000000000}},

{{727.00000000000000000000, 1205.00000000000000000000, -561.00000000000000000000},
{-1225.00000000000000000000, 2305.00000000000000000000, 481.00000000000000000000},
{349.00000000000000000000, 427.00000000000000000000, 1774.00000000000000000000}},

{{-1547.00000000000000000000, -1184.00000000000000000000, 2241.00000000000000000000},
{1477.00000000000000000000, 1726.00000000000000000000, -2400.00000000000000000000},
{1143.00000000000000000000, 1545.00000000000000000000, 2651.00000000000000000000}},

{{800.00000000000000000000, 1188.00000000000000000000, -1628.00000000000000000000},
{-821.00000000000000000000, 1288.00000000000000000000, -286.00000000000000000000},
{1047.00000000000000000000, 912.00000000000000000000, -535.00000000000000000000}},

{{467.00000000000000000000, -2504.00000000000000000000, -1389.00000000000000000000},
{594.00000000000000000000, 1656.00000000000000000000, -2275.00000000000000000000},
{-1707.00000000000000000000, -2496.00000000000000000000, -451.00000000000000000000}},

{{-758.00000000000000000000, 174.00000000000000000000, -2546.00000000000000000000},
{-469.00000000000000000000, -1200.00000000000000000000, 136.00000000000000000000},
{551.00000000000000000000, -1670.00000000000000000000, 441.00000000000000000000}}},

{{{-1854.00000000000000000000, 389.00000000000000000000, -1313.00000000000000000000},
{-351.00000000000000000000, 1454.00000000000000000000, 1028.00000000000000000000},
{-163.00000000000000000000, 640.00000000000000000000, 1416.00000000000000000000}},

{{1638.00000000000000000000, -416.00000000000000000000, 1780.00000000000000000000},
{-540.00000000000000000000, -2501.00000000000000000000, 2358.00000000000000000000},
{-1262.00000000000000000000, 2016.00000000000000000000, 1388.00000000000000000000}},

{{-2704.00000000000000000000, -2303.00000000000000000000, 1635.00000000000000000000},
{-670.00000000000000000000, -1893.00000000000000000000, -803.00000000000000000000},
{-2261.00000000000000000000, 1306.00000000000000000000, 1639.00000000000000000000}},

{{2093.00000000000000000000, 1914.00000000000000000000, 2299.00000000000000000000},
{-2311.00000000000000000000, -203.00000000000000000000, -47.00000000000000000000},
{1238.00000000000000000000, -883.00000000000000000000, 194.00000000000000000000}},

{{205.00000000000000000000, -2589.00000000000000000000, 1716.00000000000000000000},
{972.00000000000000000000, 683.00000000000000000000, -1506.00000000000000000000},
{1383.00000000000000000000, -1073.00000000000000000000, 547.00000000000000000000}},

{{1922.00000000000000000000, 921.00000000000000000000, -947.00000000000000000000},
{-1142.00000000000000000000, 1074.00000000000000000000, -1948.00000000000000000000},
{2429.00000000000000000000, 1687.00000000000000000000, -350.00000000000000000000}},

{{-1821.00000000000000000000, -595.00000000000000000000, 871.00000000000000000000},
{109.00000000000000000000, -2678.00000000000000000000, 389.00000000000000000000},
{-2596.00000000000000000000, -943.00000000000000000000, 2169.00000000000000000000}},

{{821.00000000000000000000, 2502.00000000000000000000, 1378.00000000000000000000},
{-759.00000000000000000000, -1972.00000000000000000000, 1738.00000000000000000000},
{1870.00000000000000000000, 1658.00000000000000000000, 2116.00000000000000000000}}},

{{{-340.00000000000000000000, -1248.00000000000000000000, -2092.00000000000000000000},
{-1660.00000000000000000000, 804.00000000000000000000, -1913.00000000000000000000},
{1804.00000000000000000000, -363.00000000000000000000, 1340.00000000000000000000}},

{{1495.00000000000000000000, 1997.00000000000000000000, -2607.00000000000000000000},
{1609.00000000000000000000, 890.00000000000000000000, 1104.00000000000000000000},
{2181.00000000000000000000, -105.00000000000000000000, -1345.00000000000000000000}},

{{-997.00000000000000000000, -1169.00000000000000000000, -41.00000000000000000000},
{2702.00000000000000000000, -644.00000000000000000000, -2378.00000000000000000000},
{-1579.00000000000000000000, 2465.00000000000000000000, 884.00000000000000000000}},

{{-1379.00000000000000000000, 2096.00000000000000000000, -1273.00000000000000000000},
{271.00000000000000000000, 249.00000000000000000000, 193.00000000000000000000},
{-253.00000000000000000000, -1185.00000000000000000000, 1387.00000000000000000000}},

{{1235.00000000000000000000, 1690.00000000000000000000, -1431.00000000000000000000},
{-1287.00000000000000000000, 2011.00000000000000000000, 1109.00000000000000000000},
{-1867.00000000000000000000, -2226.00000000000000000000, -2187.00000000000000000000}},

{{-1924.00000000000000000000, -1689.00000000000000000000, 164.00000000000000000000},
{1368.00000000000000000000, -2662.00000000000000000000, -2190.00000000000000000000},
{1690.00000000000000000000, 1094.00000000000000000000, -2184.00000000000000000000}},

{{2012.00000000000000000000, -238.00000000000000000000, 1832.00000000000000000000},
{-2207.00000000000000000000, 1679.00000000000000000000, -1607.00000000000000000000},
{1199.00000000000000000000, 879.00000000000000000000, 846.00000000000000000000}},

{{-162.00000000000000000000, 2074.00000000000000000000, 1152.00000000000000000000},
{424.00000000000000000000, -1073.00000000000000000000, 1561.00000000000000000000},
{1603.00000000000000000000, 1402.00000000000000000000, -1595.00000000000000000000}}},

{{{2119.00000000000000000000, 1592.00000000000000000000, 2041.00000000000000000000},
{2518.00000000000000000000, -2045.00000000000000000000, -2147.00000000000000000000},
{2230.00000000000000000000, -1209.00000000000000000000, 1506.00000000000000000000}},

{{1467.00000000000000000000, -316.00000000000000000000, -2530.00000000000000000000},
{-1145.00000000000000000000, 1222.00000000000000000000, 1249.00000000000000000000},
{2033.00000000000000000000, -988.00000000000000000000, -1345.00000000000000000000}},

{{-1526.00000000000000000000, -2077.00000000000000000000, -170.00000000000000000000},
{-1934.00000000000000000000, -2471.00000000000000000000, 1107.00000000000000000000},
{-27.00000000000000000000, 1167.00000000000000000000, 2492.00000000000000000000}},

{{-2584.00000000000000000000, 2276.00000000000000000000, -831.00000000000000000000},
{209.00000000000000000000, -29.00000000000000000000, 1911.00000000000000000000},
{-2081.00000000000000000000, -395.00000000000000000000, 1287.00000000000000000000}},

{{-95.00000000000000000000, 1472.00000000000000000000, 2255.00000000000000000000},
{565.00000000000000000000, -833.00000000000000000000, 2030.00000000000000000000},
{-1496.00000000000000000000, -2451.00000000000000000000, 423.00000000000000000000}},

{{-1069.00000000000000000000, -924.00000000000000000000, -1615.00000000000000000000},
{2036.00000000000000000000, -1447.00000000000000000000, 473.00000000000000000000},
{442.00000000000000000000, -568.00000000000000000000, -2679.00000000000000000000}},

{{-1490.00000000000000000000, -376.00000000000000000000, 206.00000000000000000000},
{-1809.00000000000000000000, -2689.00000000000000000000, 1142.00000000000000000000},
{-1985.00000000000000000000, 1049.00000000000000000000, -2621.00000000000000000000}},

{{-801.00000000000000000000, 2283.00000000000000000000, -1663.00000000000000000000},
{240.00000000000000000000, -1526.00000000000000000000, 299.00000000000000000000},
{2373.00000000000000000000, 603.00000000000000000000, -1620.00000000000000000000}}},

{{{-1847.00000000000000000000, -2326.00000000000000000000, 1484.00000000000000000000},
{-215.00000000000000000000, -1219.00000000000000000000, 1863.00000000000000000000},
{-2214.00000000000000000000, 1484.00000000000000000000, 2343.00000000000000000000}},

{{387.00000000000000000000, -2370.00000000000000000000, -909.00000000000000000000},
{-505.00000000000000000000, -2651.00000000000000000000, 130.00000000000000000000},
{-2587.00000000000000000000, 2158.00000000000000000000, -152.00000000000000000000}},

{{-1968.00000000000000000000, -570.00000000000000000000, -1420.00000000000000000000},
{-689.00000000000000000000, 842.00000000000000000000, 972.00000000000000000000},
{-224.00000000000000000000, 861.00000000000000000000, 1940.00000000000000000000}},

{{-275.00000000000000000000, -229.00000000000000000000, 1390.00000000000000000000},
{1520.00000000000000000000, -2677.00000000000000000000, -1233.00000000000000000000},
{-44.00000000000000000000, 2117.00000000000000000000, 757.00000000000000000000}},

{{-1331.00000000000000000000, 11.00000000000000000000, 2322.00000000000000000000},
{37.00000000000000000000, -2472.00000000000000000000, -314.00000000000000000000},
{-942.00000000000000000000, 1785.00000000000000000000, 943.00000000000000000000}},

{{1144.00000000000000000000, 1299.00000000000000000000, -2674.00000000000000000000},
{-480.00000000000000000000, -981.00000000000000000000, 1230.00000000000000000000},
{2115.00000000000000000000, -558.00000000000000000000, 934.00000000000000000000}},

{{42.00000000000000000000, 777.00000000000000000000, 2286.00000000000000000000},
{-2313.00000000000000000000, -361.00000000000000000000, 61.00000000000000000000},
{711.00000000000000000000, -637.00000000000000000000, -981.00000000000000000000}},

{{1745.00000000000000000000, -415.00000000000000000000, -629.00000000000000000000},
{1095.00000000000000000000, 2685.00000000000000000000, -624.00000000000000000000},
{-874.00000000000000000000, -329.00000000000000000000, 1288.00000000000000000000}}},

{{{895.00000000000000000000, -444.00000000000000000000, -916.00000000000000000000},
{-2409.00000000000000000000, -1661.00000000000000000000, 757.00000000000000000000},
{-1131.00000000000000000000, -1253.00000000000000000000, -1917.00000000000000000000}},

{{-266.00000000000000000000, -2484.00000000000000000000, 2630.00000000000000000000},
{1323.00000000000000000000, 823.00000000000000000000, 1160.00000000000000000000},
{-557.00000000000000000000, 1110.00000000000000000000, 1778.00000000000000000000}},

{{-984.00000000000000000000, 1653.00000000000000000000, -255.00000000000000000000},
{-1173.00000000000000000000, 752.00000000000000000000, 1461.00000000000000000000},
{-838.00000000000000000000, -204.00000000000000000000, -1196.00000000000000000000}},

{{-2384.00000000000000000000, 2621.00000000000000000000, 1476.00000000000000000000},
{2111.00000000000000000000, 2386.00000000000000000000, -296.00000000000000000000},
{625.00000000000000000000, 2569.00000000000000000000, -1657.00000000000000000000}},

{{-2113.00000000000000000000, -1085.00000000000000000000, -1618.00000000000000000000},
{1645.00000000000000000000, -2002.00000000000000000000, -2117.00000000000000000000},
{-355.00000000000000000000, 377.00000000000000000000, -711.00000000000000000000}},

{{-2135.00000000000000000000, -1886.00000000000000000000, 186.00000000000000000000},
{-1218.00000000000000000000, -1653.00000000000000000000, -872.00000000000000000000},
{533.00000000000000000000, 731.00000000000000000000, 2338.00000000000000000000}},

{{1961.00000000000000000000, 2244.00000000000000000000, 509.00000000000000000000},
{418.00000000000000000000, 671.00000000000000000000, 2719.00000000000000000000},
{-365.00000000000000000000, -1113.00000000000000000000, -2464.00000000000000000000}},

{{-1140.00000000000000000000, 1481.00000000000000000000, 2337.00000000000000000000},
{1534.00000000000000000000, -1584.00000000000000000000, 1957.00000000000000000000},
{-318.00000000000000000000, -2730.00000000000000000000, -2009.00000000000000000000}}},

{{{-938.00000000000000000000, 148.00000000000000000000, 1552.00000000000000000000},
{-1676.00000000000000000000, 549.00000000000000000000, 879.00000000000000000000},
{-1661.00000000000000000000, 724.00000000000000000000, -407.00000000000000000000}},

{{1243.00000000000000000000, 430.00000000000000000000, 1837.00000000000000000000},
{46.00000000000000000000, -2018.00000000000000000000, -238.00000000000000000000},
{-479.00000000000000000000, -1870.00000000000000000000, 219.00000000000000000000}},

{{-758.00000000000000000000, -1286.00000000000000000000, 947.00000000000000000000},
{-876.00000000000000000000, 2298.00000000000000000000, 1514.00000000000000000000},
{2525.00000000000000000000, 1273.00000000000000000000, -1067.00000000000000000000}},

{{-1163.00000000000000000000, 10.00000000000000000000, -1885.00000000000000000000},
{-2597.00000000000000000000, -331.00000000000000000000, -751.00000000000000000000},
{29.00000000000000000000, -1067.00000000000000000000, -249.00000000000000000000}},

{{2732.00000000000000000000, 562.00000000000000000000, -840.00000000000000000000},
{-133.00000000000000000000, 1150.00000000000000000000, 518.00000000000000000000},
{81.00000000000000000000, -207.00000000000000000000, 353.00000000000000000000}},

{{1284.00000000000000000000, 327.00000000000000000000, 1166.00000000000000000000},
{-2255.00000000000000000000, -2672.00000000000000000000, 1469.00000000000000000000},
{1603.00000000000000000000, -935.00000000000000000000, 1445.00000000000000000000}},

{{-1082.00000000000000000000, -687.00000000000000000000, -2177.00000000000000000000},
{-557.00000000000000000000, -1935.00000000000000000000, -1880.00000000000000000000},
{-1321.00000000000000000000, -2329.00000000000000000000, 1084.00000000000000000000}},

{{-2610.00000000000000000000, 1211.00000000000000000000, 232.00000000000000000000},
{465.00000000000000000000, 2662.00000000000000000000, 927.00000000000000000000},
{-1630.00000000000000000000, -2557.00000000000000000000, 618.00000000000000000000}}},

{{{1040.00000000000000000000, -387.00000000000000000000, 1186.00000000000000000000},
{71.00000000000000000000, 1019.00000000000000000000, 765.00000000000000000000},
{-1605.00000000000000000000, 580.00000000000000000000, 1814.00000000000000000000}},

{{2699.00000000000000000000, 941.00000000000000000000, -2084.00000000000000000000},
{1561.00000000000000000000, 1743.00000000000000000000, 901.00000000000000000000},
{-253.00000000000000000000, 1262.00000000000000000000, 976.00000000000000000000}},

{{141.00000000000000000000, -1822.00000000000000000000, -20.00000000000000000000},
{136.00000000000000000000, -2156.00000000000000000000, -2279.00000000000000000000},
{-1171.00000000000000000000, 1583.00000000000000000000, 552.00000000000000000000}},

{{-1108.00000000000000000000, 1975.00000000000000000000, 269.00000000000000000000},
{142.00000000000000000000, 1222.00000000000000000000, 1895.00000000000000000000},
{-1113.00000000000000000000, -992.00000000000000000000, -1195.00000000000000000000}},

{{1431.00000000000000000000, -1922.00000000000000000000, 1438.00000000000000000000},
{506.00000000000000000000, 1629.00000000000000000000, -1377.00000000000000000000},
{-1846.00000000000000000000, -2473.00000000000000000000, -863.00000000000000000000}},

{{1176.00000000000000000000, 1453.00000000000000000000, 1857.00000000000000000000},
{345.00000000000000000000, 920.00000000000000000000, -2068.00000000000000000000},
{-1725.00000000000000000000, 975.00000000000000000000, -1147.00000000000000000000}},

{{1413.00000000000000000000, -1595.00000000000000000000, 2527.00000000000000000000},
{-230.00000000000000000000, 762.00000000000000000000, -2540.00000000000000000000},
{2336.00000000000000000000, -875.00000000000000000000, 791.00000000000000000000}},

{{2438.00000000000000000000, -2473.00000000000000000000, 168.00000000000000000000},
{2556.00000000000000000000, -818.00000000000000000000, 1168.00000000000000000000},
{2151.00000000000000000000, 2331.00000000000000000000, -532.00000000000000000000}}}};

const uint16_t shape_Conv2D_2_b = 16;
const int16_t Conv2D_2_b[16] = {23.00000000000000000000, 23.00000000000000000000, -7.00000000000000000000, 23.00000000000000000000, -17.00000000000000000000, -21.00000000000000000000, -22.00000000000000000000, 24.00000000000000000000, 25.00000000000000000000, -23.00000000000000000000, -21.00000000000000000000, 22.00000000000000000000, 2.00000000000000000000, -19.00000000000000000000, -5.00000000000000000000, 22.00000000000000000000};
