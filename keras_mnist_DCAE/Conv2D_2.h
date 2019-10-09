/*
 * author : shtsno24
 * Date : 2019-10-09 15:00:55.991446
 *
 */
const float Conv2D_2_w[16][8][3][3] =
{{{{0.080562115, 0.115303695, 0.013886102},
{0.017729355, 0.014080078, 0.085437804},
{-0.048503533, 0.060917754, -0.07201539}},

{{-0.07672718, -0.15718731, 0.0056369184},
{0.11097998, 0.1454733, -0.03630317},
{0.06284934, 0.17103447, 0.12950106}},

{{-0.054353155, -0.023895992, -0.06549619},
{-0.05471176, 0.042906437, 0.06463414},
{0.13253465, -0.14388976, -0.00067393476}},

{{-0.05200444, 0.08112016, -0.16982785},
{0.02592985, 0.13140215, -0.16293083},
{0.08471828, -0.09379405, 0.11874008}},

{{-0.0070113936, -0.018424308, -0.15110762},
{0.012176019, -0.074799255, 0.05720948},
{0.08674605, 0.044402752, 0.1576364}},

{{-0.047079585, -0.092002355, 0.027163854},
{0.010804117, -0.092225514, -0.024458626},
{-0.07810382, -0.15264353, -0.075570926}},

{{-0.03654641, -0.1742702, 0.04749818},
{0.08355791, -0.0058606486, -0.02492163},
{0.15514864, -0.021763267, 0.038024932}},

{{-0.041343503, -0.016329512, -0.10657135},
{-0.027090855, -0.049545623, 0.10262668},
{-0.020231888, -0.14307417, 0.07124757}}},

{{{0.095556796, -0.053228635, 0.09738098},
{0.00077277154, 0.12681828, -0.056190934},
{0.09563976, 0.17782982, 0.13651413}},

{{-0.17075288, -0.07888183, 0.047164764},
{0.06892523, 0.08091679, 0.17892495},
{0.04886408, 0.13138819, -0.054491114}},

{{0.049105495, -0.005609082, 0.029407285},
{0.05708552, 0.002432561, -0.070884526},
{-0.012521385, 0.043177124, -0.14963515}},

{{0.078064404, -0.108879276, -0.07509662},
{-0.0012954568, -0.06058664, 0.112769224},
{-0.12418918, 0.13824604, 0.061801355}},

{{0.038410395, 0.14872268, -0.11081464},
{0.18392995, -0.080635026, -0.16326393},
{-0.017167244, -0.029240826, -0.0055529065}},

{{0.14900722, 0.03355866, -0.0048799254},
{0.11532947, 0.07455153, -0.10594028},
{-0.011027629, -0.099112675, -0.05873466}},

{{0.13751805, -0.07678388, -0.17957899},
{0.04491707, 0.05146739, -0.19185929},
{0.055073306, -0.058182, -0.029865902}},

{{-0.025230614, -0.16077384, 0.039581843},
{0.10287092, -0.19569497, 0.009114243},
{0.20780127, -0.036806203, -0.031891935}}},

{{{-0.12866868, 0.07064656, -0.07653144},
{-0.11445076, -0.15204875, -0.096712805},
{-0.1692878, 0.1403911, -0.08873565}},

{{0.14983769, 0.12181397, -0.09178406},
{-0.14102587, -0.1449319, -0.03219699},
{0.022401392, -0.14722751, 0.011444533}},

{{0.016272822, -0.04968966, 0.16019215},
{-0.12503457, -0.10750621, -0.16324787},
{-0.015707089, -0.028693508, 0.14942773}},

{{0.06746976, -0.13929862, 0.0044189813},
{0.059526782, -0.08726522, 0.08963014},
{0.0073514627, 0.13261686, 0.008774042}},

{{-0.012041931, -0.05430646, 0.12738816},
{-0.0900393, -0.08523367, 0.10435754},
{-0.10768484, -0.082365476, 0.107885525}},

{{-0.08350302, 0.056786157, 0.038875803},
{0.027322853, -0.08039982, 0.1197944},
{-0.1044242, -0.019519087, -0.053515315}},

{{0.06056581, -0.04814091, 0.13860327},
{-0.15559185, -0.12517427, -0.09803888},
{-0.14403056, -0.10674663, -0.06805045}},

{{-0.079941384, -0.011751058, -0.01757502},
{0.1238661, 0.07769983, -0.03084941},
{0.03675059, -0.11320345, -0.16160993}}},

{{{-0.1628061, 0.13698898, -0.09059322},
{0.06335485, -0.10437012, -0.027586592},
{0.040155075, -0.102604955, 0.026117867}},

{{0.153677, -0.08500992, 0.054714553},
{-0.15020986, -0.09045579, 0.028800085},
{0.047484018, -0.07863336, 0.018141728}},

{{-0.105175555, -0.050943036, -0.037379585},
{-0.05978887, -0.07637713, -0.07518587},
{0.06016293, -0.039409053, -0.1562244}},

{{0.09190851, 0.05763235, -0.06954725},
{0.018774778, -0.008492814, -0.038926475},
{0.036126826, -0.07698851, -0.058129903}},

{{-0.11756311, 0.06315944, 0.046982102},
{0.12948053, -0.049110103, 0.027497636},
{0.022080585, -0.14999722, -0.004820053}},

{{-0.015001204, 0.04066882, -0.18989234},
{-0.06868687, -0.1617011, -0.10765826},
{-0.08034561, 0.058863074, -0.16245262}},

{{-0.104651526, 0.023788616, 0.044505004},
{-0.018473469, -0.11144025, -0.08451123},
{-0.028671771, -0.11752158, 0.07878717}},

{{0.077318154, -0.0695673, 0.11600104},
{0.13004012, -0.1526306, 0.114793465},
{-0.0065690014, -0.016674617, 0.027054729}}},

{{{-0.121264674, -0.029818019, -0.12383513},
{0.080901034, -0.100028776, 0.045904435},
{0.10064402, 0.080952756, -0.15366599}},

{{-0.101282574, -0.09995841, -0.017210094},
{0.13762341, -0.13921244, -0.042471517},
{-0.12388221, -0.037047587, -0.14745879}},

{{0.12773627, -0.08427498, 0.063411854},
{-0.02390027, 0.07702646, 0.111613326},
{-0.16143505, 0.0280393, -0.04099086}},

{{0.11474975, -0.12113101, 0.15365472},
{-0.030862832, 0.12637049, 0.03991963},
{-0.06954238, 0.1642796, -0.097065896}},

{{-0.005714479, -0.033971496, 0.18933877},
{-0.06466786, -0.15399899, -0.033892605},
{-0.14630662, -0.13059473, -0.07487494}},

{{0.12095125, -0.09884027, 0.11943821},
{-0.1475256, -0.012735841, 0.034817725},
{0.12784648, 0.25997677, 0.08336313}},

{{0.053042833, -0.03586104, 0.00010134774},
{0.09934438, 0.1275474, 0.22602256},
{-0.13338542, -0.0015794082, 0.14707278}},

{{-0.069005705, 0.11512701, 0.074708045},
{0.10428938, -0.0997043, -0.021133162},
{-0.10032487, 0.12308498, 0.10708146}}},

{{{-0.009069853, 0.15753056, -0.14129624},
{0.04179336, 0.04834084, -0.043323435},
{0.031786792, 0.12578762, -0.1017998}},

{{-0.029090988, -0.052233294, 0.073032066},
{0.17285545, 0.02472123, 0.008733088},
{-0.0848037, -0.10186633, -0.03748178}},

{{-0.12016955, 0.030071374, 0.14835456},
{-0.06997911, 0.13433614, -0.05858758},
{0.09783333, -0.036182556, 0.16144092}},

{{-0.120291926, -0.01985996, -0.10947373},
{-0.053220753, 0.11827102, 0.0069225873},
{0.14700058, 0.15371661, -0.107082315}},

{{0.12650694, 0.08150358, 0.026420387},
{0.16650006, -0.14789821, 0.050262596},
{-0.1229099, -0.020690594, -0.1614833}},

{{0.09410435, -0.06669214, 0.027405376},
{0.008772936, -0.07614559, 0.028215736},
{-0.19242382, 0.16765948, 0.15827873}},

{{0.050754894, -0.059587196, -0.16931526},
{-0.047826912, 0.11728989, -0.17270316},
{-0.0008739702, -0.12184857, -0.06510162}},

{{0.030264106, 0.19229783, 0.06697448},
{-0.085735135, -0.10414419, 0.077763885},
{0.033157025, 0.061442062, 0.13635506}}},

{{{0.13781306, 0.17048039, 0.11178667},
{0.14295647, -0.09117312, 0.09611481},
{-0.12782492, 0.06515502, -0.0145251285}},

{{0.056816768, -0.13574816, -0.045513358},
{0.022288429, 0.038844015, -0.14387298},
{0.12537403, 0.063706875, -0.13339703}},

{{-0.085143976, 0.027087187, 0.15347695},
{0.104466066, -0.025857523, -0.08320444},
{0.09109508, 0.04328852, 0.0486779}},

{{0.13354647, 0.17902902, 0.02321328},
{-0.020834697, -0.10203016, 0.07882125},
{0.08659716, 0.031054404, -0.06294122}},

{{-0.07875299, 0.020734223, 0.08473241},
{-0.07697832, -0.034650385, 0.11848137},
{-0.11410608, 0.014128017, 0.10385189}},

{{-0.09282618, 0.10791333, 0.10053055},
{-0.07504364, 0.086649634, 0.13235652},
{-0.13991861, -0.14378779, -0.13253829}},

{{-0.013084436, -0.019023037, -0.07491029},
{0.02815638, 0.08135212, 0.054366168},
{-0.07922703, -0.03262204, 0.092887655}},

{{-0.11082516, -0.08863047, -0.040246516},
{0.1523513, -0.06722263, 0.045168616},
{-0.099867634, -0.14138366, 0.13742664}}},

{{{0.00962404, 0.1490731, 0.047874615},
{0.13104968, 0.09861865, -0.1207422},
{0.065102264, 0.112741224, 0.11183562}},

{{-0.110194094, -0.06687203, -0.035220046},
{0.118822, 0.17584251, 0.033959918},
{0.15626532, -0.1358738, 0.02631638}},

{{0.07450861, 0.10838904, 0.04774699},
{-0.021914396, 0.11122296, -0.059245557},
{0.08974802, -0.045563668, -0.046259213}},

{{0.08070791, 0.031101389, 0.026359154},
{0.032974135, 0.070179746, -0.12274221},
{0.107285365, 0.13113149, 0.182918}},

{{0.055077653, -0.14890026, -0.15716453},
{-0.12261258, 0.025464322, -0.14695399},
{-0.05603605, 0.17460692, 0.021205762}},

{{-0.10743642, -0.00865022, -0.052321743},
{-0.0011938618, -0.07798634, -0.11252287},
{0.003880321, -0.004010471, 0.16168487}},

{{-0.061804343, -0.107095756, -0.08352644},
{-0.094371565, 0.10531398, 0.10260357},
{0.04981127, -0.0720957, 0.16892456}},

{{-0.15841222, 0.013914098, -0.18913941},
{0.0077832383, -0.17016429, -0.14833824},
{-0.08986494, 0.14937009, -0.07910286}}},

{{{-0.034188136, 0.059304893, 0.1022938},
{-0.06373858, -0.028115455, 0.057443272},
{-0.0718294, -0.081386864, -0.1522056}},

{{-0.010007313, 0.036749594, 0.13745856},
{-0.049263317, 0.06669135, 0.078072295},
{0.010541297, -0.041202683, 0.08029854}},

{{-0.0063224933, 0.13170929, 0.10586097},
{0.15560417, -0.08733596, -0.13937949},
{-0.0038897388, -0.12682246, 0.121968046}},

{{-0.05070965, 0.079630226, 0.0745263},
{0.07357434, -0.12064433, 0.050690394},
{0.0823457, 0.086816244, 0.0047961534}},

{{0.0604831, -0.10766663, 0.10359975},
{-0.012964396, -0.09289087, -0.04687081},
{0.10341426, 0.07573664, 0.14415686}},

{{-0.09102116, 0.023667492, -0.006032281},
{-0.03630448, -0.046227165, 0.11469666},
{-0.07437736, 0.017494887, -0.14078599}},

{{-0.03283342, -0.13670918, 0.14526427},
{0.06954677, -0.03803547, 0.090456106},
{0.14352433, -0.057192177, 0.024449825}},

{{-0.052219193, 0.13453063, 0.093762025},
{-0.10952385, 0.12920675, 0.10718188},
{-0.08052251, 0.026812242, -0.14632669}}},

{{{0.074844114, 0.09792284, -0.15042607},
{-0.08402718, 0.06787814, 0.11995632},
{0.030714722, -0.12542447, -0.07760647}},

{{-0.120668255, 0.13177346, 0.0618119},
{-0.14226335, -0.18792048, 0.009345849},
{-0.1870349, -0.05699393, -0.030350255}},

{{0.037323, -0.10040561, 0.0915055},
{-0.103491485, 0.044294067, -0.09942272},
{0.11665704, 0.03211089, 0.07565867}},

{{-0.12328767, -0.10995263, 0.07510486},
{0.12187992, 0.08227911, 0.08225759},
{0.01501472, -0.063948296, 0.12000471}},

{{0.12330859, -0.14743955, -0.0070073404},
{0.1152485, 0.08294383, -0.009031816},
{0.051737975, 0.1693608, 0.08152237}},

{{0.07734117, 0.11207154, -0.036765933},
{-0.04833748, -0.09817599, -0.02135682},
{0.16835345, 0.034099396, -0.011925732}},

{{0.011407789, 0.044230364, 0.0125583485},
{-0.0031842764, -0.006601688, 0.095695935},
{0.092846096, 0.029732862, -0.15211134}},

{{0.045155805, -0.11030392, -0.07705416},
{0.040327914, 0.0148047125, 0.17269163},
{0.008642659, 0.13510647, 0.029432438}}},

{{{0.04156037, -0.07563388, -0.009065924},
{-0.0035406633, 0.056238152, -0.10987748},
{-0.13626488, 0.05224683, -0.16077247}},

{{0.17217216, 0.13976142, -0.019799003},
{-0.19692372, -0.05854613, 0.13034368},
{0.06731517, -0.17755853, 0.06873453}},

{{0.16323222, -0.07311323, -0.17301247},
{0.18094474, 0.074606895, -0.015910614},
{-0.0081100045, 0.15907392, -0.054853998}},

{{0.062258616, -0.10936492, -0.10120973},
{0.053779002, 0.021197936, 0.11899553},
{0.09956153, -0.12883873, 0.15269558}},

{{-0.14819355, -0.011651886, 0.06416003},
{-0.083979, 0.006530185, 0.09293828},
{0.012931089, 0.13122033, 0.095959276}},

{{0.061463322, -0.051535375, -0.046779007},
{0.013475998, 0.059596367, 0.028754277},
{0.087272935, -0.11320516, 0.1356739}},

{{0.14454295, -0.020591887, -0.04005706},
{0.17518899, 0.122590914, 0.11557277},
{-0.032573048, 0.11527341, 0.05657905}},

{{0.15527344, -0.18359555, 0.0032457483},
{0.08802165, -0.051224507, 0.08639289},
{-0.03361305, -0.009277137, -0.104697935}}},

{{{0.0629031, -0.14602, 0.13853274},
{0.10307907, -0.05963879, 0.10497954},
{-0.018138934, 0.04049875, -0.11058354}},

{{0.09913908, -0.034041673, 0.062368266},
{0.1290805, -0.026037514, -0.09136756},
{0.12649645, 0.062151596, 0.114994034}},

{{0.13643019, 0.052088313, 0.018218886},
{-0.02287614, 0.1716384, -0.115158394},
{-0.07689781, -0.03548578, -0.035896737}},

{{0.120807774, -0.057271294, 0.012156692},
{0.080681056, -0.030529313, 0.05206629},
{0.09694024, 0.11573255, 0.036714543}},

{{-0.04979717, -0.03776257, -0.06002408},
{0.08283988, -0.17304508, -0.18446147},
{-0.06341634, 0.00058393634, 0.09580436}},

{{-0.15083113, 0.035306014, -0.09784722},
{-0.013121316, -0.09459102, 0.15617709},
{-0.03042958, 0.14143018, 0.084171124}},

{{-0.064679414, -0.1837895, 0.060940556},
{-0.15835154, -0.16367415, 0.010672319},
{-0.12263134, 0.10024481, -0.051078677}},

{{-0.14281027, -0.15163727, 0.09243926},
{-0.07254386, 0.029429706, 0.06282013},
{0.097396515, -0.08275528, 0.17864095}}},

{{{0.075908005, -0.12777734, -0.06699668},
{0.11563907, 0.05665587, 0.13052751},
{0.085440055, -0.18764259, 0.076715775}},

{{0.025262091, -0.11853288, 0.13929507},
{-0.12659132, 0.041534454, -0.016843691},
{-0.08174296, 0.10404002, 0.032770354}},

{{-0.024050793, 0.066964954, -0.11188467},
{0.05795915, -0.1396163, -0.15300283},
{0.01680972, -0.11701612, -0.05585669}},

{{-0.02582114, -0.07453177, 0.008367395},
{0.030278835, -0.06376106, -0.09068833},
{0.06095673, -0.03917476, 0.07689714}},

{{0.16155082, -0.03719756, -0.09290078},
{0.08000756, -0.0042383997, 0.009657189},
{0.08696826, 0.08027743, 0.01435606}},

{{-0.046709787, 0.10091476, 0.0018828418},
{0.10646848, 0.07899982, -0.038291954},
{-0.059919182, -0.021544544, 0.07588618}},

{{-0.026796244, -0.093051665, 0.009115931},
{-0.1562218, 0.14718696, 0.006810259},
{-0.123050936, 0.15071552, 0.08009513}},

{{0.04777524, -0.07829353, 0.07330404},
{0.064517975, 0.11755836, -0.08049841},
{0.0018797364, 0.15111592, 0.100925356}}},

{{{-0.062815346, -0.14186116, 0.040929195},
{0.10782514, 0.13169704, 0.16035534},
{-0.03786016, 0.059358656, -0.032494288}},

{{-0.038879734, 0.12732235, -0.0068454994},
{0.062567435, -0.11053542, -0.10260039},
{0.017403496, 0.045719016, -0.053212725}},

{{-0.108999915, -0.11698232, 0.15151526},
{0.007777755, 0.11502486, -0.024505632},
{-0.15672082, -0.095143475, 0.0071916105}},

{{0.060270157, 0.0029660147, -0.16439943},
{0.05573106, -0.0903863, 0.020254364},
{0.06786344, -0.05260669, -0.1255467}},

{{-0.0770841, -0.15117075, 0.04282259},
{-0.13980573, -0.08144482, -0.014972808},
{0.005241162, 0.05054068, 0.06419092}},

{{0.10949457, -0.17687464, 0.056273524},
{0.03597526, 0.048885975, -0.040244196},
{-0.057624847, 0.11930304, -0.10174544}},

{{0.010429186, -0.14678907, 0.05644574},
{-0.05136988, 0.10471951, 0.052866053},
{-0.005671776, 0.0447059, -0.006310019}},

{{-0.18582925, 0.13341625, -0.15321372},
{-0.006558158, 0.08249171, -0.110862315},
{0.062117416, -0.0994365, 0.02316567}}},

{{{0.035771865, 0.05176217, 0.0012843662},
{-0.03893578, -0.19483311, -0.14587764},
{0.09532938, -0.06885407, 0.1001827}},

{{0.013907437, -0.16232306, 0.0007978742},
{0.03009909, -0.029848414, 0.12771848},
{-0.17528667, 0.14226511, 0.110135294}},

{{0.109558724, -0.13244516, 0.06829365},
{-0.10014866, -0.08151993, -0.07844803},
{0.051730912, -0.051464718, -0.066472836}},

{{-0.034990218, 0.06843159, 0.01712671},
{0.008831468, -0.13141152, 0.11766845},
{-0.18333757, 0.042818494, -0.05304868}},

{{0.025930002, 0.10304746, 0.10563179},
{0.0066448124, 0.12043786, 0.09328915},
{-0.09259698, -0.038396876, -0.031544045}},

{{0.080697246, 0.1735651, 0.18992315},
{-0.03841022, -0.15714633, -0.06703393},
{0.0028607436, -0.12308126, -0.07211213}},

{{0.007175915, 0.008245328, 0.17705788},
{0.21670647, 0.21886642, -0.10157814},
{0.026919017, 0.12027987, -0.08468362}},

{{-0.0104883965, 0.10790189, -0.042336073},
{0.019800736, 0.033334706, -0.12693739},
{-0.0037237327, -0.015561391, 0.07798058}}},

{{{-0.15677112, 0.042090926, -0.09347711},
{0.032618918, 0.18868035, -0.05294293},
{-0.10256851, -0.18686204, -0.029793441}},

{{0.054849118, 0.12159882, 0.051893555},
{-0.10833897, 0.11068034, -0.13640304},
{0.026952868, -0.043471843, -0.030301584}},

{{-0.07422523, -0.027721316, -0.059020024},
{0.18416536, -0.12804966, 0.02923206},
{0.00043680958, 0.06215065, 0.09569713}},

{{-0.033190116, -0.0805007, 0.0063841925},
{-0.0012406805, -0.079859674, 0.056532606},
{0.028170027, -0.1584229, 0.040598545}},

{{0.17967507, -0.03741625, -0.062893525},
{-0.07931028, 0.045772966, -0.10427599},
{0.15349494, 0.17241266, 0.069853075}},

{{-0.11521566, 0.09030451, 0.105808675},
{-0.097035006, -0.04800137, 0.18496425},
{0.12165818, 0.1394195, 0.03862901}},

{{-0.14555635, 0.02349212, -0.04345728},
{0.087721, 0.018524399, -0.0077362335},
{0.035022445, 0.00644292, -0.044335518}},

{{-0.02863498, -0.12769602, -0.008869476},
{0.038925175, 0.18427616, 0.031462666},
{0.04628152, 0.17015931, 0.015968703}}}};

const float Conv2D_2_b[16] = {-0.012929949, 0.0630273, -0.069715716, -0.08356452, 0.04429992, 0.081198364, -0.032772325, 0.0320213, -0.00046408747, -0.051308192, -0.06706433, 0.09638268, -0.029690199, -0.00930249, -0.02351821, -0.06211508};
