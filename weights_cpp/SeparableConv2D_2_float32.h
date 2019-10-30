/*
 * author : shtsno24
 * Date : 2019-10-30 15:55:45.212421
 *
 */
#pragma once
#include <cstdint>
#include <vector>

using namespace std;

const vector< uint16_t> shape_SeparableConv2D_2_w_d = {1, 8, 3, 3};
const vector< vector< vector< vector< float> > > > SeparableConv2D_2_w_d =
{{{{0.12318683415651321411, 0.19802996516227722168, 0.12955757975578308105},
{-0.13889774680137634277, -0.30675020813941955566, 0.19074983894824981689},
{0.35320898890495300293, 0.10382184386253356934, 0.15997031331062316895}},

{{-0.02889433503150939941, -0.15492206811904907227, -0.04668803885579109192},
{0.20332345366477966309, -0.46503454446792602539, 0.02512421086430549622},
{0.12253284454345703125, -0.00348307960666716099, 0.26910397410392761230}},

{{0.09175038337707519531, -0.01772274449467658997, -0.28202739357948303223},
{-0.25010973215103149414, -0.07469402998685836792, -0.04178110882639884949},
{0.21915055811405181885, 0.15225595235824584961, 0.07028492540121078491}},

{{-0.06038276106119155884, 0.19344548881053924561, 0.18556565046310424805},
{-0.14959512650966644287, -0.46939072012901306152, 0.23927064239978790283},
{0.03911873698234558105, 0.19585107266902923584, -0.16859230399131774902}},

{{0.13026094436645507812, -0.10342381894588470459, 0.13207735121250152588},
{0.34865763783454895020, 0.37455934286117553711, -0.04197940602898597717},
{0.20164336264133453369, -0.10845755785703659058, 0.07610692828893661499}},

{{-0.00292597711086273193, -0.30257508158683776855, 0.11501464992761611938},
{0.04832935333251953125, -0.37309524416923522949, -0.07227592170238494873},
{-0.00082984712207689881, 0.13254645466804504395, 0.11767245084047317505}},

{{0.02516337297856807709, -0.12725307047367095947, -0.14512166380882263184},
{-0.21690465509891510010, 0.35918226838111877441, 0.28538024425506591797},
{-0.08975135535001754761, 0.29213154315948486328, 0.28989279270172119141}},

{{0.07573156803846359253, -0.08683820813894271851, 0.29921400547027587891},
{-0.34226223826408386230, 0.08457146584987640381, 0.10928081721067428589},
{-0.04361326992511749268, 0.25210085511207580566, -0.29982227087020874023}}}};

const vector< uint16_t> shape_SeparableConv2D_2_w_p = {8, 8, 1, 1};
const vector< vector< vector< vector< float> > > > SeparableConv2D_2_w_p =
{{{{-0.56637579202651977539}},

{{-0.02495341189205646515}},

{{0.58435958623886108398}},

{{0.11627004295587539673}},

{{0.60546511411666870117}},

{{-0.08361133188009262085}},

{{0.22369673848152160645}},

{{-0.33607393503189086914}}},

{{{0.60307168960571289062}},

{{-0.41136077046394348145}},

{{-0.46094384789466857910}},

{{-0.59031760692596435547}},

{{-0.19413749873638153076}},

{{0.31197512149810791016}},

{{0.30858176946640014648}},

{{-0.36808949708938598633}}},

{{{0.12272295355796813965}},

{{0.32277292013168334961}},

{{0.56390106678009033203}},

{{0.47175627946853637695}},

{{0.25021660327911376953}},

{{-0.37189766764640808105}},

{{-0.04973595216870307922}},

{{0.48686155676841735840}}},

{{{-0.30879119038581848145}},

{{0.31035646796226501465}},

{{0.04223306849598884583}},

{{-0.27730709314346313477}},

{{-0.58754730224609375000}},

{{-0.19762070477008819580}},

{{0.10538849979639053345}},

{{0.08530753850936889648}}},

{{{0.46992608904838562012}},

{{0.24016918241977691650}},

{{-0.24646602571010589600}},

{{0.45649138092994689941}},

{{-0.54458373785018920898}},

{{0.07588491588830947876}},

{{-0.01969447545707225800}},

{{0.22233593463897705078}}},

{{{-0.02209898456931114197}},

{{0.35813239216804504395}},

{{0.28263729810714721680}},

{{0.26518449187278747559}},

{{0.17784331738948822021}},

{{0.20970593392848968506}},

{{-0.49453693628311157227}},

{{0.44285619258880615234}}},

{{{-0.25839337706565856934}},

{{-0.06908650696277618408}},

{{0.58070510625839233398}},

{{-0.63216960430145263672}},

{{0.27835106849670410156}},

{{-0.37566283345222473145}},

{{0.52163475751876831055}},

{{0.60173338651657104492}}},

{{{-0.15581607818603515625}},

{{-0.05762815102934837341}},

{{-0.57080155611038208008}},

{{-0.36093419790267944336}},

{{-0.08831298351287841797}},

{{0.36024791002273559570}},

{{-0.45055800676345825195}},

{{0.09007342904806137085}}}};

const uint16_t shape_SeparableConv2D_2_b_d = 8;
const vector< float> SeparableConv2D_2_b_d = {0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000};
const uint16_t shape_SeparableConv2D_2_b_p = 8;
const vector< float> SeparableConv2D_2_b_p = {0.06709032505750656128, 0.12191348522901535034, 0.08528288453817367554, -0.00895263068377971649, 0.04697225242853164673, 0.19480545818805694580, -0.02978890389204025269, -0.02141807973384857178};
