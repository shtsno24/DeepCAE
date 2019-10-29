/*
 * author : shtsno24
 * Date : 2019-10-29 19:58:44.204078
 *
 */
#pragma once
#include <cstdint>
#include <vector>

using namespace std;

const vector< uint16_t> shape_SeparableConv2D_3_w_d = {1, 8, 3, 3};
const vector< vector< vector< vector< float> > > > SeparableConv2D_3_w_d =
{{{{-0.10153078287839889526, 0.23160906136035919189, 0.26070573925971984863},
{-0.04601483419537544250, -0.24562819302082061768, -0.11614263057708740234},
{0.26963573694229125977, 0.26481002569198608398, 0.04479537531733512878}},

{{0.15655784308910369873, 0.22296035289764404297, 0.07139988988637924194},
{0.12279497832059860229, 0.09037029743194580078, -0.18639212846755981445},
{-0.21148377656936645508, -0.17557266354560852051, 0.04187637940049171448}},

{{-0.05122061446309089661, -0.08351156860589981079, 0.24347786605358123779},
{0.27332830429077148438, 0.28606769442558288574, 0.09331614524126052856},
{0.04697773233056068420, 0.15249191224575042725, 0.31453073024749755859}},

{{0.20268736779689788818, 0.26582828164100646973, 0.09379556775093078613},
{0.28731283545494079590, -0.18510936200618743896, -0.12081845849752426147},
{-0.21336913108825683594, -0.22002841532230377197, -0.15703842043876647949}},

{{-0.25141015648841857910, 0.17764936387538909912, -0.25841450691223144531},
{-0.23465199768543243408, 0.17640793323516845703, 0.21533684432506561279},
{0.03912385180592536926, 0.08989164233207702637, 0.19608822464942932129}},

{{-0.21881988644599914551, -0.44770687818527221680, -0.10634738951921463013},
{-0.24307061731815338135, -0.30167716741561889648, -0.42084860801696777344},
{0.10944160073995590210, 0.08253116905689239502, 0.16366775333881378174}},

{{0.30750292539596557617, -0.06286919862031936646, -0.05788666754961013794},
{0.01496137492358684540, 0.27977499365806579590, 0.15828038752079010010},
{0.12002127617597579956, -0.26045325398445129395, 0.16952849924564361572}},

{{0.07957249134778976440, 0.29253485798835754395, 0.24641929566860198975},
{0.04620231688022613525, 0.01691021397709846497, -0.14899326860904693604},
{0.07550566643476486206, 0.32178574800491333008, 0.28088730573654174805}}}};

const vector< uint16_t> shape_SeparableConv2D_3_w_p = {16, 8, 1, 1};
const vector< vector< vector< vector< float> > > > SeparableConv2D_3_w_p =
{{{{0.02721367031335830688}},

{{0.03034634329378604889}},

{{-0.16677825152873992920}},

{{0.44141560792922973633}},

{{-0.40820872783660888672}},

{{0.17397226393222808838}},

{{0.12610584497451782227}},

{{-0.50055265426635742188}}},

{{{0.29915460944175720215}},

{{0.45504868030548095703}},

{{-0.00372851267457008362}},

{{0.43259990215301513672}},

{{0.26637837290763854980}},

{{0.07734179496765136719}},

{{0.32459625601768493652}},

{{-0.35769069194793701172}}},

{{{-0.00843572895973920822}},

{{0.16118888556957244873}},

{{0.23255512118339538574}},

{{-0.08547347038984298706}},

{{0.09859781712293624878}},

{{0.09915202111005783081}},

{{0.09897317737340927124}},

{{0.01168763916939496994}}},

{{{0.27036899328231811523}},

{{-0.27398496866226196289}},

{{-0.07562900334596633911}},

{{-0.08087661862373352051}},

{{0.29410454630851745605}},

{{-0.44177159667015075684}},

{{-0.29743659496307373047}},

{{-0.06528183817863464355}}},

{{{0.21102435886859893799}},

{{-0.24932287633419036865}},

{{0.01803928613662719727}},

{{0.23055016994476318359}},

{{-0.25763973593711853027}},

{{0.41845661401748657227}},

{{-0.36371234059333801270}},

{{0.23607599735260009766}}},

{{{-0.26430588960647583008}},

{{-0.17219974100589752197}},

{{0.61013877391815185547}},

{{-0.13495798408985137939}},

{{0.42050072550773620605}},

{{0.49005183577537536621}},

{{-0.14439542591571807861}},

{{-0.44829416275024414062}}},

{{{-0.15776212513446807861}},

{{-0.13903069496154785156}},

{{0.27193230390548706055}},

{{0.20695745944976806641}},

{{0.07090736180543899536}},

{{0.05729133635759353638}},

{{0.35347607731819152832}},

{{0.06843221932649612427}}},

{{{0.03419995307922363281}},

{{-0.06531815230846405029}},

{{-0.38384419679641723633}},

{{0.25886639952659606934}},

{{0.32125198841094970703}},

{{-0.38539630174636840820}},

{{0.25236418843269348145}},

{{0.13600720465183258057}}},

{{{0.47563648223876953125}},

{{-0.27689540386199951172}},

{{-0.40427672863006591797}},

{{-0.04082143306732177734}},

{{-0.17093205451965332031}},

{{-0.30975055694580078125}},

{{0.38391029834747314453}},

{{0.41016685962677001953}}},

{{{-0.34061449766159057617}},

{{0.23091128468513488770}},

{{0.14208044111728668213}},

{{0.27447444200515747070}},

{{0.30448567867279052734}},

{{-0.13792139291763305664}},

{{0.47989684343338012695}},

{{0.08022017776966094971}}},

{{{0.43371963500976562500}},

{{-0.41110718250274658203}},

{{-0.45296037197113037109}},

{{-0.23607516288757324219}},

{{-0.35123109817504882812}},

{{0.33604872226715087891}},

{{0.34019437432289123535}},

{{-0.10567188262939453125}}},

{{{0.22413265705108642578}},

{{0.05485498905181884766}},

{{-0.49898287653923034668}},

{{-0.41426217555999755859}},

{{-0.23614096641540527344}},

{{-0.00282804598100483418}},

{{-0.39412423968315124512}},

{{0.41768449544906616211}}},

{{{0.35255348682403564453}},

{{0.05598067864775657654}},

{{0.28070956468582153320}},

{{0.00311308819800615311}},

{{0.48457145690917968750}},

{{-0.02715403027832508087}},

{{0.16436970233917236328}},

{{0.49126613140106201172}}},

{{{-0.14701221883296966553}},

{{0.15143963694572448730}},

{{0.40278500318527221680}},

{{0.48920717835426330566}},

{{-0.19786670804023742676}},

{{0.31461316347122192383}},

{{0.36257410049438476562}},

{{0.06935052573680877686}}},

{{{-0.12319360673427581787}},

{{-0.49768653512001037598}},

{{0.48731884360313415527}},

{{-0.42182120680809020996}},

{{-0.03522785007953643799}},

{{0.08998795598745346069}},

{{0.41416719555854797363}},

{{-0.27670079469680786133}}},

{{{-0.41461619734764099121}},

{{-0.41973412036895751953}},

{{-0.10864511877298355103}},

{{-0.34664776921272277832}},

{{-0.26734969019889831543}},

{{0.24602851271629333496}},

{{0.21496833860874176025}},

{{-0.51100164651870727539}}}};

const uint16_t shape_SeparableConv2D_3_b_d = 16;
const vector< float> SeparableConv2D_3_b_d = {0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000};
const uint16_t shape_SeparableConv2D_3_b_p = 16;
const vector< float> SeparableConv2D_3_b_p = {0.13494038581848144531, 0.04992418736219406128, -0.07264840602874755859, 0.09157273918390274048, 0.14677090942859649658, 0.12395865470170974731, 0.04953559488058090210, 0.01511064637452363968, -0.00215048831887543201, -0.05462574958801269531, -0.00152714247815310955, 0.00857718568295240402, -0.00344264088198542595, 0.05645483359694480896, -0.10356052964925765991, 0.15190973877906799316};
