/*
 * author : shtsno24
 * Date : 2019-10-18 23:26:07.480643
 *
 */
#pragma once
#include <cstdint>
#include <vector>

using namespace std;

const vector< uint16_t> shape_Conv2D_2_w = {8, 8, 3, 3};
const vector< vector< vector< vector< float> > > > Conv2D_2_w =
{{{{-0.01051039434969425201, -0.11328876018524169922, -0.01567168720066547394},
{0.08128318190574645996, 0.10669900476932525635, -0.12641195952892303467},
{-0.05063766986131668091, 0.11654982715845108032, -0.02077138051390647888}},

{{0.17004816234111785889, 0.05981669574975967407, -0.02931497991085052490},
{-0.04388301447033882141, 0.07661087065935134888, 0.03121014684438705444},
{-0.00753609277307987213, 0.18578775227069854736, 0.14341242611408233643}},

{{0.07250674813985824585, 0.01759067177772521973, -0.08504812419414520264},
{0.17765858769416809082, -0.14482927322387695312, -0.08689717203378677368},
{0.06925706565380096436, 0.10770107060670852661, -0.18301975727081298828}},

{{-0.11964210867881774902, -0.14393150806427001953, -0.18643102049827575684},
{0.16325940191745758057, 0.09962098300457000732, 0.17744764685630798340},
{0.04854774847626686096, -0.16578567028045654297, -0.20475038886070251465}},

{{-0.15260016918182373047, 0.08375534415245056152, 0.03792677819728851318},
{0.02393737062811851501, -0.01130547374486923218, 0.00047120964154601097},
{0.09635983407497406006, 0.13823662698268890381, -0.13534873723983764648}},

{{0.01364438328891992569, -0.15040919184684753418, 0.18805724382400512695},
{0.02582114376127719879, 0.18660280108451843262, 0.13865308463573455811},
{-0.15545944869518280029, -0.16742072999477386475, -0.01538458559662103653}},

{{0.14595450460910797119, 0.13007766008377075195, 0.17728760838508605957},
{-0.07154492288827896118, -0.17105710506439208984, -0.11184948682785034180},
{-0.01032586582005023956, -0.20441709458827972412, -0.03739241883158683777}},

{{0.09377737343311309814, -0.15255741775035858154, 0.15902763605117797852},
{-0.13539263606071472168, -0.05292579159140586853, 0.01005410123616456985},
{-0.17488333582878112793, -0.05717761814594268799, -0.03952586650848388672}}},

{{{0.01540387235581874847, -0.05206128954887390137, 0.16502901911735534668},
{-0.12819901108741760254, 0.09897742420434951782, -0.02019669301807880402},
{-0.10191505402326583862, -0.03848566114902496338, -0.03192284703254699707}},

{{-0.09744576364755630493, -0.11441279947757720947, -0.06195036321878433228},
{0.01886301487684249878, -0.13174852728843688965, -0.08804582804441452026},
{-0.00393999041989445686, 0.03031973354518413544, -0.06712237000465393066}},

{{0.12257277220487594604, -0.00990599486976861954, 0.19688642024993896484},
{-0.18549406528472900391, -0.17665085196495056152, -0.17251610755920410156},
{-0.07919731736183166504, 0.10021365433931350708, 0.02131795883178710938}},

{{0.05879301205277442932, -0.04140287265181541443, -0.03127026185393333435},
{-0.20263832807540893555, 0.14837592840194702148, 0.16472432017326354980},
{0.08573722094297409058, 0.05467176064848899841, 0.10202613472938537598}},

{{0.04283655807375907898, -0.17532344162464141846, 0.02286113984882831573},
{0.13950927555561065674, 0.17756813764572143555, 0.17899414896965026855},
{0.11375376582145690918, 0.06695294380187988281, -0.02351166494190692902}},

{{0.14636524021625518799, 0.08959787338972091675, 0.18217897415161132812},
{-0.09513178467750549316, 0.00856007542461156845, 0.05354421213269233704},
{-0.12092246115207672119, 0.02439182437956333160, 0.10382843017578125000}},

{{0.13896529376506805420, 0.06851010024547576904, 0.01978364400565624237},
{0.10648802667856216431, -0.01858087629079818726, -0.20342482626438140869},
{-0.11965187638998031616, -0.04516511037945747375, 0.10076518356800079346}},

{{0.05683837458491325378, -0.06183708459138870239, 0.06555411964654922485},
{0.14319720864295959473, -0.20239312946796417236, -0.16118051111698150635},
{-0.05756428465247154236, -0.17470605671405792236, -0.16309653222560882568}}},

{{{0.04202257841825485229, -0.07925347983837127686, 0.14447452127933502197},
{0.02053116075694561005, 0.13634169101715087891, -0.15093347430229187012},
{0.06860291212797164917, 0.05628620833158493042, 0.00857472047209739685}},

{{0.01798882707953453064, 0.14558382332324981689, -0.05707900598645210266},
{0.06010057777166366577, 0.25703653693199157715, 0.08964846283197402954},
{-0.19260725378990173340, -0.04368405789136886597, -0.00335895689204335213}},

{{0.15434476733207702637, -0.14813737571239471436, -0.00900678895413875580},
{0.08801822364330291748, -0.06646040081977844238, 0.10198429226875305176},
{-0.11120998114347457886, 0.05246210098266601562, 0.11994702368974685669}},

{{0.12441854178905487061, 0.03366388007998466492, -0.01291204802691936493},
{-0.11836311221122741699, -0.15273202955722808838, -0.02697933651506900787},
{-0.02111724577844142914, 0.06777551025152206421, 0.03836617618799209595}},

{{-0.12187261134386062622, -0.20599393546581268311, -0.06829424202442169189},
{0.14086015522480010986, 0.16691634058952331543, -0.19790540635585784912},
{0.00537103461101651192, -0.01916008070111274719, -0.02339793741703033447}},

{{0.04932324960827827454, -0.04697482287883758545, -0.04353249073028564453},
{0.17288462817668914795, -0.11419098824262619019, 0.11979037523269653320},
{0.15586866438388824463, -0.00200889352709054947, -0.02156476117670536041}},

{{0.13397644460201263428, 0.09925517439842224121, 0.17204575240612030029},
{-0.09435351192951202393, -0.09406021237373352051, 0.09379146993160247803},
{-0.13405220210552215576, -0.09761130064725875854, -0.09415820240974426270}},

{{-0.14095132052898406982, -0.16316549479961395264, 0.09588008373975753784},
{0.19981183111667633057, 0.19323308765888214111, -0.13705709576606750488},
{-0.05187631025910377502, 0.01299092825502157211, -0.14118136465549468994}}},

{{{-0.03388597443699836731, 0.06554775685071945190, -0.06961783021688461304},
{-0.12542016804218292236, -0.02714446000754833221, -0.09279932081699371338},
{-0.17282627522945404053, 0.04951312392950057983, -0.11026001721620559692}},

{{0.20245362818241119385, 0.09709957242012023926, -0.14954848587512969971},
{0.16513590514659881592, -0.04320621117949485779, -0.12711460888385772705},
{0.04323001950979232788, -0.02899229712784290314, 0.15897548198699951172}},

{{-0.07041632384061813354, -0.12577490508556365967, 0.04607550799846649170},
{-0.08400928229093551636, 0.05992970243096351624, 0.09577163308858871460},
{0.20464158058166503906, 0.16930601000785827637, 0.19007217884063720703}},

{{-0.02952303923666477203, 0.12197739630937576294, 0.16810263693332672119},
{0.06101711839437484741, 0.13301406800746917725, 0.09266266226768493652},
{0.15178863704204559326, 0.04141142219305038452, 0.14888620376586914062}},

{{0.11424277722835540771, 0.03139171376824378967, 0.10649666935205459595},
{0.01063541322946548462, -0.05201661959290504456, -0.10584120452404022217},
{-0.17128862440586090088, -0.15546484291553497314, 0.10768201202154159546}},

{{0.06126401945948600769, -0.02999795600771903992, 0.02498776093125343323},
{0.17602610588073730469, 0.09922348707914352417, 0.13351662456989288330},
{0.03384304419159889221, -0.00866345688700675964, 0.13226039707660675049}},

{{-0.10648699104785919189, 0.09497107565402984619, -0.05213749408721923828},
{0.14745430648326873779, 0.20029586553573608398, -0.13980521261692047119},
{0.09369113296270370483, -0.02790619432926177979, -0.13535943627357482910}},

{{0.18119169771671295166, -0.10831142961978912354, -0.02177685126662254333},
{0.20806129276752471924, -0.18262146413326263428, -0.13312341272830963135},
{0.05741368606686592102, -0.00552792195230722427, 0.08861703425645828247}}},

{{{0.21299237012863159180, 0.09235832840204238892, 0.07226838916540145874},
{-0.04228540882468223572, 0.10074602067470550537, -0.14816702902317047119},
{0.12134805321693420410, -0.06036099791526794434, -0.16488899290561676025}},

{{-0.04456367343664169312, 0.20211812853813171387, -0.16314651072025299072},
{-0.05279998108744621277, 0.11050227284431457520, 0.05883353576064109802},
{0.12540560960769653320, -0.08624778687953948975, 0.03205573186278343201}},

{{0.01543958205729722977, 0.19085863232612609863, -0.06392649561166763306},
{0.02230143547058105469, -0.07203523814678192139, 0.00752066168934106827},
{-0.01448190957307815552, 0.01689806580543518066, 0.03807748854160308838}},

{{0.01008486375212669373, 0.08789034932851791382, -0.06984210759401321411},
{-0.18092003464698791504, -0.14360636472702026367, -0.18302489817142486572},
{-0.04043041169643402100, 0.17610149085521697998, 0.14280208945274353027}},

{{0.02411182783544063568, 0.08242860436439514160, 0.18455417454242706299},
{-0.02509818412363529205, -0.16130280494689941406, 0.17343024909496307373},
{0.00401838868856430054, -0.02403211034834384918, -0.07673432677984237671}},

{{0.09636501967906951904, -0.18246546387672424316, -0.00261842366307973862},
{-0.14112988114356994629, -0.13142444193363189697, 0.07047258317470550537},
{-0.08207713812589645386, 0.08745522797107696533, 0.05987905710935592651}},

{{-0.12014064937829971313, 0.06838333606719970703, -0.15131098031997680664},
{0.07913344353437423706, -0.02235165424644947052, 0.17732390761375427246},
{0.12133076041936874390, -0.18248890340328216553, -0.12273100763559341431}},

{{0.12209186702966690063, 0.15109018981456756592, -0.07082413882017135620},
{0.08945965021848678589, -0.14014415442943572998, 0.09537865221500396729},
{0.01428749039769172668, 0.09917391091585159302, -0.12417428940534591675}}},

{{{-0.00952331069856882095, 0.05989594757556915283, 0.03785759210586547852},
{0.20248882472515106201, -0.06565470993518829346, 0.03705794364213943481},
{0.13061001896858215332, 0.17818138003349304199, 0.14319342374801635742}},

{{0.04308078438043594360, -0.14762173593044281006, 0.16808968782424926758},
{-0.06356511265039443970, 0.14650960266590118408, 0.11384408921003341675},
{-0.18114523589611053467, 0.06342571228742599487, -0.02598868496716022491}},

{{-0.17471775412559509277, -0.20978848636150360107, 0.02037856914103031158},
{0.10825202614068984985, -0.02269826270639896393, 0.09619472175836563110},
{0.03317178040742874146, -0.07310383766889572144, -0.02139661461114883423}},

{{0.12987354397773742676, -0.11068408191204071045, 0.08692350983619689941},
{-0.05258192867040634155, -0.00913097523152828217, 0.18122558295726776123},
{-0.04655675590038299561, -0.00638397270813584328, -0.14949616789817810059}},

{{0.04308834671974182129, -0.01981068961322307587, 0.19439597427845001221},
{-0.16404588520526885986, -0.09679170697927474976, 0.08092684298753738403},
{-0.07421238720417022705, -0.03844296559691429138, 0.19394469261169433594}},

{{0.02993977256119251251, -0.02286607027053833008, -0.11165345460176467896},
{-0.03534159436821937561, 0.18635675311088562012, 0.06694219261407852173},
{0.04305826500058174133, 0.15960140526294708252, 0.04414104297757148743}},

{{-0.07781817764043807983, 0.15064503252506256104, -0.01118763070553541183},
{0.14018760621547698975, -0.18046045303344726562, 0.13895842432975769043},
{-0.09551674872636795044, -0.07203422486782073975, 0.07978980243206024170}},

{{0.12684720754623413086, -0.20140033960342407227, 0.02437015250325202942},
{0.07516225427389144897, 0.03949967399239540100, 0.13442125916481018066},
{0.07398588210344314575, -0.03325391933321952820, 0.03853267803788185120}}},

{{{0.14698058366775512695, -0.11427383124828338623, 0.19100078940391540527},
{-0.13871438801288604736, -0.17840559780597686768, 0.00136899703647941351},
{0.03863602504134178162, 0.18470777571201324463, 0.11274406313896179199}},

{{0.05282131955027580261, 0.20495662093162536621, 0.13651096820831298828},
{0.12098519504070281982, 0.10734067112207412720, -0.04034557566046714783},
{0.18928052484989166260, -0.03537981584668159485, -0.00762340286746621132}},

{{0.12138353288173675537, 0.17583687603473663330, -0.07427589595317840576},
{-0.10754360258579254150, -0.17195390164852142334, -0.15516775846481323242},
{-0.00216018268838524818, -0.11572923511266708374, 0.02177624404430389404}},

{{0.08881495893001556396, -0.10978648066520690918, -0.07582847028970718384},
{-0.04393760114908218384, 0.18025286495685577393, -0.00739506771788001060},
{-0.19127479195594787598, 0.15419290959835052490, 0.06960101425647735596}},

{{-0.08844438940286636353, -0.00400066329166293144, 0.16732072830200195312},
{0.10686042904853820801, 0.16273188591003417969, -0.14509508013725280762},
{-0.09200859814882278442, -0.12112345546483993530, 0.06167947128415107727}},

{{0.15547302365303039551, 0.11661041527986526489, -0.18402419984340667725},
{-0.11259406805038452148, -0.05173301696777343750, 0.07428512722253799438},
{-0.19622437655925750732, 0.04068281874060630798, -0.11723151803016662598}},

{{0.00014552043285220861, -0.15434828400611877441, -0.08837433904409408569},
{0.15163223445415496826, 0.00646060518920421600, -0.04502600803971290588},
{-0.04054537415504455566, 0.07221931219100952148, -0.17175064980983734131}},

{{0.00769937923178076744, 0.10804400593042373657, 0.09166260808706283569},
{-0.00171024654991924763, 0.28405296802520751953, 0.07946906983852386475},
{-0.11891177296638488770, -0.17066246271133422852, 0.11360573023557662964}}},

{{{0.04865403473377227783, 0.03634441643953323364, 0.07098635286092758179},
{-0.22585687041282653809, 0.17930912971496582031, 0.06638069450855255127},
{-0.14570143818855285645, 0.10901255160570144653, -0.00522449566051363945}},

{{-0.07449699193239212036, 0.05605579167604446411, -0.03481240198016166687},
{-0.03504700213670730591, 0.16025964915752410889, 0.11656057089567184448},
{-0.06217519938945770264, -0.00187561172060668468, -0.10801143944263458252}},

{{-0.13526344299316406250, -0.02934800088405609131, 0.05879740044474601746},
{0.02340369485318660736, -0.04304688796401023865, 0.16675496101379394531},
{0.08603008091449737549, 0.16318003833293914795, -0.08699556440114974976}},

{{-0.07454124093055725098, -0.03053647652268409729, -0.12993086874485015869},
{-0.14423428475856781006, -0.07275083661079406738, 0.13969133794307708740},
{0.13440950214862823486, -0.01223003864288330078, 0.01834214664995670319}},

{{0.12022495269775390625, 0.18335439264774322510, -0.06382349133491516113},
{0.01235944125801324844, -0.06436862796545028687, 0.18101523816585540771},
{0.09906634688377380371, 0.04962225630879402161, -0.09655639529228210449}},

{{-0.01808640919625759125, 0.10045135766267776489, -0.14278931915760040283},
{-0.10723619163036346436, 0.09161861985921859741, -0.10623811185359954834},
{-0.07538251578807830811, -0.20029965043067932129, 0.01043881848454475403}},

{{0.18586377799510955811, 0.08884976059198379517, -0.02448416687548160553},
{0.14868535101413726807, 0.12363453209400177002, -0.10093258321285247803},
{0.14871676266193389893, 0.17356565594673156738, -0.03213327378034591675}},

{{0.09726549685001373291, -0.12885399162769317627, 0.09412573277950286865},
{-0.13956557214260101318, -0.12084616720676422119, -0.03780424594879150391},
{0.17937511205673217773, -0.05227326601743698120, 0.09392558783292770386}}}};

const uint16_t shape_Conv2D_2_b = 8;
const vector< float> Conv2D_2_b = {0.00582104874774813652, -0.02080174908041954041, -0.02099752798676490784, 0.04473752528429031372, -0.02185722813010215759, -0.03148205578327178955, -0.06129772588610649109, 0.03640363737940788269};
