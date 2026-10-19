import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const ROOT=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const W=2000,H=1320;
const C={ink:'#263442',muted:'#5D6A76',blue:'#E9F2FA',blueEdge:'#749FC3',green:'#ECF5E9',greenEdge:'#83A575',purple:'#F2EBF8',cond:'#8863A5',red:'#FCEDEC',redEdge:'#C98C88',yellow:'#FFF4D7',gold:'#C4A153',orange:'#FFF0E0',orangeEdge:'#D3A372',white:'#FFFFFF',gray:'#E8EDF0',teal:'#4E8F93'};
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
let P=[];const modules=[];
function rect(x,y,w,h,fill=C.white,stroke=C.ink,r=8,sw=1.6){P.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`);}
function text(x,y,s,size=28,{anchor='middle',bold=false,color=C.ink,maxWidth=0,math=false}={}){size=Math.max(28,size);String(s).split('\n').forEach((line,i)=>P.push(`<text x="${x}" y="${y+i*size*1.2}" text-anchor="${anchor}" font-family="${math?'Cambria Math, Cambria':'Arial'}, sans-serif" font-size="${size}" fill="${color}"${bold?' font-weight="700"':''}${maxWidth?` data-max-width="${maxWidth}"`:''}>${esc(line)}</text>`));}
function line(points,{color=C.ink,width=2.1,dash=false,arrow=true}={}){P.push(`<polyline points="${points.map(p=>p.join(',')).join(' ')}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linejoin="round" stroke-linecap="round"${dash?' stroke-dasharray="7 5"':''}/>`);if(arrow){const [x,y]=points.at(-1),[a,b]=points.at(-2),t=Math.atan2(y-b,x-a),len=8;P.push(`<polygon points="${x},${y} ${x-len*Math.cos(t)+3.8*Math.sin(t)},${y-len*Math.sin(t)-3.8*Math.cos(t)} ${x-len*Math.cos(t)-3.8*Math.sin(t)},${y-len*Math.sin(t)+3.8*Math.cos(t)}" fill="${color}"/>`);}}
function circle(x,y,r,fill=C.white,stroke=C.ink,sw=1.6){P.push(`<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`);}
function op(x,y,s,r=15){circle(x,y,r);text(x,y+7,s,24);}
function tag(x,y,s,w=52,color=C.cond){rect(x,y,w,34,C.white,color,5,1.3);text(x+w/2,y+27,s,28,{color});}
function vec(x,y,n=6,cell=17,fill=C.blue){for(let i=0;i<n;i++)rect(x+i*cell,y,cell,19,Array.isArray(fill)?fill[i%fill.length]:fill,C.ink,0,.8);}
function panel(x,y,w,h,title,fill,edge){rect(x,y,w,h,fill,edge,13,1.8);text(w<300?x+w/2:x+18,y+36,title,w<300?28:29,{anchor:w<300?'middle':'start',bold:true,maxWidth:w-(w<300?12:36)});}
function group(id,x,y,w,h,fn){const start=P.length;P.push(`<g id="${id}">`);fn();P.push('</g>');modules.push({id,x,y,w,h,svg:P.slice(start).join('\n')});}
function document(body,w=W,h=H,title='ALFNet architecture'){return `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}"><title>${esc(title)}</title><desc>Editable vector primitives. Schematic curves only; no dataset access. Architecture verified against the selected ALFNet implementation.</desc><rect width="100%" height="100%" fill="white"/>${body}</svg>`;}

function containers(){
 panel(24,40,242,265,'Visible history',C.white,C.blueEdge);
 panel(24,335,242,470,'Lifecycle',C.purple,C.cond);
 panel(24,835,242,345,'Static graph',C.white,C.blueEdge);
 rect(300,40,318,1140,C.red,C.redEdge,13,1.8);text(459,79,'Support-Aware',29,{bold:true});text(459,113,'History Completion',29,{bold:true});
 panel(660,40,1316,455,'ALFTemporal  ·  Spatial-context temporal conditioning',C.green,C.greenEdge);
 panel(660,570,1316,610,'Graph forecasting path',C.blue,C.blueEdge);
}
function skeleton(){
 text(145,180,'12-step curves',27);text(145,560,'Rₜ, aₜ, Mₜ',27);text(145,1000,'A + structure',27);
 ['BiGRIL','Keep observations','Interpolation*','Centering*','Filled history'].forEach((s,i)=>{rect(323,200+i*180,272,90);text(459,255+i*180,s,28);if(i<4)line([[459,290+i*180],[459,380+i*180]]);});
 ['Embedding','Partition interaction','Z / FiLM head'].forEach((s,i)=>{rect(690+i*420,180,380,200);text(880+i*420,285,s,28);if(i<2)line([[1070+i*420,280],[1110+i*420,280]]);});
 const items=[[690,160,'Encoder'],[880,180,'FiLM'],[1090,210,'Decoder'],[1330,135,'Restore'],[1500,280,'Refinement'],[1810,140,'Forecast']];
 items.forEach(([x,w,s],i)=>{rect(x,660,w,120);text(x+w/2,730,s,24);if(i<5)line([[x+w,720],[items[i+1][0],720]]);});
 rect(690,850,780,280);text(1080,1000,'One embedded cell / propagation definition',28);
 line([[618,1060],[638,1060],[638,280],[690,280]]);line([[638,720],[690,720]]);line([[1760,380],[1760,530],[970,530],[970,660]]);
}

function box(x,y,w,h,label,fill=C.white,size=24){rect(x,y,w,h,fill,C.ink,6,1.3);text(x+w/2,y+h/2+size*.35,label,size,{maxWidth:w-12});}
function grid(x,y,rows,cols,cell=19,highlight=-1){for(let r=0;r<rows;r++)for(let c=0;c<cols;c++)rect(x+c*cell,y+r*cell,cell,cell,r===highlight?'#B9D9AB':((r+c)%3===0?'#DBE7D8':C.white),C.greenEdge,0,.7);}
function curve(x,y,w,h,missing=[],color=C.teal){const values=[.3,.45,.37,.61,.78,.56,.48,.68,.82,.63,.52,.7];const pts=values.map((v,i)=>[x+i*w/11,y+h*(1-v)]);line([[x,y+h+7],[x+w+3,y+h+7]],{arrow:false,color:'#A8B2BA',width:1});for(let i=0;i<11;i++)if(!missing.includes(i)&&!missing.includes(i+1))line([pts[i],pts[i+1]],{arrow:false,color,width:2.4});pts.forEach(([a,b],i)=>{if(!missing.includes(i))circle(a,b,2.8,color,color,.5);else circle(a,y+h+7,2.8,C.white,'#A8B2BA',.8);});}
function inputs(){
 group('01_visible_history',24,40,242,265,()=>{
 panel(24,40,242,265,'Visible history',C.white,C.blueEdge);
 curve(57,98,175,44,[3,4]);curve(57,165,175,44,[0,1,2,3,4]);
 text(55,244,'t−11',21,{anchor:'start'});text(235,244,'t',21,{anchor:'end'});
 text(145,281,'X · 12 × N × 1',23,{math:true});
 });
 group('02_lifecycle_context',24,335,242,470,()=>{
 panel(24,335,242,470,'Lifecycle context',C.purple,C.cond);
 const names=['Unseen','Steady','Join','Fault'],colors=['#C4BAD8','#9BBFCB','#C1D6A0','#E5B6AA'];
 names.forEach((n,r)=>{const y=407+r*48;text(42,y+15,n,25,{anchor:'start'});for(let t=0;t<12;t++){const role=r===0?0:r===1?1:r===2?(t<6?0:2):(t<6?1:3);rect(134+t*8.7,y,8.7,19,colors[role],C.white,0,.8);}});
 text(145,608,'Rₜ  ·  role trajectory',24);text(83,647,'aₜ',27,{math:true});
 for(let t=0;t<12;t++)rect(112+t*10,653-(t<6?1:(t-5)*3),8,t<6?1:(t-5)*3,C.cond,C.cond,0,.3);
 text(42,704,'Mₜ',24,{anchor:'start',math:true});for(let r=0;r<2;r++)for(let c=0;c<8;c++)rect(89+c*18,678+r*18,18,18,(c<3&&r===1)||c===5?C.white:C.cond,C.white,0,1);
 text(145,754,'Sₜ = [Rₜ, aₜ, Mₜ]',26,{math:true});text(145,785,'4 + 1 + 1 channels',22);
 });
 group('03_static_graph',24,835,242,345,()=>{
 panel(24,835,242,345,'Static graph',C.white,C.blueEdge);
 const pts=[[65,940],[120,909],[169,947],[223,914],[91,1000],[199,1005]];
 [[0,1],[0,4],[1,2],[2,3],[2,4],[2,5],[4,5]].forEach(([a,b])=>line([pts[a],pts[b]],{arrow:false,color:C.blueEdge,width:2}));pts.forEach(([x,y])=>circle(x,y,9,C.blue,C.blueEdge,1.8));
 const edges=[[0,1],[0,4],[1,2],[2,3],[2,4],[2,5],[4,5]];
 for(let r=0;r<6;r++)for(let c=0;c<6;c++)rect(57+c*12,1048+r*12,12,12,edges.some(([a,b])=>(a===r&&b===c)||(a===c&&b===r))?C.blueEdge:C.white,C.blueEdge,0,.7);
 text(173,1083,'A',30,{math:true});text(145,1156,'Graph attributes',21);
 });
 tag(269,141,'X',27,C.ink);
 line([[266,456],[269,456]],{color:C.cond,dash:true,arrow:false});tag(269,441,'S',28);
 line([[266,1015],[269,1015]],{color:C.muted,arrow:false});tag(269,1000,'A',28,C.muted);
 line([[266,1117],[269,1117]],{color:C.muted,arrow:false});tag(269,1100,'G',28,C.muted);
}
function completion(){
 group('04_shared_completion',300,40,318,1140,()=>{
 rect(300,40,318,1140,C.red,C.redEdge,13,1.8);text(459,79,'Support-Aware',28,{bold:true});text(459,113,'History Completion',28,{bold:true});
 group('04a_bidirectional_reconstruction',323,145,272,302,()=>{
 rect(323,145,272,302,C.white,C.redEdge,8,1.3);text(459,177,'GRIN / BiGRIL',27,{bold:true});
 tag(332,195,'R, a',55);tag(396,195,'M',37);tag(442,195,'A',34,C.muted);
 [0,1].forEach(r=>{const y=245+r*70;for(let i=0;i<3;i++){box(341+i*80,y,55,37,'GR',C.red,24);if(i<2)line(r===0?[[396+i*80,y+18],[421+i*80,y+18]]:[[421+i*80,y+18],[396+i*80,y+18]],{width:1.5});} });
 line([[556,282],[584,282],[584,368],[501,368],[501,381]]);line([[341,334],[332,334],[332,368],[419,368],[419,381]]);
 box(360,379,211,38,'Align + merge',C.red);text(459,440,'All 12 steps',19);
 });
 line([[459,447],[459,480]]);
 group('04b_observation_selection',323,480,272,110,()=>{box(323,480,272,69,'Keep observations',C.white,25);text(459,579,'M: keep / fill',21,{math:true});});
 line([[459,590],[459,617]]);
 group('04c_window_interpolation',323,617,272,162,()=>{rect(323,617,272,162,C.white,C.redEdge,8,1.3);text(459,649,'Interpolation*',26,{bold:true});
 curve(350,666,210,46,[4,5,6],C.teal);line([[407.3,683.9],[483.6,680.7]],{color:C.redEdge,dash:true,arrow:false,width:2});
 text(459,758,'Missing slots; q > 0',22);});
 line([[459,779],[459,819]]);group('04d_reference_centering',323,819,272,174,()=>{box(323,819,272,87,'Center by αv*',C.white,24);text(459,939,'X̄ = Xfilled − α v',27,{math:true});
 tag(532,963,'αv',48,C.ink);line([[532,978],[516,978],[516,956]],{arrow:false,width:1.5});});
 line([[459,955],[459,1012]]);
 group('04e_shared_filled_history',323,1012,272,127,()=>{rect(323,1012,272,127,C.white,C.redEdge,8,1.3);text(459,1045,'Shared history',25,{bold:true});vec(363,1064,8,24,[C.red,'#EED1CD']);text(459,1122,'X̄  ·  12 × N × 1',25,{math:true});});
 text(459,1166,'q = 0: retain GRIN',24);
 });
}
function temporal(){
 panel(660,40,1316,455,'ALFTemporal  ·  Spatial-context temporal conditioning',C.green,C.greenEdge);
 group('05_history_calendar_structure',684,108,306,363,()=>{
 rect(684,108,306,363,C.white,C.greenEdge,8,1.3);
 text(837,139,'History embedding',26,{bold:true});
 tag(696,154,'S',30);line([[726,169],[758,169]],{color:C.cond,dash:true});text(855,178,'[X̄, S, calendar]',24,{math:true});
 line([[837,186],[837,205]]);box(709,205,246,42,'Conv (12) → 64',C.green,24);
 text(837,277,'1 temporal patch',25,{bold:true});
 box(699,296,158,51,'ToD / DoW',C.green,24);text(921,329,'32 + 32',24);
 // The thin bar is concatenation, not summation. All three feature groups enter it.
 box(699,359,158,48,'Struct MLP',C.green,24);text(920,390,'32',24);tag(699,418,'G',30,C.muted);
 line([[714,418],[714,407]],{color:C.muted,width:1.4});
 rect(972,215,7,213,'#BCD5AC',C.greenEdge,0,1);
 line([[955,226],[972,226]],{width:1.4});line([[857,340],[972,340]],{width:1.4});line([[857,400],[972,400]],{width:1.4});text(854,451,'Concat → 160',25,{bold:true});
 });
 group('06_cross_partition_attention',1018,108,320,363,()=>{
 rect(1018,108,320,363,C.white,C.greenEdge,8,1.3);text(1178,139,'Spatial partitions',26,{bold:true});text(1178,170,'Reorder / pad · P = 32',23);
 grid(1055,199,4,6,28,1);text(1250,258,'…',27);text(1036,266,'C',25,{math:true});text(1250,215,'P',23,{math:true});text(1178,347,'Cross-partition',27,{bold:true});text(1178,380,'attention',27,{bold:true});
 line([[1069,240],[1209,240]],{color:'#3A704A',width:2.5});line([[1209,240],[1069,240]],{color:'#3A704A',width:2.5});text(1178,416,'Same slot across P',25);text(1178,450,'C slots / partition',25);
 });
 group('07_within_partition_attention',1366,108,310,363,()=>{
 rect(1366,108,310,363,C.white,C.greenEdge,8,1.3);text(1521,139,'Within-partition',25,{bold:true});text(1521,169,'interaction',25,{bold:true});
 vec(1405,198,6,36,['#E3EEDC','#CAE0BB']);
 line([[1414,217],[1385,217],[1385,302],[1434,302]],{color:C.greenEdge});line([[1612,217],[1654,217],[1654,302],[1595,302]],{color:C.greenEdge});text(1521,265,'Gather C slots',23);
 for(let i=0;i<4;i++)rect(1434+i*44,302,29,29,C.greenEdge,C.ink,3,1);
 text(1521,365,'4 latent tokens',23,{bold:true});
 line([[1434,331],[1385,331],[1385,434],[1405,434]],{color:C.greenEdge});line([[1595,331],[1654,331],[1654,434],[1621,434]],{color:C.greenEdge});text(1521,401,'Broadcast',22);vec(1405,425,6,36,['#C1D7B2','#DBE9D3']);
 });
 group('08_features_and_film_head',1704,108,248,363,()=>{
 box(1704,108,248,64,'Restore order',C.white,23);line([[1828,172],[1828,205]]);
 vec(1740,205,8,22,C.green);text(1828,256,'Z ∈ ℝᴺ×¹⁶⁰',28,{math:true});
 line([[1828,271],[1828,291]]);box(1704,291,248,64,'FiLM head',C.yellow,27);text(1828,385,'160 → 32 → 128',24);
 vec(1767,405,6,22,C.yellow);vec(1767,434,6,22,C.orange);text(1737,422,'Δγ',22,{math:true});text(1737,451,'β',23,{math:true});
 line([[1952,323],[1963,323],[1963,460],[1828,460],[1828,471]],{width:1.5});
 });
 line([[979,270],[1018,270]]);line([[1338,270],[1366,270]]);line([[1621,443],[1691,443],[1691,140],[1704,140]]);
 text(1340,486,'One attention block · residual, norm and FFN',20);
 text(1292,522,'Δγ, β  ·  N × 64 each',25,{math:true});
}
function graphPath(){
 rect(660,570,1316,610,C.blue,C.blueEdge,13,1.8);
 text(686,613,'ALFGraph',29,{anchor:'start',bold:true});text(1495,600,'Graph forecasting path',26,{anchor:'start',bold:true});
 group('09_graph_encoder',686,650,174,170,()=>{
 rect(686,650,174,170,C.white,C.blueEdge,8,1.3);text(773,682,'Encoder',27,{bold:true});box(705,700,136,37,'Cell · L2',C.blue,22);box(705,751,136,37,'Cell · L1',C.blue,22);line([[773,751],[773,737]],{width:1.4});text(773,811,'hidden = 64',20);
 });
 group('10_film_modulation',886,650,174,170,()=>{
 rect(886,650,174,170,C.yellow,C.gold,8,1.3);text(973,681,'FiLM',28,{bold:true});op(930,734,'⊙');op(1018,734,'+');line([[945,734],[1003,734]]);line([[886,720],[908,720],[908,734],[915,734]],{width:1.4});line([[1033,734],[1045,734],[1045,720],[1060,720]],{width:1.4});text(930,710,'1+Δγ',21,{math:true});text(1018,710,'β',22,{math:true});text(973,778,'Terminal H',22);text(973,809,'both layers',22);
 });
 group('11_residual_ar_decoder',1086,650,216,170,()=>{
 rect(1086,650,216,170,C.white,C.blueEdge,8,1.3);text(1194,677,'Residual AR',26,{bold:true});text(1194,710,'decoder',26,{bold:true});box(1102,718,83,30,'L1',C.blue,24);box(1202,718,83,30,'L2',C.blue,24);line([[1185,733],[1202,733]],{width:1.4});
 line([[1243,748],[1243,758]],{width:1.4});box(1103,758,181,33,'yₖ₋₁ + d·Δyₖ',C.yellow,24);
 line([[1284,775],[1294,775],[1294,803],[1095,803],[1095,734],[1102,734]],{width:1.4});line([[1294,775],[1294,720],[1302,720]],{width:1.4});
 });
 group('12_reference_restoration',1328,650,136,170,()=>{
 rect(1328,650,136,170,C.white,C.blueEdge,8,1.3);text(1396,676,'Restore',24,{bold:true});text(1396,710,'reference*',22);op(1396,755,'+');tag(1372,784,'αv',48,C.ink);
 line([[1328,720],[1355,720],[1355,755],[1381,755]],{width:1.4});line([[1396,784],[1396,770]],{width:1.4});line([[1411,755],[1448,755],[1448,720],[1464,720]],{width:1.4});
 });
 line([[860,720],[886,720]]);line([[1060,720],[1086,720]]);line([[1302,720],[1328,720]]);line([[1464,720],[1490,720]]);
 tag(699,830,'Sₜ',45);line([[721,830],[721,820]],{color:C.cond,dash:true});tag(1105,830,'Sᴸ',45);line([[1127,830],[1127,820]],{color:C.cond,dash:true});tag(1170,830,'X̄ᴸ',55,C.ink);line([[1197,830],[1197,820]],{width:1.5});
 // Definition leader lines have no arrowheads and do not belong to the forward path.
 line([[820,820],[820,865],[830,865],[830,882]],{arrow:false,color:C.blueEdge,width:1.2});line([[1267,820],[1267,866],[1320,866],[1320,882]],{arrow:false,color:C.blueEdge,width:1.2});
 group('13_lifecycle_graph_cell',686,882,778,278,()=>{
 rect(686,882,778,278,C.white,C.blueEdge,8,1.3);text(1075,915,'DCRNN cell · lifecycle-conditioned propagation',24,{bold:true});
 tag(701,936,'S',30);line([[731,951],[750,951]],{color:C.cond,dash:true});box(750,933,277,37,'Linear → Softmax',C.purple,21);text(1083,959,'w₁…w₅',23,{math:true});
 const labs=['Identity','Mean','W.Mean','Forward','Backward'];
 labs.forEach((s,i)=>{const x=701+i*89;rect(x,990,79,69,C.blue,C.blueEdge,5,1);text(x+39.5,1015,['I','Mean','WM','Fwd','Bwd'][i],i===2?20:24);const patterns=[[1,0,0,0,1,0],[0,1,1,1,0,0],[0,.4,1,.7,0,0],[0,.4,1,.7,0,0],[0,1,0,.4,0,.7]];for(let r=0;r<2;r++)for(let c=0;c<3;c++){const v=patterns[i][r*3+c];rect(x+22+c*11,1029+r*11,11,11,v===0?C.white:v===1?C.blueEdge:'#BED1E2',C.white,0,.5);}});
 line([[1027,951],[1040,951]],{width:1.5});line([[1127,951],[1145,951],[1145,979],[740,979]],{color:C.cond,dash:true,arrow:false});for(let i=0;i<5;i++)line([[740+i*89,979],[740+i*89,990]],{color:C.cond,dash:true,width:1.3});text(920,1090,'G = Σⱼ wⱼ ⊙ (Pⱼ U Θⱼ) + b',26,{math:true});text(920,1120,'Dense mixture · Pⱼ(A)',24,{math:true});
 line([[1158,937],[1158,1121]],{color:'#B9C7D3',arrow:false,width:1});
 text(1308,961,'Graph-GRU gates',23,{bold:true});
 for(let i=0;i<3;i++){box(1172+i*94,980,85,44,['σ Gᵣ','σ Gᵤ','tanh'][i],C.blue,21);text(1214.5+i*94,1048,['r','u','c'][i],28,{math:true});}
 text(1308,1082,'h′ = u h + (1−u)c',23,{math:true});text(1308,1117,'G꜀ uses r ⊙ h',23,{math:true});
 text(1075,1150,'Encoder Sₜ · decoder Sᴸ · separate gate / layer weights',21);
 });
 group('14_role_scaled_refinement',1490,610,282,550,()=>{
 rect(1490,610,282,550,C.orange,C.orangeEdge,8,1.5);text(1631,645,'Role-Scaled',27,{bold:true});text(1631,678,'Graph Refinement',26,{bold:true});
 text(1531,728,'Y₀',27,{math:true});line([[1552,720],[1575,720],[1575,755]]);tag(1690,703,'A',32,C.muted);line([[1706,737],[1706,755]],{color:C.muted,width:1.4});
 box(1510,755,242,58,'[Y₀, P⁺Y₀, P⁻Y₀]',C.white,26);line([[1631,813],[1631,843]]);
 box(1510,843,242,51,'MLP  3 → 32 → 1',C.white,24);line([[1631,894],[1631,917]]);box(1547,917,168,44,'tanh',C.white,25);
 line([[1631,961],[1631,989]]);box(1520,989,222,46,'× 0.05 s(Rᴸ)',C.white,25);tag(1509,1049,'Rᴸ',46);line([[1555,1063],[1580,1063],[1580,1035]],{color:C.cond,dash:true,width:1.5});
 line([[1631,1035],[1631,1089]]);op(1631,1105,'+');
 line([[1565,720],[1565,738],[1502,738],[1502,1105],[1616,1105]],{width:1.4});circle(1565,720,2.5,C.ink,C.ink,0);text(1703,1144,'+ Y₀',23,{math:true});
 });
 group('15_forecast',1805,681,147,280,()=>{
 text(1878,714,'Forecast',28,{bold:true});text(1878,754,'Ŷ',31,{math:true});curve(1812,788,129,95,[],C.blueEdge);text(1810,923,'t+1',20,{anchor:'start'});text(1950,923,'t+12',20,{anchor:'end'});text(1878,956,'12 × N × 1',21,{math:true});
 });
 line([[1646,1105],[1790,1105],[1790,739],[1805,739]]);
}
function footer(){
 line([[24,1209],[1976,1209]],{arrow:false,color:'#C5CDD2',width:1});
 text(27,1244,'Training only',24,{anchor:'start',bold:true});text(224,1244,'0.9 Lmean + 0.1 Lrole + Lroute',25,{anchor:'start',math:true});text(775,1244,'Masked teacher forcing',23,{anchor:'start'});
 line([[1210,1236],[1257,1236]],{color:C.cond,dash:true});text(1270,1244,'Lifecycle condition',21,{anchor:'start'});tag(1505,1221,'S',29);text(1543,1244,'Named input reference',21,{anchor:'start'});
 text(27,1290,'* Dataset-specific processing. Batch dimension omitted; curves and sampled matrix entries are schematic.',22,{anchor:'start'});
}

inputs();completion();temporal();graphPath();footer();
// Inter-container connectors are drawn last so no container obscures an endpoint.
line([[266,177],[323,177]]);
line([[595,1092],[638,1092],[638,227],[684,227]]);circle(638,770,3,C.ink,C.ink,0);line([[638,770],[686,770]]);
line([[1828,471],[1828,531],[972,531],[972,650]]);
fs.mkdirSync(path.join(ROOT,'qa'),{recursive:true});
fs.writeFileSync(path.join(ROOT,'_source','alfnet_architecture.svg'),document(P.join('\n')));
fs.writeFileSync(path.join(ROOT,'qa','modules.json'),JSON.stringify(modules,null,2));
// Extract only after the complete figure has been laid out and verified.
// Each asset is an editable group of SVG text and basic geometry, with no raster image.
const moduleDir=path.join(ROOT,'module_library');fs.mkdirSync(moduleDir,{recursive:true});
const exported=modules.filter(m=>m.id!=='04_shared_completion');
const svgAsset=(body,w,h,title)=>`<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}"><title>${esc(title)}</title>${body}</svg>`;
const board=[];let bx=24,by=100,rowH=0;const boardW=2000,gap=30,pad=18;
for(const m of exported){
 const w=m.w+pad*2,h=m.h+pad*2;
 const asset=`<g transform="translate(${pad-m.x} ${pad-m.y})">${m.svg}</g>`;
 fs.writeFileSync(path.join(moduleDir,`${m.id}.svg`),svgAsset(asset,w,h,m.id));
 if(bx+w>boardW-24){bx=24;by+=rowH+gap;rowH=0;}
 board.push(`<g transform="translate(${bx} ${by})">${asset}</g>`);
 bx+=w+gap;rowH=Math.max(rowH,h);
}
board.unshift('<text x="42" y="55" font-family="Arial, sans-serif" font-size="34" font-weight="700" fill="#263442">ALFNet · Editable SVG module library</text>');
fs.writeFileSync(path.join(ROOT,'module_library.svg'),document(board.join('\n'),boardW,by+rowH+24,'ALFNet editable module library'));
fs.writeFileSync(path.join(ROOT,'qa','module_index.json'),JSON.stringify(exported.map(m=>({id:m.id,path:`module_library/${m.id}.svg`,width:m.w+pad*2,height:m.h+pad*2})),null,2));
console.log(`Saved architecture SVG and ${exported.length} editable modules`);
