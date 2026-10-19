"""Editable SVG authoring; no datasets, model imports, or raster tracing."""
from pathlib import Path
from html import escape
import math

OUT = Path(__file__).resolve().parent
INK = '#253342'
BLUE = '#e6f2fc'
GREEN = '#edf7e8'
PURPLE = '#f1eafa'
PEACH = '#fcebdc'
YELLOW = '#fff5cf'
COND = '#8c5bbb'
P = []

def start(w, h, title):
    global P
    P = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
         f'<title>{escape(title)}</title>',
         '<desc>ALFNet final implementation. Editable text and vector primitives; no embedded bitmap.</desc>',
         '<rect width="100%" height="100%" fill="white"/>']

def finish(name):
    (OUT / name).write_text('\n'.join(P + ['</svg>']), encoding='utf-8')

def rect(x, y, w, h, fill='white', stroke=INK, r=7, dash=False, sw=1.6):
    P.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'+(' stroke-dasharray="7 5"' if dash else '')+'/>')

def txt(x, y, lines, size=21, anchor='middle', bold=False, color=INK, italic=False):
    if isinstance(lines, str): lines = lines.split('\n')
    for i, line in enumerate(lines):
        P.append(f'<text x="{x}" y="{y+i*size*1.25}" text-anchor="{anchor}" font-family="Times New Roman, Liberation Serif, serif" font-size="{size}" fill="{color}"'+(' font-weight="bold"' if bold else '')+(' font-style="italic"' if italic else '')+'>'+escape(line)+'</text>')

def line(points, color=INK, dash=False, arrow=True, sw=2):
    P.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in points)+f'" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round"'+(' stroke-dasharray="7 5"' if dash else '')+'/>')
    if arrow:
        x,y=points[-1]; a,b=points[-2]; ang=math.atan2(y-b,x-a)
        coords=[(x,y),(x-10*math.cos(ang)+4*math.sin(ang),y-10*math.sin(ang)-4*math.cos(ang)),(x-10*math.cos(ang)-4*math.sin(ang),y-10*math.sin(ang)+4*math.cos(ang))]
        P.append('<polygon points="'+' '.join(f'{a:.2f},{b:.2f}' for a,b in coords)+f'" fill="{color}"/>')

def dot(x,y,rad=3,fill=INK):
    P.append(f'<circle cx="{x}" cy="{y}" r="{rad}" fill="{fill}"/>')

def op(x,y,s):
    P.append(f'<circle cx="{x}" cy="{y}" r="16" fill="white" stroke="{INK}" stroke-width="1.8"/>')
    txt(x,y+7,s,24)

def vector(x,y,colors,n=5,s=18):
    if isinstance(colors,str): colors=[colors]
    for i in range(n): rect(x+i*(s+2),y,s,s,colors[i%len(colors)],r=0,sw=1)

def grid(x,y,cols=6,rows=4,s=15):
    for j in range(rows):
        for i in range(cols):
            fill='#9fcce8' if ((i+j*3)%7 not in [1,2,5]) else '#eef0f2'
            rect(x+i*s,y+j*s,s,s,fill,r=0,sw=.7)

def graph(x,y,scale=1,missing=False):
    pts=[(8,21),(42,8),(69,29),(27,49),(67,63),(92,4)]
    for a,b in [(0,1),(0,3),(1,2),(2,3),(2,4),(2,5),(3,4)]:
        line([(x+pts[a][0]*scale,y+pts[a][1]*scale),(x+pts[b][0]*scale,y+pts[b][1]*scale)],sw=1.2,arrow=False)
    colors=['#9ecae4','#f7c99d','#a9d38e','#ffe39b','#d9c1e6','#a8d8c5']
    for i,(a,b) in enumerate(pts):
        P.append(f'<circle cx="{x+a*scale}" cy="{y+b*scale}" r="{6*scale}" fill="{("#edf0f2" if missing and i in [2,4] else colors[i])}" stroke="{INK}" stroke-width="1.3"/>')

def snapshots(x,y):
    for dx,dy in [(20,-16),(10,-8),(0,0)]:
        rect(x+dx,y+dy,118,87,'white','#aab5bf',r=0,sw=1)
    graph(x+10,y+9,missing=True)

def box(x,y,w,h,label,fill,size=22):
    rect(x,y,w,h,fill)
    lines=label.split('\n');txt(x+w/2,y+h/2-(len(lines)-1)*size*.625+size*.32,lines,size)

def panel(x,y,w,h,label,fill):
    rect(x,y,w,h,fill)
    txt(x+15,y+29,label,24,anchor='start',bold=True)

def tensor(x,y,fill='#b8d9ec',n=4):
    for i in reversed(range(n)):
        dx=i*18
        P.append(f'<polygon points="{x+dx},{y} {x+dx+7},{y-7} {x+dx+20},{y-7} {x+dx+13},{y}" fill="#f5fbff" stroke="{INK}" stroke-width="1"/>')
        P.append(f'<polygon points="{x+dx+13},{y} {x+dx+20},{y-7} {x+dx+20},{y+42} {x+dx+13},{y+49}" fill="{fill}" stroke="{INK}" stroke-width="1"/>')
        rect(x+dx,y,13,49,fill,r=0,sw=1)

def overall():
    panel(20,20,1760,595,'(a) Overall architecture','white')
    # Input history and its observation mask.
    txt(142,126,'Visible traffic history',24,bold=True)
    snapshots(62,165);txt(135,282,'Xvis : 12 historical steps',19)
    grid(82,324,7,4,17);txt(141,418,'Observation mask M',21)
    graph(88,455,.85);txt(141,548,'Static graph A',20)
    # Lifecycle context is explicitly supplied, not a learned global state encoder.
    panel(290,103,365,166,'Lifecycle context  S = [R, a, M]',PURPLE)
    txt(309,164,['R: AAS / NAS / VS / FS','a: state duration','M: observed / missing'],20,anchor='start')
    vector(547,151,['#bca2d6','#ddcdec'],4,17)
    vector(547,185,'#d8c4eb',4,17)
    vector(547,219,['#9fcce8','#eef0f2'],4,17)
    # Shared history reconstruction and per-dataset support handling.
    panel(290,345,190,184,'Shared GRIN',PEACH)
    graph(332,398,.9);txt(385,498,['Preserve observed;','fill missing'],18)
    box(520,381,135,109,'Window\nsupport\nhandling*','#f2f3f4',21)
    line([(215,237),(252,237),(252,434),(290,434)])
    txt(269,398,'Xvis',17)
    line([(480,434),(520,434)])
    line([(655,434),(686,434),(686,230),(720,230)])
    line([(686,434),(720,434)]);dot(686,434)
    txt(687,560,'Same filled history → both branches',18)
    # Upper feature-only temporal branch.
    panel(720,141,355,165,'ALFTemporal',GREEN)
    tensor(746,204,'#b7d6a7');line([(830,228),(852,228)])
    box(852,188,120,76,'Spatial\ncontext',GREEN,21)
    line([(972,228),(994,228)]);vector(995,217,'#b4d3a3',3,18)
    txt(1025,277,'Z',22,italic=True)
    txt(889,326,'History + S + calendar + structural features',17)
    box(1140,188,164,82,'FiLM head\nMLP',YELLOW,23)
    line([(1075,228),(1140,228)])
    # Main graph forecast branch.
    panel(720,365,210,164,'ALFGraph',BLUE)
    box(737,411,176,69,'Routed graph\nencoder',BLUE,22)
    box(983,403,114,79,'FiLM',YELLOW,25)
    line([(930,444),(983,444)]);txt(955,429,'H',23,italic=True)
    line([(1097,444),(1140,444)]);txt(1118,429,'H′',23,italic=True)
    box(1140,395,164,101,'Residual\ndecoder',BLUE,23)
    # A sole arrow from the temporal head supplies both modulation parameters.
    line([(1222,270),(1222,348),(1040,348),(1040,403)])
    txt(1122,338,'1 + Δγ, β',20)
    box(1340,405,104,78,'Restore\ncenter*','#f2f3f4',20)
    line([(1304,444),(1340,444)])
    box(1481,392,150,107,'Graph\nRefinement',PEACH,22)
    line([(1444,444),(1481,444)])
    line([(1631,444),(1660,444)])
    graph(1660,409,.9);txt(1711,515,'Ŷ',26,italic=True)
    txt(1711,542,'12 future steps',17)
    # Exact conditioning routes, kept separate from FiLM.
    line([(385,269),(385,345)],COND,True);txt(411,316,'R, a, M',17,color=COND)
    line([(655,143),(683,143),(683,83),(1556,83),(1556,392)],COND,True)
    txt(1522,118,'R_L',18,color=COND)
    line([(810,83),(810,141)],COND,True);dot(810,83,3,COND)
    line([(683,143),(683,341),(825,341),(825,365)],COND,True)
    txt(766,353,'S_t',18,color=COND)
    line([(1392,83),(1392,365),(1262,365),(1262,395)],COND,True)
    txt(1327,356,'S_L',18,color=COND)
    dot(1392,83,3,COND)
    txt(1040,585,'A supplies reconstruction, graph propagation, structural features and refinement.',19)

def routing():
    panel(20,638,660,409,'(b) Observation-conditioned graph propagation',BLUE)
    txt(69,711,'S',23,italic=True);vector(45,726,'#cbb4e0',3,13)
    box(107,691,94,53,'Linear',PURPLE,21);box(227,691,104,53,'Softmax',PURPLE,21)
    line([(85,716),(107,716)]);line([(201,716),(227,716)])
    line([(331,716),(360,716)]);vector(369,705,['#abd0e8','#b9d6a5','#f8dda3','#f3bf9f','#d6bce2'],5,24)
    txt(514,763,'π₁ … π₅  (all five active)',19)
    txt(60,800,'U',23,italic=True)
    line([(61,809),(61,823),(621,823)],arrow=False)
    labels=['Identity','Mean','Weighted\nmean','Forward','Backward']
    cols=['#d8ebf8','#e0efd6','#fff1cd','#fae4d5','#ece0f4']
    for i in range(5):
        x=94+i*113;mid=x+47
        line([(mid,823),(mid,839)])
        box(x,839,94,60,labels[i],cols[i],18)
        txt(mid,921,f'A{i+1} U Θ{i+1}',17)
        line([(mid,927),(mid,945)])
        op(mid,961,'⊙');txt(mid+25,958,f'π{i+1}',16,anchor='start',color=COND)
        line([(mid,977),(mid,991),(350,991)],arrow=False)
    line([(350,991),(350,1010)]);txt(367,1032,'Σₑ πₑ Aₑ U Θₑ + b',21,anchor='start')
    txt(42,1031,'Dense soft routing',19,anchor='start',bold=True)
    txt(44,780,'A → five fixed supports',17,anchor='start')

def temporal():
    panel(700,638,625,409,'(c) Spatial-context temporal features',GREEN)
    txt(1012,709,'Filled history + S + calendar',22)
    line([(1012,720),(1012,743)])
    box(810,743,405,52,'12-step history embedding',GREEN,22)
    txt(1270,767,['One','temporal','patch'],16)
    line([(1012,795),(1012,820)])
    vector(859,825,['#b6d6a4','#bad8ec','#f2d3a8'],9,22)
    txt(1012,875,'Partition tokens: P partitions × C slots',20)
    line([(1012,881),(1012,900)])
    box(797,900,430,49,'Cross-partition interaction',GREEN,22)
    line([(1012,949),(1012,968)])
    box(797,968,430,49,'Within-partition interaction',GREEN,22)
    line([(1227,992),(1260,992)]);txt(1290,999,'Z',26,italic=True)
    txt(1012,1036,'Structural features: shared mapping, then concatenation.',16)

def film():
    panel(1345,638,435,409,'(d) Temporal–FiLM',YELLOW)
    txt(1390,723,'Z',24,italic=True);vector(1368,738,'#b5d7a3',3,18)
    line([(1434,748),(1460,748)])
    box(1460,711,87,72,'MLP',YELLOW,22)
    line([(1547,748),(1575,748),(1575,712),(1603,712)])
    line([(1575,748),(1575,805),(1689,805)])
    dot(1575,748)
    txt(1640,690,'1 + Δγ',21);vector(1604,702,'#a9d0e9',3,20)
    txt(1719,785,'β',23,italic=True);vector(1689,795,'#cbb3df',3,20)
    line([(1637,722),(1637,907)])
    line([(1721,815),(1721,907)])
    txt(1400,903,'H',24,italic=True);vector(1366,919,'#a9d0e9',4,20)
    line([(1454,929),(1621,929)]);op(1637,929,'⊙')
    line([(1653,929),(1705,929)]);op(1721,929,'+')
    line([(1737,929),(1762,929)]);txt(1750,973,'H′',24,italic=True)
    txt(1562,1025,'H′ = (1 + Δγ) ⊙ H + β',24)

start(1800,1160,'ALFNet — observation-support-conditioned forecasting')
overall();routing();temporal();film()
rect(20,1067,660,65,PURPLE)
txt(350,1093,'Training only:  0.9 Lmean + 0.1 Lrole + Lroute',22)
txt(350,1118,'Masked teacher forcing during training; no reconstruction loss.',17)
line([(713,1088),(773,1088)]);txt(783,1095,'Forward',19,anchor='start')
line([(899,1088),(961,1088)],COND,True);txt(973,1095,'Conditioning',19,anchor='start')
txt(713,1125,'* Dataset-specific interpolation, residual scale d and centering c.',19,anchor='start')
txt(1770,1125,'History / horizon: 12 / 12',19,anchor='end')
finish('alfnet_architecture_editable.svg')

# A separate palette of small primitives for copy/paste into PowerPoint.
start(1400,680,'ALFNet editable module library')
txt(35,47,'ALFNet — editable module library',29,anchor='start',bold=True)
panel(25,75,310,245,'History / graph snapshots','white');snapshots(110,150)
panel(355,75,310,245,'Mask and feature tokens','white');grid(395,135,7,4,18);vector(395,250,['#aad1e8','#b9d6a5','#f4caa6','#cbb4e0'],8,23)
panel(685,75,310,245,'Feature stacks','white');tensor(735,150,'#afd0e4',5);tensor(735,245,'#b8d6a6',5)
panel(1015,75,355,245,'Operations / connectors','white');op(1080,152,'+');op(1160,152,'⊙');line([(1060,220),(1180,220)]);line([(1060,270),(1180,270)],COND,True)
for i,(label,col) in enumerate([('ALFGraph',BLUE),('ALFTemporal',GREEN),('Lifecycle context',PURPLE),('Shared GRIN',PEACH),('FiLM',YELLOW)]):
    box(25+i*275,355,255,86,label,col,25)
box(25,485,255,86,'Residual decoder',BLUE,24);box(300,485,255,86,'Graph Refinement',PEACH,23)
box(575,485,255,86,'Window handling*','#f2f3f4',23);box(850,485,255,86,'Restore center*','#f2f3f4',23)
txt(35,633,'Each element is native SVG text, line, rectangle, polygon or circle; no raster content.',22,anchor='start')
finish('alfnet_module_library.svg')
print('Wrote architecture and module-library SVGs.')
