"""Structure-locked skeleton for the ALFNet v4 figure.

Draws every box, port, arrow and mechanism glyph at its verified position so that the
image model only restyles it. Output: _source/skeleton.svg (2400 x 1350 design units).
Only basic primitives (rect, circle, polyline, polygon, path, text) are used.
"""
import math
from pathlib import Path

W, H = 2400, 1350
C = dict(
    ink='#25313C', muted='#6B7680', white='#FFFFFF',
    blue='#EAF3FB', blueE='#7FA7CF', green='#EEF6EA', greenE='#8DB07E',
    red='#FCEDEC', redE='#D79A95', purple='#F3ECFA', cond='#8A62B0',
    gold='#FFF2CC', goldE='#CFA845', orange='#FFF0E0', orangeE='#E0A86E',
    gray='#8A949C', teal='#3E8E92',
    vs='#C9BEE0', aas='#9CC7D1', nas='#BBD89C', fs='#EDB3A6',
    seg1='#BFDDB2', seg2='#BCD7EE', seg3='#D6C7EC', seg4='#F6CDB4',
)
P = []


def esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def rect(x, y, w, h, fill=C['white'], stroke=C['ink'], r=8, sw=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    P.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" '
             f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=26, anchor='middle', bold=False, color=None, italic=False, math_font=False):
    size = max(size, 19)
    fam = "Cambria Math, Cambria, 'Times New Roman', serif" if math_font else 'Arial, Helvetica, sans-serif'
    P.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{fam}" font-size="{size}" '
             f'fill="{color or C["ink"]}"{" font-weight=\"700\"" if bold else ""}'
             f'{" font-style=\"italic\"" if italic else ""}>{esc(s)}</text>')


def circle(x, y, r, fill=C['white'], stroke=C['ink'], sw=1.6):
    P.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def head(x, y, a, b, color, size=11):
    t = math.atan2(y - b, x - a)
    p1 = (x - size * math.cos(t) + 4.8 * math.sin(t), y - size * math.sin(t) - 4.8 * math.cos(t))
    p2 = (x - size * math.cos(t) - 4.8 * math.sin(t), y - size * math.sin(t) + 4.8 * math.cos(t))
    P.append(f'<polygon points="{x},{y} {p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" fill="{color}"/>')


def line(pts, color=None, width=2.2, dash=False, arrow=True):
    color = color or C['ink']
    d = ' stroke-dasharray="8 6"' if dash else ''
    P.append(f'<polyline points="{" ".join(f"{a},{b}" for a, b in pts)}" fill="none" stroke="{color}" '
             f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{d}/>')
    if arrow:
        (x, y), (a, b) = pts[-1], pts[-2]
        head(x, y, a, b, color)


def path(d, color=None, width=2.0, dash=False, fill='none'):
    dd = ' stroke-dasharray="8 6"' if dash else ''
    P.append(f'<path d="{d}" fill="{fill}" stroke="{color or C["ink"]}" stroke-width="{width}"{dd}/>')


def op(x, y, sym, r=17):
    circle(x, y, r)
    text(x, y + 9, sym, 28, bold=True)


def tag(x, y, s, kind='cond', w=None):
    color = C['cond'] if kind == 'cond' else (C['gray'] if kind == 'graph' else C['ink'])
    w = w or (22 + 15 * len(s))
    rect(x, y, w, 36, C['white'], color, 6, 1.6)
    text(x + w / 2, y + 27, s, 24, color=color, bold=True)
    return w


def box(x, y, w, h, label, fill=C['white'], size=24, stroke=None, bold=False):
    rect(x, y, w, h, fill, stroke or C['ink'], 7, 1.4)
    lines = label.split('\n')
    for i, ln in enumerate(lines):
        text(x + w / 2, y + h / 2 + size * 0.36 + (i - (len(lines) - 1) / 2) * size * 1.15, ln, size, bold=bold)


def slab(x, y, w, h, fill, depth=10, stroke=None):
    """Thin extruded tensor: front face plus top and side faces."""
    s = stroke or C['ink']
    P.append(f'<polygon points="{x},{y} {x+depth},{y-depth} {x+w+depth},{y-depth} {x+w},{y}" fill="{fill}" '
             f'stroke="{s}" stroke-width="1.1" opacity="0.85"/>')
    P.append(f'<polygon points="{x+w},{y} {x+w+depth},{y-depth} {x+w+depth},{y+h-depth} {x+w},{y+h}" '
             f'fill="{fill}" stroke="{s}" stroke-width="1.1" opacity="0.7"/>')
    rect(x, y, w, h, fill, s, 0, 1.2)


def cells_h(x, y, n, cw, ch, fills, stroke=None):
    for i in range(n):
        f = fills[i % len(fills)] if isinstance(fills, list) else fills
        rect(x + i * cw, y, cw, ch, f, stroke or C['ink'], 0, 0.9)


def cells_v(x, y, n, cw, ch, fills, stroke=None):
    for i in range(n):
        f = fills[i % len(fills)] if isinstance(fills, list) else fills
        rect(x, y + i * ch, cw, ch, f, stroke or C['ink'], 0, 0.9)


def panel(x, y, w, h, fill, edge, title=None, size=30, anchor='middle'):
    rect(x, y, w, h, fill, edge, 14, 2.0)
    if title:
        tx = x + w / 2 if anchor == 'middle' else x + 22
        text(tx, y + 42, title, size, anchor=anchor, bold=True)


def sub(x, y, w, h, title, edge, size=26):
    rect(x, y, w, h, C['white'], edge, 10, 1.4)
    text(x + w / 2, y + 34, title, size, bold=True)


VALS = [.35, .5, .42, .62, .78, .58, .5, .68, .83, .64, .55, .72]


def curve(x, y, w, h, missing=(), color=None, dashed_bridge=None):
    color = color or C['teal']
    pts = [(x + i * w / 11, y + h * (1 - v)) for i, v in enumerate(VALS)]
    line([(x - 4, y + h + 10), (x + w + 6, y + h + 10)], color='#A9B3BB', width=1.2, arrow=False)
    for i in range(11):
        if i not in missing and i + 1 not in missing:
            line([pts[i], pts[i + 1]], color=color, width=2.6, arrow=False)
    for i, (a, b) in enumerate(pts):
        if i in missing:
            circle(a, y + h + 10, 4, C['white'], '#A9B3BB', 1.1)
        else:
            circle(a, b, 4.2, color, color, 0.6)
    if dashed_bridge:
        i, j = dashed_bridge
        line([pts[i], pts[j]], color='#D0605A', width=2.2, dash=True, arrow=False)
        for k in range(i + 1, j):
            fx = pts[i][0] + (pts[j][0] - pts[i][0]) * (k - i) / (j - i)
            fy = pts[i][1] + (pts[j][1] - pts[i][1]) * (k - i) / (j - i)
            circle(fx, fy, 4.2, '#F2B2AC', '#D0605A', 1.2)


# ---------------------------------------------------------------- inputs (x 20-300)
def inputs():
    panel(20, 20, 280, 300, C['white'], C['blueE'], 'Visible Traffic History', 26)
    curve(52, 90, 216, 60, missing=(4, 5))
    curve(52, 175, 216, 60, missing=(0, 1, 2, 3, 4, 5, 6))
    text(52, 268, 't−11', 22, anchor='start', math_font=True)
    text(268, 268, 't', 22, anchor='end', math_font=True)
    text(160, 302, 'X : 12 × N × 1', 26, math_font=True)

    panel(20, 340, 280, 520, C['purple'], C['cond'], 'Lifecycle Context', 26)
    names = ['Unseen', 'Steady', 'Join', 'Fault']
    for r, n in enumerate(names):
        y = 400 + r * 46
        text(36, y + 20, n, 23, anchor='start')
        for t in range(12):
            role = ['vs', 'aas', 'vs' if t < 6 else 'nas', 'aas' if t < 6 else 'fs'][r]
            rect(126 + t * 13, y, 13, 24, C[role], C['white'], 0, 0.8)
    for i, (k, lab) in enumerate([('aas', 'AAS'), ('nas', 'NAS'), ('vs', 'VS'), ('fs', 'FS')]):
        rect(32 + i * 67, 594, 16, 16, C[k], C['ink'], 2, 0.8)
        text(51 + i * 67, 610, lab, 17, anchor='start')
    text(160, 646, 'Rₜ : role trajectory', 22, math_font=True)
    text(48, 700, 'aₜ', 26, anchor='start', math_font=True)
    for t in range(12):
        hh = 3 if t < 6 else (t - 5) * 7
        rect(98 + t * 14, 704 - hh, 10, hh, C['cond'], C['cond'], 0, 0.3)
    text(48, 752, 'Mₜ', 26, anchor='start', math_font=True)
    for r in range(2):
        for c in range(12):
            on = not ((r == 0 and c in (4, 5)) or (r == 1 and c < 7))
            rect(98 + c * 14, 722 + r * 16, 14, 16, C['cond'] if on else C['white'], '#CBB8E0', 0, 0.8)
    text(160, 800, 'Sₜ = [Rₜ, aₜ, Mₜ]', 26, math_font=True)
    text(160, 836, '4 + 1 + 1 channels', 22)

    panel(20, 880, 280, 350, C['white'], C['blueE'], 'Static Graph', 26)
    pts = [(62, 990), (118, 952), (176, 994), (238, 960), (92, 1056), (206, 1060)]
    edges = [(0, 1), (0, 4), (1, 2), (2, 3), (2, 4), (2, 5), (4, 5)]
    for a, b in edges:
        line([pts[a], pts[b]], color=C['blueE'], width=2.2, arrow=False)
    for x, y in pts:
        circle(x, y, 11, C['blue'], C['blueE'], 2)
    for r in range(6):
        for c in range(6):
            on = any((a == r and b == c) or (a == c and b == r) for a, b in edges)
            rect(52 + c * 14, 1100 + r * 14, 14, 14, C['blueE'] if on else C['white'], C['blueE'], 0, 0.7)
    text(160, 1150, 'A', 30, math_font=True, italic=True)
    text(160, 1208, 'Graph attributes G', 22)


# ---------------------------------------------------------- completion (x 320-640)
def completion():
    panel(320, 20, 320, 1210, C['red'], C['redE'])
    text(480, 58, 'Support-Aware', 28, bold=True)
    text(480, 92, 'History Completion', 28, bold=True)

    # (a) shared bidirectional reconstruction
    sub(338, 110, 284, 330, 'GRIN / BiGRIL', C['redE'])
    tag(346, 124, 'S')
    tag(578, 124, 'A', 'graph')
    for i in range(3):
        box(360 + i * 86, 172, 62, 40, 'GR', '#F8D3CF', 22)
        box(360 + i * 86, 262, 62, 40, 'GR', '#D5E6F5', 22)
        if i < 2:
            line([(422 + i * 86, 192), (446 + i * 86, 192)], width=1.8)
            line([(446 + i * 86, 282), (422 + i * 86, 282)], width=1.8)
    text(480, 240, 'Forward  t−11 → t', 19)
    text(480, 326, 'Backward  t → t−11', 19)
    line([(594, 192), (608, 192), (608, 360), (540, 360), (540, 372)], width=1.8)
    line([(360, 282), (350, 282), (350, 360), (420, 360), (420, 372)], width=1.8)
    box(366, 372, 228, 50, 'Align + merge · 12 steps', '#F8D3CF', 18)

    line([(480, 440), (480, 466)])
    # (b) keep observations
    sub(338, 466, 284, 156, 'Keep observations', C['redE'], 22)
    tag(346, 478, 'M')
    for i in range(8):
        obs = i not in (2, 3, 6)
        circle(372 + i * 31, 536, 11, '#4F86C6' if obs else '#E8807A', '#2F5E93' if obs else '#B9504A', 1.4)
    circle(360, 580, 7, '#4F86C6', '#2F5E93', 1)
    text(372, 587, 'observed', 18, anchor='start')
    circle(482, 580, 7, '#E8807A', '#B9504A', 1)
    text(494, 587, 'GRIN-filled', 18, anchor='start')

    line([(480, 622), (480, 648)])
    # (c) optional interpolation overwrite
    sub(338, 648, 284, 196, 'Interpolation*', C['redE'], 24)
    curve(362, 690, 236, 54, missing=(), dashed_bridge=(3, 7))
    text(480, 800, 'q > 0: overwrite missing', 20)
    text(480, 828, 'q = 0: keep GRIN fill', 20, color=C['muted'])

    line([(480, 844), (480, 870)])
    # (d) centering
    sub(338, 870, 284, 120, 'Reference centering*', C['redE'], 24)
    text(480, 946, 'X̄ = X_filled − αv', 26, math_font=True)

    line([(480, 990), (480, 1018)])
    # (e) shared processed history
    slab(370, 1030, 216, 40, '#CFE1F2', 12)
    for i in range(1, 12):
        line([(370 + i * 18, 1030), (370 + i * 18, 1070)], color=C['ink'], width=0.8, arrow=False)
    text(480, 1110, 'X̄ : 12 × N × 1', 26, math_font=True)
    text(480, 1146, 'shared by both', 21, color=C['muted'])
    text(480, 1174, 'branches', 21, color=C['muted'])


# ------------------------------------------------------------ temporal (660-2380)
def temporal():
    panel(660, 20, 1720, 548, C['green'], C['greenE'],
          'ALFTemporal · Spatial-Context Temporal Conditioning', 30, anchor='start')

    # T1 temporal patch embedding (concat order verified: patch | ToD | DoW | Struct)
    sub(680, 74, 500, 474, 'Temporal patch embedding', C['greenE'], 25)
    text(700, 140, '[X̄, S, ToD/288, DoW/7] · 9 ch', 21, anchor='start', math_font=True)
    tag(1000, 116, 'S')
    colors = ['#BFDDB2', '#CFE1F2', '#D6C7EC', '#BFDDB2', '#CFE1F2', '#D6C7EC']
    for i in range(12):
        slab(704 + i * 21, 204, 15, 96, colors[i % 6], 7)
    path('M 704 196 Q 704 184 716 184 L 818 184 Q 828 184 832 172 Q 836 184 846 184 L 950 184 Q 962 184 962 196',
         width=1.8)
    text(832, 166, 'Conv  k = s = 12', 20, math_font=True)
    text(704, 324, 't−11', 19, anchor='start', math_font=True)
    text(956, 322, 't', 19, anchor='end', math_font=True)
    line([(972, 252), (1004, 252)])
    cells_v(1008, 196, 8, 22, 14, ['#9FCB8C', '#BFDDB2'])
    text(1019, 184, '1 patch · 64', 17, bold=True)
    text(1034, 336, 'per node', 17)
    bx = 1110
    segs = [(196, 128, C['seg1'], '64'), (324, 64, C['seg2'], '32'), (388, 64, C['seg3'], '32'),
            (452, 64, C['seg4'], '32')]
    for y, h, f, lab in segs:
        rect(bx, y, 30, h, f, C['ink'], 0, 1.2)
        text(bx + 36, y + h / 2 + 7, lab, 17, anchor='start')
    text(bx + 15, 184, '160', 19, bold=True)
    text(bx + 15, 538, 'concat', 17, color=C['muted'])
    line([(1030, 252), (bx - 2, 252)], width=1.8)
    cx, cy = 722, 370
    circle(cx, cy, 15, C['white'], C['ink'], 1.8)
    line([(cx, cy), (cx, cy - 9)], width=1.8, arrow=False)
    line([(cx, cy), (cx + 7, cy + 4)], width=1.8, arrow=False)
    text(748, 377, 'ToD embedding', 19, anchor='start')
    line([(900, 356), (bx - 2, 356)], width=1.8)
    rect(708, 404, 30, 30, C['white'], C['ink'], 4, 1.6)
    rect(708, 404, 30, 8, '#E07A5F', C['ink'], 2, 1.2)
    for r in range(2):
        for c in range(3):
            rect(713 + c * 8, 417 + r * 7, 6, 5, '#D0D6DB', '#D0D6DB', 0, 0.5)
    text(748, 427, 'DoW embedding', 19, anchor='start')
    line([(900, 420), (bx - 2, 420)], width=1.8)
    tag(700, 466, 'G', 'graph')
    box(748, 462, 196, 44, 'Struct MLP 7→64→32', C['white'], 17)
    line([(944, 484), (bx - 2, 484)], width=1.8)

    line([(1180, 300), (1200, 300)])

    # T2 spatial partitions
    sub(1200, 74, 270, 474, 'Spatial partitioning', C['greenE'], 24)
    node_cols = ['#7FB069', '#5B8DBE', '#C97C5D', '#9B7DC1', '#E2B44C', '#5FA8A0']
    for i in range(6):
        rect(1214, 136 + i * 20, 30, 14, node_cols[i], C['ink'], 2, 0.8)
    text(1229, 276, 'N', 18, italic=True)
    line([(1250, 190), (1278, 190)])
    gx, gy, cs = 1300, 150, 30
    for r in range(4):
        for c in range(5):
            if c == 3:
                continue
            f = node_cols[(r * 2 + c) % 6] if (r + c) % 2 == 0 else '#E5F0E0'
            rect(gx + c * cs, gy + r * cs, cs, cs, f, C['greenE'], 0, 1)
    for r in range(4):
        text(gx + 3 * cs + 15, gy + r * cs + 21, '…', 18)
    for c, lab in enumerate(['1', '2', '3', '', 'P']):
        text(gx + c * cs + 15, gy - 10, lab, 17, italic=(lab == 'P'))
    for r, lab in enumerate(['1', '2', '…', 'C']):
        text(gx - 12, gy + r * cs + 21, lab, 17, italic=(lab == 'C'))
    text(1335, 312, 'P = 32 partitions', 19)
    text(1335, 338, 'C slots each', 19)
    text(1335, 400, 'reorder / pad', 18, color=C['muted'])
    text(1335, 422, 'by spatial layout', 18, color=C['muted'])

    line([(1470, 300), (1490, 300)])

    # T3 + T4 within one two-stage attention block
    rect(1490, 74, 600, 474, C['white'], C['greenE'], 10, 1.4)
    text(1790, 108, 'One two-stage attention block (1 head)', 24, bold=True)
    # T3 cross-partition attention
    text(1620, 146, 'Cross-partition', 22, bold=True)
    gx, gy, cs = 1528, 196, 34
    for r in range(4):
        for c in range(5):
            if c == 3:
                continue
            f = '#9FCB8C' if r == 1 else '#F2F7EF'
            rect(gx + c * cs, gy + r * cs, cs, cs, f, C['greenE'], 0, 1)
    for r in range(4):
        text(gx + 3 * cs + 17, gy + r * cs + 24, '…', 20)
    rect(gx - 3, gy + cs - 3, 5 * cs + 6, cs + 6, 'none', C['ink'], 3, 3)
    for c1, c2 in [(0, 1), (1, 2), (0, 2), (2, 4)]:
        xa, xb = gx + c1 * cs + 17, gx + c2 * cs + 17
        path(f'M {xa} {gy + cs - 4} Q {(xa + xb) / 2} {gy + cs - 26 - 6 * (c2 - c1)} {xb} {gy + cs - 4}',
             C['ink'], 1.6)
    text(1605, gy + 4 * cs + 34, 'same slot across P', 19)
    text(1605, gy + 4 * cs + 60, 'partitions', 19)
    line([(1700, 300), (1736, 300)])

    # T4 within-partition interaction
    text(1910, 146, 'Within-partition', 22, bold=True)
    for i in range(5):
        rect(1752, 196 + i * 30, 40, 30, '#9FCB8C' if i % 2 == 0 else '#BFDDB2', C['ink'], 0, 1)
    text(1772, 362, 'C slots', 18)
    for i in range(4):
        rect(1880, 214 + i * 30, 36, 24, '#4E8F5B', C['ink'], 3, 1)
    text(1898, 362, '4 latent', 18)
    text(1898, 384, 'tokens', 18)
    for i in range(5):
        rect(2026, 196 + i * 30, 40, 30, '#9FCB8C' if i % 2 == 0 else '#BFDDB2', C['ink'], 0, 1)
    text(2046, 362, 'C slots', 18)
    for i in range(5):
        line([(1792, 211 + i * 30), (1878, 226 + min(i, 3) * 30)], color=C['muted'], width=1.0, arrow=False)
        line([(1916, 226 + min(i, 3) * 30), (2024, 211 + i * 30)], color=C['muted'], width=1.0, arrow=False)
    line([(1800, 420), (1868, 420)], width=2.2)
    text(1834, 446, 'Gather', 19)
    line([(1930, 420), (2018, 420)], width=2.2)
    text(1974, 446, 'Broadcast', 19)

    line([(2090, 300), (2110, 300)])

    # T5 restore node order + FiLM head
    sub(2110, 74, 256, 474, 'Restore node order', C['greenE'], 23)
    order_a = [2, 0, 4, 1, 5, 3]
    for i, k in enumerate(order_a):
        rect(2130, 120 + i * 20, 34, 14, node_cols[k], C['ink'], 2, 0.8)
        j = k
        line([(2166, 127 + i * 20), (2236, 127 + j * 20)], color=C['muted'], width=1.0, arrow=False)
    for i in range(6):
        rect(2238, 120 + i * 20, 34, 14, node_cols[i], C['ink'], 2, 0.8)
    text(2300, 160, 'Z', 26, math_font=True, italic=True)
    text(2316, 190, 'N × 160', 17, math_font=True)
    line([(2238, 246), (2238, 272)])
    box(2128, 274, 220, 70, 'FiLM head\n160 → 32 → 128', C['gold'], 20, stroke=C['goldE'], bold=False)
    line([(2238, 344), (2238, 370)], arrow=True)
    cells_h(2146, 380, 6, 22, 20, '#BFDDB2')
    text(2290, 397, 'Δγ', 22, anchor='start', math_font=True)
    cells_h(2146, 414, 6, 22, 20, '#F6CDB4')
    text(2290, 431, 'β', 22, anchor='start', math_font=True)
    text(2238, 470, 'each N × 64', 18)
    # parameter line to the FiLM block in the graph region
    line([(2238, 482), (2238, 584), (1110, 584), (1110, 668)], width=2.6)
    rect(1580, 566, 120, 34, C['white'], C['white'], 4, 0)
    text(1640, 592, 'Δγ, β', 24, math_font=True, bold=True)


# --------------------------------------------------------------- graph (660-2380)
def graph():
    panel(660, 604, 1720, 626, C['blue'], C['blueE'])
    text(1150, 644, 'ALFGraph · Observation-Conditioned Graph Forecasting', 28, anchor='start', bold=True)
    y0 = 760  # main data line

    # encoder
    sub(680, 668, 250, 236, 'Graph Encoder', C['blueE'], 24)
    text(805, 726, '2 layers · hidden 64', 18, color=C['muted'])
    slab(744, 742, 46, 120, '#CFE1F2', 12)
    slab(820, 742, 46, 120, '#B5D1EC', 12)
    text(767, 808, 'L1', 22, bold=True)
    text(843, 808, 'L2', 22, bold=True)
    line([(790, 802), (818, 802)], width=1.6)
    tag(688, 860, 'Sₜ')
    tag(880, 860, 'A', 'graph')
    text(805, 896, 'H : 2 × N × 64', 19, math_font=True)

    # FiLM
    rect(960, 668, 300, 236, C['gold'], C['goldE'], 12, 2)
    text(978, 702, 'FiLM', 26, anchor='start', bold=True)
    line([(1110, 668), (1110, 720)], arrow=False, width=2.6)
    line([(1110, 720), (1060, 720), (1060, y0 - 19)], width=2.2)
    line([(1110, 720), (1170, 720), (1170, y0 - 19)], width=2.2)
    text(1018, 744, '1 + Δγ', 19, math_font=True)
    text(1206, 744, 'β', 20, math_font=True)
    op(1060, y0, '⊙', 18)
    op(1170, y0, '+', 18)
    text(1110, 836, 'H_mod = (1 + Δγ) ⊙ H + β', 20, math_font=True)
    text(1110, 872, 'once · both layers', 19, color=C['muted'])

    # encoder -> FiLM -> decoder
    line([(878, y0), (1042, y0)])
    line([(1078, y0), (1152, y0)])
    line([(1188, y0), (1388, y0)])

    # decoder
    sub(1290, 668, 360, 236, 'Residual AR Decoder', C['blueE'], 24)
    text(1470, 726, '2 layers × 12 steps', 18, color=C['muted'])
    text(1296, y0 + 32, 'h₀=H_mod', 19, anchor='start', math_font=True)
    slab(1390, 742, 46, 104, '#CFE1F2', 12)
    slab(1466, 742, 46, 104, '#B5D1EC', 12)
    text(1413, 800, 'L1', 22, bold=True)
    text(1489, 800, 'L2', 22, bold=True)
    line([(1436, 794), (1464, 794)], width=1.6)
    line([(1524, 794), (1552, 794)], width=1.6)
    box(1554, 776, 60, 36, 'yₖ', C['white'], 20)
    # feedback y_k -> next step input (the unrefined step prediction)
    line([(1584, 812), (1584, 872), (1413, 872), (1413, 850)], width=1.8)
    text(1500, 898, 'yₖ fed back × 12', 19, color=C['muted'])
    tag(1592, 700, 'Sᴸ')
    tag(1296, 856, 'y₀ = X̄ᴸ', 'data', w=96)
    line([(1392, 874), (1413, 874)], width=1.8, arrow=False)
    # decoder -> restore
    line([(1614, 794), (1662, 794), (1662, y0), (1700, y0)])
    text(1678, y0 - 12, 'y₁:₁₂', 17, math_font=True)
    sub(1700, 668, 130, 236, 'Restore', C['blueE'], 22)
    text(1765, 728, 'reference*', 18)
    op(1765, y0 + 20, '+', 18)
    line([(1700, y0), (1720, y0), (1720, y0 + 20), (1747, y0 + 20)])
    line([(1765, 852), (1765, y0 + 38)])
    box(1735, 852, 60, 34, 'αv', C['white'], 20)
    line([(1783, y0 + 20), (1850, y0 + 20)])
    text(1814, y0 + 6, 'Y₀', 22, math_font=True, bold=True)

    # refinement
    rect(1850, 668, 290, 400, C['orange'], C['orangeE'], 12, 2)
    text(1995, 700, 'Role-Scaled', 22, bold=True)
    text(1995, 726, 'Graph Refinement', 22, bold=True)
    line([(1850, y0 + 20), (1880, y0 + 20)])
    slab(1884, 760, 46, 40, '#CFE1F2', 8)
    slab(1940, 760, 46, 40, '#D6C7EC', 8)
    slab(1996, 760, 46, 40, '#F6CDB4', 8)
    text(1966, 826, '[Y₀, P⁺Y₀, P⁻Y₀]', 19, math_font=True)
    tag(2066, 752, 'A', 'graph')
    line([(1966, 834), (1966, 852)])
    box(1860, 850, 212, 42, 'MLP 3→32→1, GELU', C['white'], 16)
    line([(1966, 890), (1966, 906)])
    box(1912, 906, 108, 34, 'tanh', C['white'], 19)
    line([(1966, 940), (1966, 956)])
    box(1882, 956, 168, 36, '× 0.05 · s(Rᴸ)', C['white'], 19)
    tag(1856, 1002, 'Rᴸ')
    line([(1906, 1004), (1912, 992)], color=C['cond'], dash=True, width=1.6)
    line([(1966, 992), (1966, 1018)])
    op(1966, 1036, '+', 18)
    # Y0 residual bypass
    line([(1866, y0 + 20), (1866, 742), (2116, 742), (2116, 1036), (1984, 1036)], width=2.0)
    text(2108, 900, 'Y₀', 19, anchor='end', math_font=True)
    # output
    line([(1966, 1054), (1966, 1086), (2160, 1086), (2160, 900), (2176, 900)])
    text(2060, 1112, 'Ŷ', 24, math_font=True, bold=True)

    # forecast (single output)
    sub(2176, 668, 190, 400, 'Forecast', C['blueE'], 26)
    for k, (col, off) in enumerate([('#3E7CB1', 0), ('#8A62B0', 34), ('#5FA35C', 68)]):
        pts = [(2196 + i * 14, 730 + off + 46 * (1 - VALS[(i + 3 * k) % 12])) for i in range(12)]
        line(pts, color=col, width=2.2, arrow=False)
        for a, b in pts:
            circle(a, b, 3, col, col, 0.5)
    line([(2192, 880), (2352, 880)], color='#A9B3BB', width=1.2, arrow=False)
    text(2196, 908, 't+1', 17, anchor='start', math_font=True)
    text(2350, 908, 't+12', 17, anchor='end', math_font=True)
    text(2271, 960, 'Ŷ : 12 × N × 1', 21, math_font=True)

    # cell definition inset
    rect(680, 930, 1150, 286, C['white'], C['blueE'], 10, 1.6, dash='10 6')
    line([(805, 904), (805, 930)], color=C['blueE'], dash=True, arrow=False, width=1.6)
    line([(1470, 904), (1470, 930)], color=C['blueE'], dash=True, arrow=False, width=1.6)
    text(700, 964, 'DCRNN cell · lifecycle-conditioned propagation', 22, anchor='start', bold=True)
    text(1250, 964, 'separate router per layer and gate', 19, anchor='start', color=C['muted'])
    tag(700, 1010, 'S')
    line([(738, 1028), (760, 1028)], color=C['cond'], dash=True, width=1.6)
    box(762, 1006, 176, 44, 'Linear → Softmax', '#F3ECFA', 17)
    hs = [26, 18, 30, 12, 22]
    for i, hh in enumerate(hs):
        rect(956 + i * 16, 1046 - hh, 12, hh, C['cond'], C['cond'], 0, 0.5)
    text(1046, 1036, 'dense w₁…w₅', 19, anchor='start')
    line([(938, 1028), (952, 1028)], width=1.4)
    names = ['I', 'Mean', 'WM', 'Fwd', 'Bwd']
    pats = [[(i, i) for i in range(4)], [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)],
            [(0, 1), (1, 2), (2, 3), (3, 0)], [(0, 1), (1, 2), (2, 3), (3, 0)],
            [(1, 0), (2, 1), (3, 2), (0, 3)]]
    for k, (n, pat) in enumerate(zip(names, pats)):
        x = 700 + k * 92
        text(x + 28, 1100, n, 17, bold=True)
        for r in range(4):
            for c in range(4):
                rect(x + c * 14, 1110 + r * 14, 14, 14, C['blueE'] if (r, c) in pat else C['white'],
                     C['blueE'], 0, 0.6)
        text(x + 28, 1188, f'w{"₁₂₃₄₅"[k]}', 15, math_font=True)
        line([(x + 56, 1138), (x + 70, 1138)], width=1.2, arrow=False)
    op(1180, 1138, 'Σ', 18)
    line([(1150, 1138), (1162, 1138)], width=1.4)
    text(1006, 1212, 'G(U, S) = Σⱼ wⱼ ⊙ (Pⱼ U Θⱼ) + b', 19, math_font=True)
    line([(1290, 1000), (1290, 1206)], color='#C7D3DE', width=1.4, arrow=False)
    text(1560, 1008, 'Graph-GRU gates', 22, bold=True)
    for i, eq in enumerate(['r = σ(G_r([x, h], S))', 'u = σ(G_u([x, h], S))',
                            'c = tanh(G_c([x, r ⊙ h], S))', 'h′ = u ⊙ h + (1 − u) ⊙ c']):
        text(1560, 1048 + i * 38, eq, 21, math_font=True)


def wiring():
    # input ports and main data lines
    tag(304, 152, 'X', 'data')
    line([(300, 170), (304, 170)], arrow=False)
    tag(304, 580, 'S')
    tag(304, 1000, 'A', 'graph')
    tag(304, 1100, 'G', 'graph')
    # X into GRIN (enters the reconstruction box from the left)
    line([(341, 170), (346, 170), (346, 192), (358, 192)], width=2.0)
    # shared X-bar distribution bus
    line([(598, 1050), (650, 1050), (650, 300), (680, 300)], width=2.4)
    line([(650, y_main()), (744, y_main())], width=2.4)
    circle(650, y_main(), 4.5, C['ink'], C['ink'], 1)
    text(694, y_main() - 10, 'X̄', 20, math_font=True)


def y_main():
    return 802


def footer():
    rect(20, 1250, 2360, 82, C['white'], '#C4CCD3', 10, 1.4)
    text(44, 1300, 'Training only:  L = 0.9 L_mean + 0.1 L_role + L_route  ·  masked teacher forcing', 22,
         anchor='start', math_font=True)
    line([(1150, 1292), (1210, 1292)], width=2.2)
    text(1220, 1300, 'data / parameters', 20, anchor='start')
    line([(1410, 1292), (1470, 1292)], color=C['cond'], dash=True, width=2.2)
    text(1480, 1300, 'lifecycle condition', 20, anchor='start')
    tag(1680, 1274, 'A', 'graph')
    text(1720, 1300, 'static-graph port', 20, anchor='start')
    line([(1910, 1292), (1960, 1292)], color=C['blueE'], dash=True, width=1.8, arrow=False)
    text(1970, 1300, 'cell definition', 20, anchor='start')
    text(2360, 1300, '* dataset-specific', 20, anchor='end')


def build():
    inputs()
    completion()
    temporal()
    graph()
    wiring()
    footer()
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
           f'<title>ALFNet structure skeleton v4</title><rect width="100%" height="100%" fill="white"/>'
           + '\n'.join(P) + '</svg>')
    out = Path(__file__).with_name('skeleton.svg')
    out.write_text(svg, encoding='utf-8')
    print(out, len(P))


if __name__ == '__main__':
    build()
