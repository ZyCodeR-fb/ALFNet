"""Deterministic text patch for the ChatGPT P1 raster.

P1 had exactly three text errors (refinement feature order; u-gate subscript; capital X
in the gate inputs). This script covers only those three formula lines and the one
refinement label with the sampled local background and re-typesets the correct text in
Cambria. Everything else is the unmodified model output. Input raw/P1_gpt_raw.png is
not changed; output is alfnet_architecture_v4_gpt_P1_textfix.png.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'raw' / 'P1_gpt_raw.png'
OUT = ROOT / 'alfnet_architecture_v4_gpt_P1_textfix.png'
FONT_R = 'C:/Windows/Fonts/cambria.ttc'
FONT_I = 'C:/Windows/Fonts/cambriai.ttf'
SS = 4  # supersampling factor
INK = (38, 46, 56)

im = Image.open(SRC).convert('RGB')
arr = np.asarray(im)


def bg(box):
    x0, y0, x1, y1 = box
    ring = np.concatenate([arr[y0 - 3:y0 - 1, x0:x1].reshape(-1, 3), arr[y1 + 1:y1 + 3, x0:x1].reshape(-1, 3)])
    return tuple(int(v) for v in np.median(ring, axis=0))


def runs_width(runs, size):
    w = 0
    for s, style in runs:
        f = ImageFont.truetype(FONT_I if style in ('i', 'si') else FONT_R, int(size * (0.68 if style.startswith('s') else 1)))
        w += f.getlength(s)
    return w


def patch(box, runs, size, center_x=None):
    """Cover box with local background, then typeset runs centred in it.

    runs: list of (text, style) with style in {'r','i','sr','si','sup'}; 's*' are
    subscripts, 'sup' superscript (upright)."""
    x0, y0, x1, y1 = box
    color = bg(box)
    W, H = (x1 - x0) * SS, (y1 - y0) * SS
    tile = Image.new('RGB', (W, H), color)
    d = ImageDraw.Draw(tile)
    big = size * SS
    total = runs_width(runs, big)
    cx = (center_x - x0) * SS if center_x is not None else W / 2
    x = cx - total / 2
    base = H * 0.78
    for s, style in runs:
        small = style.startswith('s')
        f = ImageFont.truetype(FONT_I if style in ('i', 'si') else FONT_R, int(big * (0.68 if small else 1)))
        dy = 0
        if style in ('sr', 'si'):
            dy = big * 0.22
        elif style == 'sup':
            dy = -big * 0.42
        d.text((x, base + dy), s, font=f, fill=INK, anchor='ls')
        x += f.getlength(s)
    tile = tile.resize((x1 - x0, y1 - y0), Image.LANCZOS)
    im.paste(tile, (x0, y0))


# Graph-GRU gate lines (box = covered area; centres taken from the original lines)
patch((1026, 716, 1176, 736), [('r', 'i'), (' = ', 'r'), ('σ', 'i'), ('(', 'r'), ('G', 'r'), ('r', 'si'),
                                ('([', 'r'), ('x', 'i'), (', ', 'r'), ('h', 'i'), ('], ', 'r'), ('S', 'i'),
                                ('))', 'r')], 17, center_x=1101)
patch((1022, 744, 1174, 764), [('u', 'i'), (' = ', 'r'), ('σ', 'i'), ('(', 'r'), ('G', 'r'), ('u', 'si'),
                                ('([', 'r'), ('x', 'i'), (', ', 'r'), ('h', 'i'), ('], ', 'r'), ('S', 'i'),
                                ('))', 'r')], 17, center_x=1097)
patch((1004, 771, 1198, 793), [('c', 'i'), (' = tanh(', 'r'), ('G', 'r'), ('c', 'si'), ('([', 'r'), ('x', 'i'),
                                (', ', 'r'), ('r', 'i'), (' ⊙ ', 'r'), ('h', 'i'), ('], ', 'r'), ('S', 'i'),
                                ('))', 'r')], 17, center_x=1100)
# refinement feature label
patch((1322, 563, 1428, 584), [('[', 'r'), ('Y', 'i'), ('0', 'sr'), (', ', 'r'), ('P', 'i'), ('+', 'sup'),
                                ('Y', 'i'), ('0', 'sr'), (', ', 'r'), ('P', 'i'), ('−', 'sup'), ('Y', 'i'),
                                ('0', 'sr'), (']', 'r')], 16, center_x=1374)
im.save(OUT)
print(OUT, im.size)
