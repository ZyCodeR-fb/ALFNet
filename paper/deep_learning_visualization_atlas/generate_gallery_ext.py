#!/usr/bin/env python3
"""Extension of the visualization atlas (2026-11-16): 48 further chart types.

Four categories that the first 75 charts do not cover: 3D / multi-dimensional
parameter spaces, composite and multi-panel figures, structure / relation /
attention figures, and time-series / spatio-temporal case figures.
All data are synthetic. Needs numpy, matplotlib and scipy only (no seaborn),
and does not redraw the original 75 figures.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle, PathPatch, Polygon
from matplotlib.path import Path as MPath
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from scipy.ndimage import gaussian_filter, gaussian_filter1d

ROOT = Path(__file__).resolve().parent
PNG = ROOT / 'figures'
SVG = ROOT / 'figures_svg'
PNG.mkdir(exist_ok=True)
SVG.mkdir(exist_ok=True)
R = np.random.default_rng(20261116)
C = {'blue': '#4179AE', 'teal': '#189D8F', 'gold': '#EDAC4A', 'red': '#D76670', 'purple': '#8C77B7',
     'ink': '#243449', 'pale': '#E7ECF2', 'gray': '#8593A2', 'green': '#68A75C'}
SERIES = [C['blue'], C['gold'], C['purple'], C['teal'], C['red'], C['green']]
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9.4, 'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'axes.labelsize': 9.2, 'axes.edgecolor': '#B4BEC9', 'axes.labelcolor': C['ink'], 'xtick.color': C['ink'],
    'ytick.color': C['ink'], 'text.color': C['ink'], 'grid.color': '#E4E8ED', 'axes.spines.top': False,
    'axes.spines.right': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'legend.frameon': False, 'axes.axisbelow': True})


def pretty(ax, grid='y'):
    if grid:
        ax.grid(axis=grid, alpha=.8, lw=.8)


def export(slug, title, draw):
    fig = plt.figure(figsize=(6.6, 4.0), dpi=135)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=.115, right=.945, bottom=.18, top=.84)
    draw(fig, ax)
    fig.suptitle(title, x=.085, y=.975, ha='left', fontsize=13, color=C['ink'], weight='bold')
    fig.text(.91, .974, 'SYNTHETIC DATA', ha='right', va='top', fontsize=6.8, color=C['gray'])
    fig.savefig(PNG / (slug + '.png'), dpi=135, facecolor='white')
    fig.savefig(SVG / (slug + '.svg'), facecolor='white')
    plt.close(fig)


def ax3d(fig, ax, elev=24, azim=-58, rect=(.06, .03, .78, .86)):
    fig.delaxes(ax)
    a = fig.add_axes(rect, projection='3d')
    a.view_init(elev, azim)
    for axis in (a.xaxis, a.yaxis, a.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis._axinfo['grid'].update(color='#DDE3EA', linewidth=.6)
    a.tick_params(labelsize=7, pad=0)
    return a


def panels(fig, ax, rows, cols, **kw):
    fig.delaxes(ax)
    grid = dict(left=.1, right=.95, bottom=.17, top=.82, wspace=.32, hspace=.3)
    grid.update(kw)
    gs = fig.add_gridspec(rows, cols, **grid)
    return gs


# --- 9. 3D and multi-dimensional parameter spaces -----------------------------------

def d_scatter3d_stem(fig, ax):
    a = ax3d(fig, ax, 22, -52)
    grid = [.1, .5, .7, .8, .9]
    pts = np.array([(x, y, z) for x in grid for y in grid for z in grid])
    pts = pts[R.choice(len(pts), 46, replace=False)]
    score = .19 - .06 * ((pts[:, 0] - .8) ** 2 + (pts[:, 1] - .8) ** 2 + .6 * (pts[:, 2] - .7) ** 2) + R.normal(0, .002, len(pts))
    for (x, y, z) in pts:
        a.plot([x, x], [y, y], [.1, z], color='#B8C2CC', lw=.6, ls=':')
    sc = a.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=score, cmap='RdYlBu_r', s=34, edgecolor='white', lw=.4, depthshade=False)
    best = pts[score.argmax()]
    a.plot([best[0], best[0]], [best[1], 1.], [best[2], best[2]], color=C['red'], lw=1, ls='--')
    a.plot([.05, best[0]], [best[1], best[1]], [best[2], best[2]], color=C['red'], lw=1, ls='--')
    a.scatter(*best, s=90, facecolor='none', edgecolor=C['red'], lw=1.4, depthshade=False)
    a.set_xlabel(r'$\alpha$ (loss A)', labelpad=-4)
    a.set_ylabel(r'$\beta$ (loss B)', labelpad=-4)
    a.set_zlabel(r'$\gamma$ (loss C)', labelpad=-6)
    a.set(xlim=(.05, 1), ylim=(.05, 1), zlim=(.1, .95))
    cb = fig.colorbar(sc, ax=a, shrink=.62, pad=.06)
    cb.set_label('Validation score')


def d_bar3d(fig, ax):
    a = ax3d(fig, ax, 26, -55)
    xs, ys = np.meshgrid(np.arange(5), np.arange(4), indexing='ij')
    z = 3.6 - .5 * np.exp(-((xs - 2.6) ** 2 / 3 + (ys - 1.8) ** 2 / 2)) + R.normal(0, .015, xs.shape)
    colors = plt.cm.viridis_r((z - z.min()) / (z.max() - z.min()))
    a.bar3d(xs.ravel() - .3, ys.ravel() - .3, np.full(z.size, 3.0), .6, .6, (z - 3.0).ravel(),
            color=colors.reshape(-1, 4), edgecolor='white', lw=.3, shade=True)
    a.set_xticks(range(5))
    a.set_xticklabels(['16', '32', '64', '128', '256'])
    a.set_yticks(range(4))
    a.set_yticklabels(['1', '2', '3', '4'])
    a.set_xlabel('Hidden size', labelpad=-4)
    a.set_ylabel('Layers', labelpad=-4)
    a.set_zlabel('MAE', labelpad=-6)
    a.set_zlim(3.0, 3.65)


def _surface():
    x = np.linspace(-2.5, 2.5, 60)
    xx, yy = np.meshgrid(x, x)
    zz = .25 * (xx ** 2 + .6 * yy ** 2) + .5 * np.sin(1.6 * xx) * np.cos(1.3 * yy) + 1.2
    return xx, yy, zz


def d_surface_contour(fig, ax):
    a = ax3d(fig, ax, 28, -60)
    xx, yy, zz = _surface()
    a.plot_surface(xx, yy, zz, cmap='viridis', alpha=.88, lw=0, antialiased=True, rstride=2, cstride=2)
    a.contourf(xx, yy, zz, zdir='z', offset=-.6, cmap='viridis', levels=14, alpha=.9)
    a.set_zlim(-.6, zz.max())
    a.set_xlabel('log learning rate', labelpad=-4)
    a.set_ylabel('log weight decay', labelpad=-4)
    a.set_zlabel('Validation loss', labelpad=-6)


def d_wireframe(fig, ax):
    a = ax3d(fig, ax, 24, -62)
    xx, yy, zz = _surface()
    a.plot_wireframe(xx, yy, zz + .7, rstride=5, cstride=5, color=C['gray'], lw=.7, label='Baseline')
    a.plot_wireframe(xx, yy, zz * .8, rstride=5, cstride=5, color=C['teal'], lw=.9, label='Proposed')
    a.set_xlabel('Parameter A', labelpad=-4)
    a.set_ylabel('Parameter B', labelpad=-4)
    a.set_zlabel('Error', labelpad=-6)
    a.legend(loc='upper left', fontsize=7, bbox_to_anchor=(.0, .95))


def d_waterfall3d(fig, ax):
    a = ax3d(fig, ax, 26, -66)
    x = np.linspace(0, 8, 160)
    steps = [3, 6, 9, 12]
    verts = []
    for i, k in enumerate(steps):
        y = np.exp(-(x - 1.6 - .32 * i) ** 2 / (.5 + .25 * i)) * (1 - .12 * i)
        verts.append([(x[0], 0), *zip(x, y), (x[-1], 0)])
    poly = PolyCollection(verts, facecolors=[plt.cm.viridis(v) for v in np.linspace(.15, .85, 4)], alpha=.72, edgecolor='white')
    a.add_collection3d(poly, zs=range(4), zdir='y')
    a.set(xlim=(0, 8), ylim=(-.3, 3.3), zlim=(0, 1.05))
    a.set_yticks(range(4))
    a.set_yticklabels([f'h={k}' for k in steps])
    a.set_xlabel('Absolute error', labelpad=-4)
    a.set_zlabel('Density', labelpad=-6)


def d_trisurf(fig, ax):
    a = ax3d(fig, ax, 27, -58)
    p = R.uniform(-2.4, 2.4, (70, 2))
    z = .25 * (p[:, 0] ** 2 + .6 * p[:, 1] ** 2) + .4 * np.sin(1.6 * p[:, 0]) + 1.2
    a.plot_trisurf(p[:, 0], p[:, 1], z, cmap='viridis', alpha=.75, lw=.2, edgecolor='white')
    a.scatter(p[:, 0], p[:, 1], z, c=C['ink'], s=8, depthshade=False)
    a.set_xlabel('Parameter A', labelpad=-4)
    a.set_ylabel('Parameter B', labelpad=-4)
    a.set_zlabel('Loss', labelpad=-6)


def d_loss_landscape(fig, ax):
    a = ax3d(fig, ax, 34, -50)
    xx, yy, zz = _surface()
    a.plot_surface(xx, yy, zz, cmap='coolwarm', alpha=.62, lw=0, rstride=2, cstride=2)
    t = np.linspace(0, 1, 40)
    px = 2.2 * (1 - t) ** 1.4 * np.cos(4 * t) - .55 * t
    py = 2.0 * (1 - t) ** 1.2 * np.sin(3 * t)
    pz = .25 * (px ** 2 + .6 * py ** 2) + .5 * np.sin(1.6 * px) * np.cos(1.3 * py) + 1.2 + .04
    a.plot(px, py, pz, color=C['ink'], lw=1.8)
    a.scatter(px[[0, -1]], py[[0, -1]], pz[[0, -1]], c=[C['gold'], C['teal']], s=50, depthshade=False, zorder=5)
    a.set_xlabel('Direction 1', labelpad=-4)
    a.set_ylabel('Direction 2', labelpad=-4)
    a.set_zlabel('Loss', labelpad=-6)


def d_scatter3d_embedding(fig, ax):
    a = ax3d(fig, ax, 20, -60)
    centers = np.array([[0, 0, 0], [2.4, 1.2, .6], [.8, 2.6, 2.0], [2.6, -1.0, 2.2]])
    for i, c in enumerate(centers):
        p = c + R.normal(0, .45, (45, 3))
        a.scatter(p[:, 0], p[:, 1], p[:, 2], s=14, color=SERIES[i], alpha=.8, label=f'Class {i + 1}', depthshade=False)
    a.set_xlabel('Component 1', labelpad=-4)
    a.set_ylabel('Component 2', labelpad=-4)
    a.set_zlabel('Component 3', labelpad=-6)
    a.legend(loc='upper left', fontsize=7, ncol=2, bbox_to_anchor=(.0, .97))


def d_tensor_slices(fig, ax):
    a = ax3d(fig, ax, 20, -62)
    nodes, horizon = np.meshgrid(np.arange(13), np.arange(9), indexing='ij')
    for i, t in enumerate([0, 1, 2, 3]):
        v = gaussian_filter(R.random((13, 9)), 1.3)
        v = (v - v.min()) / (v.max() - v.min())
        a.plot_surface(np.full_like(nodes, t, dtype=float), nodes, horizon, facecolors=plt.cm.magma(v),
                       shade=False, rstride=1, cstride=1, alpha=.92, lw=0)
    a.set_xticks(range(4))
    a.set_xticklabels(['t-3', 't-2', 't-1', 't'])
    a.set_xlabel('Time slice', labelpad=-4)
    a.set_ylabel('Node', labelpad=-4)
    a.set_zlabel('Feature', labelpad=-6)
    sm = plt.cm.ScalarMappable(cmap='magma')
    sm.set_array([0, 1])
    fig.colorbar(sm, ax=a, shrink=.6, pad=.06, label='Normalized value')


def d_contour_trajectory(fig, ax):
    xx, yy, zz = _surface()
    ax.contour(xx, yy, zz, levels=14, cmap='Greys', linewidths=.7)
    ax.contourf(xx, yy, zz, levels=14, cmap='Blues_r', alpha=.55)
    t = np.linspace(0, 1, 22)
    start, end = np.array([2.2, 2.1]), np.array([-.65, 0.])
    for color, label, bend, wiggle in [(C['gold'], 'SGD', 1.5, .16), (C['teal'], 'AdamW', -.5, .0)]:
        decay = 1 - (1 - t) ** 2.2
        path = start + (end - start) * decay[:, None]
        path[:, 1] += bend * np.sin(np.pi * decay) * (1 - decay) + wiggle * np.sin(9 * t) * (1 - t)
        ax.plot(path[:, 0], path[:, 1], '-o', ms=2.8, lw=1.4, color=color, label=label)
    ax.scatter(*start, s=40, color=C['ink'], zorder=4)
    ax.scatter(*end, marker='*', s=130, color=C['red'], edgecolor='white', zorder=4)
    ax.set(xlabel='Direction 1', ylabel='Direction 2')
    ax.legend(loc='lower left', fontsize=7.5)


def d_ternary_heatmap(fig, ax):
    ax.axis('off')
    n = 24
    pts = np.array([(i / n, j / n, 1 - i / n - j / n) for i in range(n + 1) for j in range(n + 1 - i)])
    V = np.array([[.1, .1], [.9, .1], [.5, .86]])
    xy = pts @ V
    val = 1 - ((pts[:, 0] - .5) ** 2 + (pts[:, 1] - .3) ** 2 + (pts[:, 2] - .2) ** 2) * 2.2
    cf = ax.tricontourf(xy[:, 0], xy[:, 1], val, levels=14, cmap='viridis')
    ax.add_patch(Polygon(V, closed=True, fill=False, lw=1.6, edgecolor=C['ink']))
    best = xy[val.argmax()]
    ax.scatter(*best, marker='*', s=150, c=C['red'], edgecolor='white', zorder=5)
    for (x, y), name, dx, dy in zip(V, ['weight A = 1', 'weight B = 1', 'weight C = 1'], [-.03, .03, 0], [-.07, -.07, .05]):
        ax.text(x + dx, y + dy, name, ha='center', fontsize=8.5)
    ax.set(xlim=(0, 1), ylim=(0, .98))
    fig.colorbar(cf, ax=ax, shrink=.75, label='Score')


def d_annotated_heatmap(fig, ax):
    lr = ['1e-4', '3e-4', '1e-3', '3e-3', '1e-2']
    wd = ['0', '1e-5', '1e-4', '1e-3']
    v = 3.2 + .5 * (np.arange(4)[:, None] - 2) ** 2 / 4 + .6 * (np.arange(5)[None] - 2.2) ** 2 / 4 + R.normal(0, .02, (4, 5))
    im = ax.imshow(v, cmap='YlGnBu_r', aspect='auto')
    for i in range(4):
        for j in range(5):
            ax.text(j, i, f'{v[i, j]:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if v[i, j] < np.median(v) else C['ink'])
    i, j = np.unravel_index(v.argmin(), v.shape)
    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor=C['red'], lw=2.2))
    ax.set(xticks=range(5), xticklabels=lr, yticks=range(4), yticklabels=wd, xlabel='Learning rate', ylabel='Weight decay')
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=.85, label='MAE')


def d_bubble_matrix(fig, ax):
    xs, ys = np.meshgrid(np.arange(6), np.arange(4))
    mean = 1 - ((xs - 3.2) ** 2 / 14 + (ys - 1.6) ** 2 / 7) + R.normal(0, .03, xs.shape)
    std = R.uniform(.01, .09, xs.shape)
    sc = ax.scatter(xs, ys, s=60 + 420 * (mean - mean.min()) / (mean.max() - mean.min()), c=std, cmap='YlOrRd', edgecolor=C['ink'], lw=.5)
    ax.set(xticks=range(6), xticklabels=['2', '4', '8', '16', '32', '64'], yticks=range(4), yticklabels=['0.0', '0.1', '0.2', '0.3'],
           xlabel='Patch size', ylabel='Dropout', xlim=(-.6, 5.6), ylim=(-.6, 3.6))
    pretty(ax, None)
    fig.colorbar(sc, ax=ax, shrink=.85, label='Std over seeds')
    ax.text(.0, 1.03, 'bubble area = mean score', transform=ax.transAxes, fontsize=7.5, color=C['gray'])


def d_hexbin(fig, ax):
    y = R.gamma(4, 12, 6000)
    p = y + R.normal(0, 4 + .08 * y)
    hb = ax.hexbin(y, p, gridsize=28, cmap='viridis', mincnt=1, bins='log')
    lim = (0, 150)
    ax.plot(lim, lim, color='white', lw=1.2, ls='--')
    ax.set(xlim=lim, ylim=lim, xlabel='Ground truth', ylabel='Prediction')
    fig.colorbar(hb, ax=ax, shrink=.85, label='Count (log)')


def d_dual_axis(fig, ax):
    x = np.arange(5)
    mae = np.array([3.52, 3.31, 3.22, 3.20, 3.21])
    cost = np.array([1.0, 1.4, 2.1, 3.6, 6.3])
    ax.bar(x, mae, width=.55, color=C['blue'], alpha=.85, label='MAE')
    ax.set(xticks=x, xticklabels=['16', '32', '64', '128', '256'], xlabel='Hidden size', ylabel='MAE', ylim=(3.0, 3.6))
    b = ax.twinx()
    b.plot(x, cost, '-o', color=C['red'], lw=2, label='Training time')
    b.set_ylabel('Relative training time')
    b.spines['right'].set_visible(True)
    b.spines['top'].set_visible(False)
    handles = ax.get_legend_handles_labels()[0] + b.get_legend_handles_labels()[0]
    ax.legend(handles, ['MAE', 'Training time'], loc='upper center', ncol=2, fontsize=7.5)
    pretty(ax)


def d_slice_plot(fig, ax):
    gs = panels(fig, ax, 1, 3, wspace=.12, left=.1, right=.9)
    trial = np.arange(60)
    names = ['log learning rate', 'dropout', 'hidden size (log2)']
    opt = [-3.2, .15, 6.5]
    lo_hi = [(-5, -1.5), (0, .5), (4, 9)]
    first = None
    for i in range(3):
        a = fig.add_subplot(gs[0, i], sharey=first)
        first = first or a
        x = R.uniform(*lo_hi[i], 60)
        y = 3.2 + .6 * ((x - opt[i]) / (lo_hi[i][1] - lo_hi[i][0])) ** 2 * 4 + R.normal(0, .07, 60)
        sc = a.scatter(x, y, c=trial, cmap='Blues', s=16, edgecolor=C['gray'], lw=.3)
        a.set_xlabel(names[i], fontsize=8)
        a.tick_params(labelsize=7)
        a.grid(alpha=.3)
        if i:
            plt.setp(a.get_yticklabels(), visible=False)
        else:
            a.set_ylabel('Objective (MAE)')
    fig.colorbar(sc, ax=fig.axes, shrink=.85, pad=.02, label='Trial')


def d_importance(fig, ax):
    names = ['Learning rate', 'Patch size', 'Hidden size', 'Weight decay', 'Dropout', 'Batch size']
    v = np.array([.41, .22, .15, .11, .07, .04])
    err = np.array([.05, .04, .03, .03, .02, .01])
    y = np.arange(6)
    fig.subplots_adjust(left=.22)
    ax.barh(y, v, xerr=err, color=[C['teal']] + [C['blue']] * 5, height=.6, error_kw=dict(ecolor=C['ink'], capsize=3, lw=1))
    ax.set(yticks=y, yticklabels=names, xlabel='Importance (share of explained variance)')
    ax.invert_yaxis()
    pretty(ax, 'x')


# --- 10. Composite and multi-panel figures ----------------------------------------------

def d_grouped_bar(fig, ax):
    sets = ['Set A', 'Set B', 'Set C', 'Set D']
    models = ['Model A', 'Model B', 'Proposed']
    v = np.array([[3.6, 3.4, 3.1], [2.3, 2.2, 2.0], [4.4, 4.1, 3.9], [3.2, 3.3, 2.9]])
    e = R.uniform(.04, .16, v.shape)
    x = np.arange(4)
    for i, (m, col, hatch) in enumerate(zip(models, [C['gray'], C['blue'], C['teal']], ['//', '..', ''])):
        ax.bar(x + (i - 1) * .26, v[:, i], .24, yerr=e[:, i], color=col, hatch=hatch, edgecolor='white', label=m,
               error_kw=dict(ecolor=C['ink'], capsize=2.5, lw=.9))
    ax.set(xticks=x, xticklabels=sets, ylabel='MAE (mean ± std, 3 seeds)')
    ax.legend(ncol=3, fontsize=7.5, loc='upper center', bbox_to_anchor=(.5, 1.12))
    pretty(ax)


def d_bar_significance(fig, ax):
    names = ['Baseline', 'Variant', 'Proposed']
    v = np.array([3.52, 3.41, 3.12])
    e = np.array([.08, .09, .07])
    ax.bar(names, v, yerr=e, width=.5, color=[C['gray'], C['blue'], C['teal']], error_kw=dict(ecolor=C['ink'], capsize=4))
    for (i, j, h, text) in [(0, 1, 3.72, 'n.s.'), (1, 2, 3.62, '**'), (0, 2, 3.9, '***')]:
        ax.plot([i, i, j, j], [h - .03, h, h, h - .03], color=C['ink'], lw=1)
        ax.text((i + j) / 2, h + .01, text, ha='center', fontsize=9)
    ax.set(ylabel='MAE', ylim=(2.8, 4.05))
    pretty(ax)


def d_broken_axis(fig, ax):
    gs = panels(fig, ax, 2, 1, hspace=.08, height_ratios=[1, 2.2], left=.13)
    top, bottom = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    names = ['A', 'B', 'C', 'D', 'Proposed']
    v = [12.6, 3.9, 3.6, 3.4, 3.1]
    colors = [C['gray']] * 4 + [C['teal']]
    for a in (top, bottom):
        a.bar(names, v, color=colors, width=.55)
        pretty(a)
    top.set_ylim(11.5, 13.2)
    bottom.set_ylim(0, 4.6)
    top.spines['bottom'].set_visible(False)
    top.tick_params(bottom=False, labelbottom=False)
    d = .012
    for a, y in ((top, 0), (bottom, 1)):
        a.plot((-d, d), (y - d * 2, y + d * 2), transform=a.transAxes, color=C['ink'], clip_on=False, lw=1)
    bottom.set_ylabel('MAE')
    top.text(0, 12.75, '12.6', ha='center', fontsize=8)


def d_inset_zoom(fig, ax):
    x = np.arange(1, 101)
    for name, col, final in [('Model A', C['gray'], .262), ('Model B', C['blue'], .251), ('Proposed', C['teal'], .243)]:
        ax.plot(x, final + .5 * np.exp(-x / 9) + R.normal(0, .0015, 100).cumsum() * .15, color=col, lw=1.8, label=name)
    ax.set(xlabel='Epoch', ylabel='Validation loss', ylim=(.2, .78))
    ax.legend(loc='center', fontsize=7.5, bbox_to_anchor=(.42, .34))
    pretty(ax)
    ins = ax.inset_axes([.42, .5, .5, .44])
    for line in ax.get_lines():
        ins.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), lw=1.6)
    ins.set(xlim=(70, 100), ylim=(.236, .27))
    ins.tick_params(labelsize=6.5)
    ins.grid(alpha=.3)
    ax.indicate_inset_zoom(ins, edgecolor=C['ink'], alpha=.7)


def d_diverging_bar(fig, ax):
    names = ['w/o A', 'w/o B', 'w/o C', 'w/o D', 'w/ variant E', 'w/ variant F']
    v = np.array([11.9, 7.8, 5.4, 2.2, -1.1, -2.4])
    y = np.arange(6)
    ax.barh(y, v, color=[C['red'] if t > 0 else C['teal'] for t in v], height=.6)
    ax.axvline(0, color=C['ink'], lw=1)
    for t, yy in zip(v, y):
        ax.text(t + (.3 if t > 0 else -.3), yy, f'{t:+.1f}%', va='center', ha='left' if t > 0 else 'right', fontsize=8)
    ax.set(yticks=y, yticklabels=names, xlabel='Change in MAE relative to the full model (%)', xlim=(-5, 14.5))
    ax.invert_yaxis()
    pretty(ax, 'x')


def d_stacked_100(fig, ax):
    models = ['Model A', 'Model B', 'Model C', 'Proposed']
    share = np.array([[.22, .31, .27, .20], [.25, .28, .26, .21], [.30, .24, .24, .22], [.27, .25, .25, .23]])
    left = np.zeros(4)
    for k, (name, col) in enumerate(zip(['Unseen', 'Steady', 'Join', 'Fault'], [C['blue'], C['teal'], C['gold'], C['red']])):
        ax.barh(models, share[:, k], left=left, color=col, label=name, height=.55, edgecolor='white')
        for i in range(4):
            ax.text(left[i] + share[i, k] / 2, i, f'{share[i, k] * 100:.0f}%', ha='center', va='center', color='white', fontsize=7.5)
        left += share[:, k]
    ax.set(xlim=(0, 1), xlabel='Share of total absolute error')
    ax.invert_yaxis()
    ax.legend(ncol=4, fontsize=7.5, loc='upper center', bbox_to_anchor=(.5, 1.13))


def d_heatmap_marginal(fig, ax):
    fig.delaxes(ax)
    gs = fig.add_gridspec(2, 2, left=.13, right=.93, bottom=.17, top=.83, width_ratios=[4, 1], height_ratios=[1, 3.2], wspace=.05, hspace=.07)
    main, top, right = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 1])
    v = gaussian_filter(R.random((7, 12)), 1.1) * 4 + 2
    main.imshow(v, aspect='auto', cmap='YlOrRd')
    main.set(xlabel='Forecast step', ylabel='Node group', xticks=range(0, 12, 2), xticklabels=range(1, 13, 2), yticks=range(7), yticklabels=[f'G{i + 1}' for i in range(7)])
    top.bar(range(12), v.mean(0), color=C['blue'], width=.8)
    top.set_xlim(-.5, 11.5)
    top.tick_params(labelbottom=False, labelsize=7)
    right.barh(range(7), v.mean(1), color=C['blue'], height=.8)
    right.set_ylim(6.5, -.5)
    right.tick_params(labelleft=False, labelsize=7)
    for a in (main,):
        for s in a.spines.values():
            s.set_visible(False)


def d_clustered_heatmap(fig, ax):
    fig.delaxes(ax)
    gs = fig.add_gridspec(1, 2, left=.05, right=.9, bottom=.17, top=.83, width_ratios=[1, 4.2], wspace=.02)
    den, main = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    base = np.repeat(R.normal(0, 1, (3, 10)), [5, 4, 5], axis=0) + R.normal(0, .35, (14, 10))
    base = base[R.permutation(14)]
    link = linkage(base, 'average')
    dendrogram(link, orientation='left', ax=den, color_threshold=0, above_threshold_color=C['gray'], no_labels=True)
    den.axis('off')
    order = leaves_list(link)[::-1]
    im = main.imshow(base[order], aspect='auto', cmap='RdBu_r', vmin=-2.4, vmax=2.4)
    main.set(xlabel='Feature', yticks=[], xticks=range(10), xticklabels=[f'f{i}' for i in range(10)])
    for s in main.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=main, shrink=.85, label='Standardized value')


def d_joint_plot(fig, ax):
    fig.delaxes(ax)
    gs = fig.add_gridspec(2, 2, left=.13, right=.92, bottom=.17, top=.83, width_ratios=[4, 1], height_ratios=[1, 3.4], wspace=.05, hspace=.06)
    main, top, right = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 1])
    for col, mu in [(C['gold'], (3.8, 3.6)), (C['teal'], (3.0, 3.1))]:
        p = R.multivariate_normal(mu, [[.25, .16], [.16, .22]], 220)
        main.scatter(p[:, 0], p[:, 1], s=9, color=col, alpha=.55)
        top.hist(p[:, 0], bins=24, color=col, alpha=.6)
        right.hist(p[:, 1], bins=24, color=col, alpha=.6, orientation='horizontal')
    main.set(xlabel='Error at step 3', ylabel='Error at step 12')
    top.tick_params(labelbottom=False, labelsize=7)
    right.tick_params(labelleft=False, labelsize=7)
    top.set_xlim(main.get_xlim())
    right.set_ylim(main.get_ylim())
    main.grid(alpha=.3)


def d_rank_heatmap(fig, ax):
    models = ['Model A', 'Model B', 'Model C', 'Model D', 'Proposed']
    sets = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
    rank = np.array([R.permutation(5) + 1 for _ in sets]).T
    rank[4] = np.minimum(rank[4], 2)
    for j in range(len(sets)):
        rank[:, j] = np.argsort(np.argsort(rank[:, j] + np.linspace(0, .1, 5))) + 1
    im = ax.imshow(rank, cmap='YlGn_r', aspect='auto', vmin=1, vmax=5)
    for i in range(5):
        for j in range(8):
            ax.text(j, i, str(rank[i, j]), ha='center', va='center', fontsize=8.5, color='white' if rank[i, j] <= 2 else C['ink'])
    ax.set(xticks=range(8), xticklabels=sets, yticks=range(5), yticklabels=models)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=.85, label='Rank (1 = best)', ticks=range(1, 6))


def d_rose(fig, ax):
    fig.delaxes(ax)
    a = fig.add_axes([.2, .08, .6, .72], polar=True)
    n = 8
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    v = np.array([.82, .64, .71, .55, .9, .6, .47, .76])
    a.bar(theta, v, width=2 * np.pi / n * .92, color=plt.cm.viridis(v / v.max() * .85), edgecolor='white', lw=1)
    a.set_xticks(theta)
    a.set_xticklabels([f'Task {i + 1}' for i in range(n)], fontsize=7.5)
    a.set_yticks([.25, .5, .75, 1])
    a.set_yticklabels([])
    a.spines['polar'].set_visible(False)
    a.grid(alpha=.35)


def d_paired_plot(fig, ax):
    base = R.normal(3.6, .3, 14)
    prop = base - R.normal(.28, .16, 14)
    for b, p in zip(base, prop):
        ax.plot([0, 1], [b, p], color=C['teal'] if p < b else C['red'], lw=1, alpha=.7)
    ax.scatter(np.zeros(14), base, color=C['gray'], s=28, zorder=3)
    ax.scatter(np.ones(14), prop, color=C['teal'], s=28, zorder=3)
    ax.boxplot([base, prop], positions=[-.28, 1.28], widths=.16, patch_artist=True, showfliers=False,
               boxprops=dict(facecolor=C['pale'], edgecolor=C['gray']), medianprops=dict(color=C['ink']))
    ax.set(xticks=[0, 1], xticklabels=['Baseline', 'Proposed'], xlim=(-.55, 1.55), ylabel='MAE per dataset / seed')
    pretty(ax)


def d_pyramid(fig, ax):
    stages = ['k = 1–3', 'k = 4–6', 'k = 7–9', 'k = 10–12']
    left = np.array([4.4, 3.9, 3.5, 3.3])
    right = np.array([3.4, 3.8, 4.3, 4.9])
    y = np.arange(4)
    ax.barh(y, -left, color=C['blue'], height=.6, label='Join')
    ax.barh(y, right, color=C['gold'], height=.6, label='Fault')
    for i in range(4):
        ax.text(-left[i] - .1, i, f'{left[i]:.1f}', ha='right', va='center', fontsize=8)
        ax.text(right[i] + .1, i, f'{right[i]:.1f}', ha='left', va='center', fontsize=8)
    ax.axvline(0, color=C['ink'], lw=1)
    ax.set(yticks=y, yticklabels=stages, xlim=(-6, 6), xlabel='MAE')
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_xticklabels(['4', '2', '0', '2', '4'])
    ax.invert_yaxis()
    ax.legend(ncol=2, fontsize=7.5, loc='upper center', bbox_to_anchor=(.5, 1.12))


# --- 11. Structure, relations and attention ---------------------------------------------

def d_adjacency_compare(fig, ax):
    gs = panels(fig, ax, 1, 2, wspace=.12, left=.09, right=.9)
    n = 24
    idx = np.arange(n)
    pre = np.exp(-np.abs(idx[:, None] - idx[None]) / 2.2)
    pre[pre < .2] = 0
    learned = pre * .6 + gaussian_filter(R.random((n, n)), 1) * .5
    learned[8:14, 16:22] += .45
    for i, (m, title) in enumerate([(pre, 'Predefined (distance)'), (learned, 'Learned (adaptive)')]):
        a = fig.add_subplot(gs[0, i])
        im = a.imshow(m, cmap='magma', vmin=0, vmax=1.1)
        a.set_title(title, fontsize=9, weight='normal')
        a.set(xticks=[], yticks=[], xlabel='Target node')
        if i == 0:
            a.set_ylabel('Source node')
        for s in a.spines.values():
            s.set_visible(False)
    fig.colorbar(im, ax=fig.axes, shrink=.8, pad=.02, label='Edge weight')


def d_attention_map(fig, ax):
    n = 12
    idx = np.arange(n)
    att = np.exp(-(idx[:, None] - idx[None]) ** 2 / 6) + .5 * np.exp(-(idx[None] - 11) ** 2 / 3) + R.random((n, n)) * .15
    att /= att.sum(1, keepdims=True)
    im = ax.imshow(att, cmap='Blues', aspect='auto')
    ax.set(xticks=idx, xticklabels=[f't-{11 - i}' if i < 11 else 't' for i in idx], yticks=idx, yticklabels=[f'h{i + 1}' for i in idx],
           xlabel='Input step (key)', ylabel='Forecast step (query)')
    ax.tick_params(labelsize=7)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=.85, label='Attention weight')


def d_arc_diagram(fig, ax):
    n = 12
    ax.scatter(range(n), np.zeros(n), s=120, color=C['blue'], zorder=3)
    for i in range(n):
        ax.text(i, -.09, f'v{i + 1}', ha='center', va='top', fontsize=7.5)
    for _ in range(16):
        i, j = sorted(R.choice(n, 2, replace=False))
        w = R.uniform(.3, 1)
        t = np.linspace(0, np.pi, 60)
        ax.plot((i + j) / 2 + (j - i) / 2 * np.cos(t), (j - i) / 8 * np.sin(t), color=C['teal'], lw=.6 + 2.6 * w, alpha=.25 + .5 * w)
    ax.set(ylim=(-.25, 1.5), xlim=(-.6, n - .4))
    ax.axis('off')


def d_chord(fig, ax):
    ax.axis('off')
    ax.set_aspect('equal')
    n = 8
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    xy = np.c_[np.cos(ang), np.sin(ang)]
    for _ in range(14):
        i, j = R.choice(n, 2, replace=False)
        w = R.uniform(.2, 1)
        path = MPath([xy[i], (0, 0), xy[j]], [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3])
        ax.add_patch(PathPatch(path, fill=False, edgecolor=SERIES[i % 6], lw=.8 + 4 * w, alpha=.45))
    ax.scatter(xy[:, 0], xy[:, 1], s=260, color=[SERIES[i % 6] for i in range(n)], edgecolor='white', zorder=3)
    for i, (x, y) in enumerate(xy):
        ax.text(x * 1.2, y * 1.2, f'Region {i + 1}', ha='center', va='center', fontsize=7.5)
    ax.set(xlim=(-1.6, 1.6), ylim=(-1.35, 1.35))


def d_corr_triangle(fig, ax):
    n = 9
    a = R.normal(0, 1, (200, n))
    a[:, 1] += a[:, 0]
    a[:, 4] -= .8 * a[:, 3]
    a[:, 7] += .6 * a[:, 2]
    corr = np.corrcoef(a.T)
    mask = np.triu(np.ones((n, n), bool), 1)
    im = ax.imshow(np.ma.array(corr, mask=mask), cmap='RdBu_r', vmin=-1, vmax=1)
    for i in range(n):
        for j in range(i + 1):
            ax.text(j, i, f'{corr[i, j]:.1f}', ha='center', va='center', fontsize=6.8, color='white' if abs(corr[i, j]) > .55 else C['ink'])
    ax.set(xticks=range(n), yticks=range(n), xticklabels=[f'x{i + 1}' for i in range(n)], yticklabels=[f'x{i + 1}' for i in range(n)])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=.85, label='Pearson correlation')


def d_upset(fig, ax):
    fig.delaxes(ax)
    gs = fig.add_gridspec(2, 1, left=.2, right=.94, bottom=.14, top=.83, height_ratios=[2.2, 1], hspace=.06)
    bars, dots = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    combos = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    count = [42, 31, 18, 27, 12, 9, 55]
    order = np.argsort(count)[::-1]
    x = np.arange(7)
    bars.bar(x, np.array(count)[order], color=C['blue'], width=.6)
    bars.set(ylabel='Samples', xlim=(-.6, 6.6))
    bars.tick_params(labelbottom=False, bottom=False)
    pretty(bars)
    for k, o in enumerate(order):
        on = [r for r in range(3) if combos[o][r]]
        dots.scatter([k] * 3, range(3), s=60, color='#D5DCE4', zorder=2)
        dots.scatter([k] * len(on), on, s=60, color=C['ink'], zorder=3)
        if len(on) > 1:
            dots.plot([k, k], [min(on), max(on)], color=C['ink'], lw=2, zorder=3)
    dots.set(yticks=range(3), yticklabels=['Model A correct', 'Model B correct', 'Proposed correct'], xlim=(-.6, 6.6), ylim=(-.6, 2.6))
    dots.tick_params(labelbottom=False, bottom=False, left=False, labelsize=7.5)
    for s in dots.spines.values():
        s.set_visible(False)


def d_treemap(fig, ax):
    ax.axis('off')
    items = [('Graph branch', 46, C['blue']), ('Temporal branch', 27, C['teal']), ('Completion', 15, C['gold']),
             ('Router', 7, C['purple']), ('Head', 5, C['red'])]
    x, y, w, h = 0, 0, 1, 1
    remaining = sum(v for _, v, _ in items)
    for k, (name, v, col) in enumerate(items):
        frac = v / remaining
        if k % 2 == 0:
            rect = (x, y, w * frac, h)
            x, w = x + w * frac, w * (1 - frac)
        else:
            rect = (x, y + h * (1 - frac), w, h * frac)
            h = h * (1 - frac)
        remaining -= v
        ax.add_patch(Rectangle(rect[:2], rect[2], rect[3], facecolor=col, edgecolor='white', lw=2, alpha=.9))
        ax.text(rect[0] + rect[2] / 2, rect[1] + rect[3] / 2, f'{name}\n{v}%', ha='center', va='center', color='white',
                fontsize=8 if v > 10 else 6.5, weight='bold')
    ax.set(xlim=(0, 1), ylim=(0, 1))


def d_partition_map(fig, ax):
    centers = np.array([[1, 1], [3.2, 1.4], [2, 3.2], [4.4, 3.4], [.6, 3.4]])
    pts, lab = [], []
    for i, c in enumerate(centers):
        p = c + R.normal(0, .42, (14, 2))
        pts.append(p)
        lab += [i] * 14
    pts, lab = np.vstack(pts), np.array(lab)
    for i in range(len(pts)):
        d = np.linalg.norm(pts - pts[i], axis=1)
        for j in np.argsort(d)[1:3]:
            ax.plot(*zip(pts[i], pts[j]), color='#C3CCD6', lw=.6, zorder=1)
    for i in range(5):
        p = pts[lab == i]
        from scipy.spatial import ConvexHull
        hull = ConvexHull(p)
        ax.add_patch(Polygon(p[hull.vertices], closed=True, facecolor=SERIES[i], alpha=.13, edgecolor=SERIES[i], lw=1))
        ax.scatter(p[:, 0], p[:, 1], s=22, color=SERIES[i], zorder=3, label=f'Part {i + 1}')
    ax.set(xlabel='Longitude (a.u.)', ylabel='Latitude (a.u.)')
    ax.legend(fontsize=7, ncol=5, loc='upper center', bbox_to_anchor=(.5, 1.12), handletextpad=.1, columnspacing=.8)


# --- 12. Time series and spatio-temporal cases --------------------------------------------

def _traffic(n, seed=0):
    r = np.random.default_rng(seed)
    t = np.arange(n)
    return 62 - 18 * np.exp(-((t % 288) - 100) ** 2 / 900) - 22 * np.exp(-((t % 288) - 210) ** 2 / 1400) + gaussian_filter1d(r.normal(0, 4, n), 2)


def d_forecast_case(fig, ax):
    y = _traffic(288, 3)[150:234]
    t = np.arange(len(y))
    origin = 48
    ax.plot(t[:origin + 1], y[:origin + 1], color=C['ink'], lw=1.6, label='History')
    ax.plot(t[origin:], y[origin:], color=C['ink'], lw=1.6, ls='--', label='Ground truth')
    for name, col, bias, noise in [('Model A', C['gold'], 5.5, 1.6), ('Model B', C['blue'], -4, 1.2), ('Proposed', C['teal'], 1.0, .7)]:
        pred = y[origin:] + bias * np.linspace(.2, 1, len(y) - origin) + gaussian_filter1d(R.normal(0, noise * 2, len(y) - origin), 2)
        ax.plot(t[origin:], pred, color=col, lw=1.8, label=name)
    ax.axvline(origin, color=C['gray'], lw=1, ls=':')
    ax.axvspan(30, 42, color=C['red'], alpha=.12, lw=0)
    ax.text(36, y.min() - 1.5, 'missing', ha='center', fontsize=7.5, color=C['red'])
    ax.set(xlabel='Time step (5 min)', ylabel='Speed (mph)')
    ax.legend(ncol=5, fontsize=7, loc='upper center', bbox_to_anchor=(.5, 1.13), columnspacing=.9, handlelength=1.5)
    pretty(ax)


def d_missing_pattern(fig, ax):
    mask = (R.random((26, 120)) > .06).astype(float)
    mask[4, 30:75] = 0
    mask[11, :48] = 0
    mask[17, 82:] = 0
    mask[21, 55:63] = 0
    ax.imshow(mask, aspect='auto', cmap=matplotlib.colors.ListedColormap(['#F1F3F6', C['blue']]), interpolation='nearest')
    for row, x, text in [(11, 50, 'joins'), (17, 60, 'fails'), (4, 77, 'block missing')]:
        ax.text(x, row, text, va='center', fontsize=7.5, color=C['red'], weight='bold')
    ax.set(xlabel='Time step', ylabel='Sensor')
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=C['blue']), Rectangle((0, 0), 1, 1, color='#F1F3F6', ec=C['gray'])],
              labels=['observed', 'unavailable'], ncol=2, fontsize=7.5, loc='upper center', bbox_to_anchor=(.5, 1.12))


def d_hour_weekday(fig, ax):
    h = np.arange(24)
    profile = 1 + 1.6 * np.exp(-(h - 8) ** 2 / 5) + 1.9 * np.exp(-(h - 17.5) ** 2 / 7)
    v = np.vstack([profile * s for s in [1, 1.02, 1.03, 1.05, 1.1, .55, .45]]) + R.normal(0, .05, (7, 24))
    im = ax.imshow(v, aspect='auto', cmap='YlOrRd')
    ax.set(xticks=range(0, 24, 3), xlabel='Hour of day', yticks=range(7), yticklabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=.85, label='MAE')


def d_polar_clock(fig, ax):
    fig.delaxes(ax)
    a = fig.add_axes([.2, .07, .6, .72], polar=True)
    h = np.linspace(0, 24, 97)
    theta = h / 24 * 2 * np.pi
    for label, col, scale in [('Weekday', C['blue'], 1), ('Weekend', C['gold'], .55)]:
        v = 1 + scale * (1.6 * np.exp(-(h - 8) ** 2 / 5) + 1.9 * np.exp(-(h - 17.5) ** 2 / 7))
        a.plot(theta, v, color=col, lw=2, label=label)
        a.fill(theta, v, color=col, alpha=.12)
    a.set_theta_zero_location('N')
    a.set_theta_direction(-1)
    a.set_xticks(np.arange(0, 24, 3) / 24 * 2 * np.pi)
    a.set_xticklabels([f'{i}:00' for i in range(0, 24, 3)], fontsize=7.5)
    a.set_yticklabels([])
    a.grid(alpha=.35)
    a.spines['polar'].set_visible(False)
    a.legend(fontsize=7.5, loc='upper right', bbox_to_anchor=(1.42, 1.1))


def d_periodogram(fig, ax):
    n = 288 * 14
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 288) + .5 * np.sin(2 * np.pi * t / 144) + .35 * np.sin(2 * np.pi * t / (288 * 7)) + R.normal(0, .6, n)
    power = np.abs(np.fft.rfft(x - x.mean())) ** 2 / n
    period_h = 1 / np.fft.rfftfreq(n, d=1 / 12)[1:]
    ax.plot(period_h, power[1:], color=C['blue'], lw=1.2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    for p, name in [(12, '12 h'), (24, '24 h'), (168, '1 week')]:
        ax.axvline(p, color=C['red'], lw=.8, ls='--')
        ax.text(p * 1.06, power.max() * .5, name, fontsize=7.5, color=C['red'])
    ax.set(xlabel='Period (hours)', ylabel='Power', xlim=(1, 400))
    pretty(ax)


def d_acf(fig, ax):
    x = _traffic(288 * 6, 5)
    x = x - x.mean()
    lags = np.arange(0, 300, 6)
    acf = np.array([1 if k == 0 else np.corrcoef(x[:-k], x[k:])[0, 1] for k in lags])
    ax.vlines(lags, 0, acf, color=C['blue'], lw=1.6)
    ax.scatter(lags, acf, s=14, color=C['blue'], zorder=3)
    band = 1.96 / np.sqrt(len(x))
    ax.axhspan(-band, band, color=C['gray'], alpha=.2, lw=0)
    ax.axhline(0, color=C['ink'], lw=.8)
    ax.set(xlabel='Lag (time steps, 5 min)', ylabel='Autocorrelation')
    pretty(ax)


def d_streamgraph(fig, ax):
    t = np.linspace(0, 24, 120)
    parts = [np.clip(1 + a * np.exp(-(t - m) ** 2 / s) + .1 * np.sin(t + k), .05, None)
             for k, (a, m, s) in enumerate([(2.2, 8, 6), (1.6, 12.5, 14), (2.6, 17.5, 8), (.9, 21, 5)])]
    ax.stackplot(t, parts, baseline='wiggle', colors=[C['blue'], C['teal'], C['gold'], C['purple']], labels=['Type A', 'Type B', 'Type C', 'Type D'], alpha=.9)
    ax.set(xlabel='Hour of day', yticks=[], xlim=(0, 24))
    ax.spines['left'].set_visible(False)
    ax.legend(ncol=4, fontsize=7.5, loc='upper center', bbox_to_anchor=(.5, 1.12))


def d_time_space(fig, ax):
    t = np.linspace(6, 11, 150)
    s = np.linspace(0, 12, 60)
    tt, ss = np.meshgrid(t, s)
    speed = 65 - 42 * np.exp(-((ss - 8 + 2.4 * (tt - 7.4)) ** 2) / 2.4) * np.exp(-(tt - 8.2) ** 2 / 1.6) + R.normal(0, 1.2, tt.shape)
    pm = ax.pcolormesh(tt, ss, speed, cmap='RdYlGn', shading='auto', vmin=20, vmax=68)
    ax.set(xlabel='Time of day (h)', ylabel='Position along corridor (km)')
    fig.colorbar(pm, ax=ax, shrink=.85, label='Speed (mph)')


def d_stacked_panels(fig, ax):
    gs = panels(fig, ax, 3, 1, hspace=.12, left=.13, top=.84)
    t = np.arange(120)
    y = _traffic(288, 8)[60:180]
    series = [(y, 'Speed', C['ink']), (np.abs(gaussian_filter1d(R.normal(0, 3, 120), 2)) + 3 * ((t > 55) & (t < 80)), '|Error|', C['red']),
              (np.clip(.25 + .5 / (1 + np.exp(-(t - 58) / 3)) - .45 / (1 + np.exp(-(t - 82) / 3)) + R.normal(0, .015, 120), 0, 1), 'Gate', C['teal'])]
    first = None
    for i, (v, name, col) in enumerate(series):
        a = fig.add_subplot(gs[i], sharex=first)
        first = first or a
        a.plot(t, v, color=col, lw=1.5)
        a.axvspan(55, 80, color=C['gold'], alpha=.16, lw=0)
        a.set_ylabel(name, fontsize=8)
        a.tick_params(labelsize=7)
        a.grid(alpha=.3)
        if i < 2:
            plt.setp(a.get_xticklabels(), visible=False)
    a.set_xlabel('Time step')


def d_robustness_curve(fig, ax):
    rate = np.array([0, 10, 20, 30, 40, 50, 60, 70])
    for name, col, slope, marker in [('Model A', C['gray'], .034, 's'), ('Model B', C['blue'], .025, '^'), ('Model C', C['gold'], .021, 'D'), ('Proposed', C['teal'], .011, 'o')]:
        m = 3.1 + slope * rate + .00012 * slope * 40 * rate ** 2 + (0 if name == 'Proposed' else .12)
        sd = .04 + .002 * rate
        ax.plot(rate, m, marker=marker, ms=4.5, color=col, lw=1.8, label=name)
        ax.fill_between(rate, m - sd, m + sd, color=col, alpha=.15, lw=0)
    ax.set(xlabel='Missing rate (%)', ylabel='MAE (mean ± std)')
    ax.legend(fontsize=7.5, loc='upper left')
    pretty(ax)


EXT_CATEGORIES = [
    ('三维与多维参数空间', '多个参数或多个维度如何共同影响结果？', [
        ('scatter3d_stem', '3D Scatter with Drop Lines', '三维散点图（带垂线）', '三个损失权重或三个超参数的搜索空间，颜色表示得分', '三维位置难以精确读数：加垂线或投影线，标出最优点，并在正文或附录给出数值表。', d_scatter3d_stem),
        ('bar3d_grid', '3D Bar Chart', '三维柱状图', '两个离散超参数的网格结果', '后排柱子会被遮挡；网格大于约 6×6 时改用带数值的热力图。', d_bar3d),
        ('surface_contour_projection', '3D Surface with Contour Projection', '三维曲面＋底面等高线投影', '两个连续参数的响应曲面，投影便于读出最优区域', '曲面来自插值时须标出真实采样点；透视会夸大或缩小差异。', d_surface_contour),
        ('wireframe_3d', '3D Wireframe Comparison', '三维线框对比图', '两个模型在同一参数平面上的误差曲面对比', '最多画两层曲面；曲面交叉时读者难以判断高低。', d_wireframe),
        ('waterfall_3d', '3D Waterfall / Stacked Curves', '三维瀑布图（层叠曲线）', '多个条件（预测步长、层、时间）下的曲线或分布', '条件多于 6 个时改用山峦图或热力图。', d_waterfall3d),
        ('trisurf_3d', '3D Triangulated Surface', '三角剖分曲面图', '随机搜索等不规则采样点构成的响应曲面', '三角面片是线性插值，不代表采样点之间的真实取值。', d_trisurf),
        ('loss_landscape_3d', 'Loss Landscape with Trajectory', '损失地形与优化轨迹', '沿两个方向的损失曲面及训练轨迹', '方向的选取与归一化方式（如 filter normalization）必须说明；不同模型的地形不可直接比较尺度。', d_loss_landscape),
        ('scatter3d_embedding', '3D Embedding Scatter', '三维嵌入散点图', '降维到三维的表示及其类别结构', '视角会改变簇的可分程度；最好同时给二维投影。', d_scatter3d_embedding),
        ('tensor_slices', 'Stacked Tensor Slices', '张量切片堆叠图', '时间 × 节点 × 特征的三阶数据或注意力张量', '切片之间有遮挡，只适合展示 3–5 个切片。', d_tensor_slices),
        ('contour_trajectory', 'Contour with Optimization Paths', '等高线＋优化路径图', '不同优化器或训练策略在损失平面上的路径', '路径是高维轨迹的投影，投影方法须说明。', d_contour_trajectory),
        ('ternary_heatmap', 'Ternary Contour Plot', '三元等值图', '和为 1 的三个权重上的连续响应', '只适用于三者非负且和固定的情形；三个独立权重应使用三维散点图。', d_ternary_heatmap),
        ('annotated_heatmap', 'Annotated Grid Heatmap', '带数值标注的网格热力图', '两个超参数网格搜索的精确数值与最优格', '颜色映射方向要与“越低越好”一致；标出最优格并说明选择依据（验证集）。', d_annotated_heatmap),
        ('bubble_matrix', 'Bubble Matrix (Balloon Plot)', '气泡矩阵图', '网格上同时展示均值（面积）与波动（颜色）', '面积与数值成正比；图例须给出面积含义。', d_bubble_matrix),
        ('hexbin_density', 'Hexbin Density Plot', '六边形分箱密度图', '样本量很大时的预测—真值或两变量联合分布', '计数跨度大时用对数颜色并标明。', d_hexbin),
        ('dual_axis_bar_line', 'Dual-Axis Bar–Line Chart', '双轴柱线组合图', '同一参数下的精度（柱）与开销（线）', '两个纵轴的范围选择会影响观感；两轴都要标注单位，避免暗示相关。', d_dual_axis),
        ('slice_plot', 'Hyperparameter Slice Plot', '超参数切片图', '搜索试验在每个超参数上的边际表现', '边际图掩盖参数之间的交互；颜色若表示试验顺序须说明。', d_slice_plot),
        ('hyperparameter_importance', 'Hyperparameter Importance Plot', '超参数重要性图', '各超参数对结果方差的贡献（如 fANOVA）', '重要性依赖搜索范围与估计方法，不代表因果。', d_importance),
    ]),
    ('组合图与多面板', '如何在一张图里同时给出均值、波动、显著性或局部细节？', [
        ('grouped_bar_error', 'Grouped Bar Chart with Error Bars', '分组柱状图（误差线＋填充纹理）', '多个数据集上多个模型的均值与波动', '误差线含义（SD、SE、CI）与重复次数写入图注；纹理保证灰度打印可分辨。', d_grouped_bar),
        ('bar_significance', 'Bar Chart with Significance Brackets', '带显著性标记的柱状图', '成对比较的统计检验结果', '星号必须来自实际检验，并说明检验方法、样本单位与多重比较校正。', d_bar_significance),
        ('broken_axis_bar', 'Broken-Axis Bar Chart', '断轴柱状图', '个别方法数值远大于其他方法时保留细节', '断轴处必须有明显标记，并标出被截断柱的数值。', d_broken_axis),
        ('inset_zoom', 'Line Plot with Inset Zoom', '带局部放大的折线图', '收敛后期或接近的曲线的细微差别', '放大框与主图的对应关系要画出；放大区域不能只挑有利的一段。', d_inset_zoom),
        ('diverging_bar', 'Diverging Bar Chart', '发散条形图', '消融或变体相对完整模型的增减百分比', '基准与正负方向的含义写清楚（正值表示变差还是变好）。', d_diverging_bar),
        ('stacked_100_bar', '100% Stacked Bar Chart', '百分比堆叠条形图', '误差、耗时或样本在各类别中的占比', '只比较占比，不能比较绝对量；必要时另给总量。', d_stacked_100),
        ('heatmap_marginal', 'Heatmap with Marginal Bars', '带边际统计的热力图', '二维结果及其按行、按列的平均', '边际统计的聚合方式（均值或求和）需说明。', d_heatmap_marginal),
        ('clustered_heatmap', 'Clustered Heatmap with Dendrogram', '聚类热力图（带树状图）', '按相似性重排后的特征、节点或模型矩阵', '聚类距离与连接方式影响排序；标准化方式需说明。', d_clustered_heatmap),
        ('joint_plot', 'Joint Plot with Marginal Histograms', '联合分布图（带边际直方图）', '两个指标的联合分布与各自的分布', '两组样本量不同时，边际直方图用密度而非计数。', d_joint_plot),
        ('rank_heatmap', 'Rank Heatmap', '排名热力图', '多个模型在多个数据集或指标上的名次', '排名隐藏差距大小，应与数值表配合使用。', d_rank_heatmap),
        ('rose_chart', 'Nightingale Rose Chart', '南丁格尔玫瑰图', '少量类别的指标概览（展示性强）', '半径编码使面积随数值平方增长，容易夸大差异；正式比较用柱状图。', d_rose),
        ('paired_plot', 'Paired Comparison Plot', '配对比较图', '同一数据集或种子上两种方法的成对结果', '只连接真正配对的结果；同时报告变好与变差的数量。', d_paired_plot),
        ('pyramid_bar', 'Back-to-Back Bar Chart', '背靠背条形图', '两个场景在同一分组下的对照', '两侧使用相同刻度，否则长度不可比较。', d_pyramid),
    ]),
    ('结构、关系与注意力', '模型学到了怎样的节点关系、注意力或组成结构？', [
        ('adjacency_compare', 'Adjacency Matrix Comparison', '邻接矩阵对比图', '预定义图与学习到的图结构', '节点排序决定视觉上的块结构；两图用同一排序与同一色标。', d_adjacency_compare),
        ('attention_map', 'Attention Map', '注意力权重图', '查询（预测步）对键（输入步或节点）的注意力', '注意力权重不等同于重要性或因果；说明取自哪一层、哪个头或如何平均。', d_attention_map),
        ('arc_diagram', 'Arc Diagram', '弧线图', '一维排列的节点之间的连接强度', '节点顺序影响可读性；边多时只画权重最高的部分并说明阈值。', d_arc_diagram),
        ('chord_diagram', 'Chord Diagram', '弦图', '区域或类别之间的相互流量或依赖', '弦宽应与数量成正比；类别多于约 10 个时难以阅读。', d_chord),
        ('correlation_triangle', 'Triangular Correlation Heatmap', '三角相关性热力图', '变量或节点之间的成对相关', '使用以 0 为中心的发散色标；说明相关系数类型。', d_corr_triangle),
        ('upset_plot', 'UpSet Plot', 'UpSet 集合交集图', '多个模型预测正确或失败样本的重叠', '集合多于 3 个时比韦恩图清晰；交集按规模排序。', d_upset),
        ('treemap', 'Treemap', '矩形树图', '参数量、耗时或数据量的层级组成', '面积与数值成正比；小块标签容易看不清。', d_treemap),
        ('partition_map', 'Graph Partition Map', '图分区示意图', '节点的空间分区、聚类或社区划分', '位置应来自真实坐标或明确标注为抽象布局；凸包只为便于辨认。', d_partition_map),
    ]),
    ('时间序列与时空个例', '具体的时间段、节点或缺失模式下模型表现如何？', [
        ('forecast_case', 'Forecast Case Study', '预测个例图', '历史、真值与多个模型预测的对比', '个例的选取规则须事先确定并写明，不能挑选对本文有利的窗口。', d_forecast_case),
        ('missing_pattern', 'Missingness Pattern Map', '缺失模式图', '传感器 × 时间的观测可用性（随机缺失、块缺失、接入与失效）', '区分自然缺失与人为掩蔽；标注时间范围。', d_missing_pattern),
        ('hour_weekday_heatmap', 'Hour × Weekday Heatmap', '小时 × 星期热力图', '误差或流量的日内与周内模式', '每格的样本量可能不同，必要时标注。', d_hour_weekday),
        ('polar_clock', 'Polar Clock Plot', '极坐标时钟图', '24 小时周期内的指标变化', '极坐标下面积会夸大外圈差异；精确比较仍用折线图。', d_polar_clock),
        ('periodogram', 'Periodogram', '周期图（功率谱）', '序列中的主要周期成分', '说明采样间隔、是否去趋势以及坐标是否取对数。', d_periodogram),
        ('acf_plot', 'Autocorrelation Plot', '自相关图', '序列或残差在不同滞后下的相关性', '置信带基于独立性假设；残差的自相关提示模型未捕捉的结构。', d_acf),
        ('streamgraph', 'Streamgraph', '河流图', '多个类别的量随时间的此消彼长', '基线不固定，难以读取绝对数值；用于展示总体形态。', d_streamgraph),
        ('time_space_diagram', 'Time–Space Diagram', '时空图（走廊速度图）', '沿道路位置与时间的速度或拥堵传播', '位置轴应为真实里程或明确的传感器顺序。', d_time_space),
        ('stacked_panels', 'Stacked Shared-Axis Panels', '共享时间轴的多层面板图', '输入、误差与模型内部量在同一时间轴上的对应', '各面板纵轴独立时须分别标注；事件区间在所有面板中对齐。', d_stacked_panels),
        ('robustness_curve', 'Robustness Curve', '鲁棒性曲线', '缺失率、噪声强度或分布偏移程度增大时的性能变化', '各扰动水平使用相同的评估样本；阴影含义须说明。', d_robustness_curve),
    ]),
]


def overview():
    items = [(slug, en) for _, _, group in EXT_CATEGORIES for slug, en, *_ in group]
    cols = 8
    rows = -(-len(items) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 1.5), dpi=110)
    for a in axes.ravel():
        a.axis('off')
    for a, (slug, en) in zip(axes.ravel(), items):
        a.imshow(plt.imread(PNG / (slug + '.png')))
        a.set_title(en, fontsize=5.5, weight='normal', pad=2)
    fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.96, wspace=.04, hspace=.16)
    fig.savefig(ROOT / 'overview_ext.png', dpi=110, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    count = 0
    for cat, _, group in EXT_CATEGORIES:
        print('Generating:', cat, len(group), flush=True)
        for slug, en, cn, caption, caution, func in group:
            try:
                export(slug, en, func)
            except Exception as exc:
                print('FAILED', slug, type(exc).__name__, str(exc), flush=True)
                raise
            count += 1
    overview()
    print('Saved', count, 'PNG and SVG pairs', flush=True)
