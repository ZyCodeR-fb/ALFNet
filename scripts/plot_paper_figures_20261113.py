"""2026-11-13: analysis figures on four datasets (no training, no inference; reads saved results only).

  fig3  response    arrival / interruption arms, mean window MAE of five models (2 x 4 small multiples)
  fig4  layout      number and temporal layout of the visible readings, five models (4 x 4 grouped bars), on the
                    windows whose focal history varies at least as much as the dataset median (see fig4)
  fig6  ratio       (dot plot) unseen-node ratio scan on METR-LA (test, seed 32; scripts/test_ratio_20261113.py)
  fig7  cases       full interruption: window-error ECDF and one example window per dataset (five models)
  figA1             paired MAE differences to ALFNet for the arms of fig3 (appendix)

fig3/4/7 and figA1 read paper/experiments_20261113/interventions/validation/ (scripts/paired_interventions_20261113.py):
validation, seed-32 final weights, every eligible window, ALFNet and the best method of each comparison group by
validation LifeAvg MAE on that dataset. Bands and intervals are 95% percentile intervals of a bootstrap over
windows (paired for differences; 2,000 resamples, seed 20261113); they do not cover training seeds or node splits.
Every figure is written as pdf/svg/png with its plotted values as csv to paper/experiments_20261113/figures/,
and the pdf is copied to the LaTeX figures folder. summary.json holds the numbers quoted in the text.
"""
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.plot_paper_figures_20261112 import COL, DATASETS, DISPLAY, MARK, WIDTH  # noqa: E402  (also sets the style)

OUT = ROOT / 'paper/experiments_20261113/figures'
TEX = ROOT / 'paper/latex_neurips_20261020/figures'
SRC = ROOT / 'paper/experiments_20261113/interventions/validation'
RATIO = next((ROOT / 'artifacts/evaluation/ratio_test_20261113').glob('*/test_results.json'))
UNIT = {'METRLA': 'mph', 'PEMSBAY': 'mph', 'PEMS04': 'veh/5 min', 'PEMS08': 'veh/5 min'}
COL = dict(COL, PatchSTG=COL['STGCN'])
MARK = dict(MARK, PatchSTG=MARK['STGCN'])
BOOT, BOOT_SEED = 2000, 20261113
TRAJ = ['join_3', 'join_6', 'full', 'fault_3', 'fault_6', 'outage']
TRAJ_NAME = ['Join 3', 'Join 6', 'Full', 'Fault 3', 'Fault 6', 'Outage']
SUMMARY = {}


def save(fig, name, tex_name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    data.to_csv(OUT / f'{name}_values.csv', index=False)
    shutil.copyfile(OUT / f'{name}.pdf', TEX / tex_name)
    print('saved', name, '->', tex_name)


def load(ds):
    """models (ALFNet first, then the group representatives), window MAE [model][condition] -> [W], raw arrays."""
    sel = json.loads((SRC / ds / 'selection.json').read_text())
    assert (SRC / ds / 'COMPLETE.json').exists(), ds
    order = ['ALFNet'] + [m for g in ('unseen', 'incomplete', 'shift', 'backbone')
                          for m, r in sel['models'].items() if r['group'] == g]
    raw = {m: np.load(SRC / ds / f'{m}.npz') for m in order}
    cond = [str(c) for c in raw['ALFNet']['conditions']]
    for m in order:
        assert [str(c) for c in raw[m]['conditions']] == cond
        assert np.array_equal(raw[m]['index'], raw['ALFNet']['index']) and np.array_equal(raw[m]['node'], raw['ALFNet']['node'])
        assert np.allclose(raw[m]['target'], raw['ALFNet']['target'], atol=1e-3)
    mae = {m: {c: np.abs(raw[m]['pred'][i] - raw[m]['target']).mean(1) for i, c in enumerate(cond)} for m in order}
    return order, mae, raw, cond


def stats(ds, order, mae, conditions):
    n = len(mae['ALFNet'][conditions[0]])
    idx = np.random.default_rng(BOOT_SEED).integers(0, n, size=(BOOT, n), dtype=np.int32)

    def ci(v):
        b = v[idx].mean(1)
        return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
    rows = []
    for c in conditions:
        for m in order:
            mean, lo, hi = ci(mae[m][c])
            rows.append(dict(dataset=ds, condition=c, model=m, quantity='MAE', mean=mean, lo=lo, hi=hi, win_rate=np.nan,
                             windows=n))
        for m in order[1:]:
            g = mae[m][c] - mae['ALFNet'][c]
            mean, lo, hi = ci(g)
            rows.append(dict(dataset=ds, condition=c, model=m, quantity='difference', mean=mean, lo=lo, hi=hi,
                             win_rate=float((g > 0).mean()), windows=n))
    return pd.DataFrame(rows)


def summarize(key, s):
    """Counts quoted in the text: arms where ALFNet has the lowest mean MAE; differences whose interval excludes 0."""
    out = {}
    for ds, g in s.groupby('dataset', sort=False):
        mean = g[g.quantity == 'MAE'].pivot(index='condition', columns='model', values='mean')
        d = g[g.quantity == 'difference']
        out[ds] = dict(
            windows=int(g.windows.iloc[0]), arms=int(len(mean)),
            alfnet_lowest=int((mean.idxmin(axis=1) == 'ALFNet').sum()),
            not_lowest={c: mean.loc[c].idxmin() for c in mean.index if mean.loc[c].idxmin() != 'ALFNet'},
            alfnet_mae={c: float(mean.loc[c, 'ALFNet']) for c in mean.index},
            reduction_vs_best_baseline_percent={
                c: float((1 - mean.loc[c, 'ALFNet'] / mean.loc[c].drop('ALFNet').min()) * 100) for c in mean.index},
            intervals_above_zero={m: int((x.lo > 0).sum()) for m, x in d.groupby('model', sort=False)},
            intervals_below_zero={m: int((x.hi < 0).sum()) for m, x in d.groupby('model', sort=False)},
            mean_win_rate={m: float(x.win_rate.mean()) for m, x in d.groupby('model', sort=False)})
    SUMMARY[key] = out


def model_legend(fig, y, backbone=True):
    names = ['ALFNet', 'MoGERNN', 'Merlin', 'STBP', 'STGCN']
    label = dict(STGCN='STGCN / PatchSTG') if backbone else {}
    handles = [plt.Line2D([], [], color=COL[m], marker=MARK[m], mec='white', mew=0.4, lw=1.5 if m == 'ALFNet' else 1.0,
                          ms=4.4 if m == 'ALFNet' else 3.8) for m in names]
    fig.legend(handles, [label.get(m, m) for m in names], loc='upper center', ncol=5, bbox_to_anchor=(0.5, y),
               columnspacing=1.4, handletextpad=0.4)


def fig3(data):
    arrival, interruption = ['join_3', 'join_6', 'full'], ['full', 'fault_3', 'fault_6', 'outage']
    fig, axes = plt.subplots(2, 4, figsize=(WIDTH, 2.9), gridspec_kw=dict(
        wspace=0.34, hspace=0.5, left=0.07, right=0.995, bottom=0.12, top=0.86))
    frames = []
    for j, ds in enumerate(DATASETS):
        order, mae, _, _ = data[ds]
        s = stats(ds, order, mae, TRAJ)
        frames.append(s)
        for i, (arms, ticks) in enumerate(((arrival, ['3', '6', '12']), (interruption, ['0', '3', '6', '12']))):
            ax = axes[i, j]
            for m in order[::-1]:
                v = s[(s.model == m) & (s.quantity == 'MAE')].set_index('condition').loc[arms]
                x = np.arange(len(arms))
                ax.plot(x, v['mean'], color=COL[m], marker=MARK[m], mec='white', mew=0.4,
                        lw=1.5 if m == 'ALFNet' else 1.0, ms=4.2 if m == 'ALFNet' else 3.4, zorder=4 if m == 'ALFNet' else 3)
            ax.set_xticks(range(len(arms)), ticks)
            ax.set_xlim(-0.25, len(arms) - 0.75)
            ax.grid(axis='y')
            if i == 0:
                ax.set_title(f'{DISPLAY[ds]} ({UNIT[ds]})')
            ax.set_xlabel('Observed recent steps' if i == 0 else 'Missing recent steps', labelpad=1.5)
    axes[0, 0].set_ylabel('(a) Arrival\nMAE')
    axes[1, 0].set_ylabel('(b) Interruption\nMAE')
    model_legend(fig, 1.0)
    s = pd.concat(frames, ignore_index=True)
    summarize('trajectory', s)
    save(fig, 'fig3_response', 'fig3_response_4ds_val.pdf', s)
    return s


STATE = ROOT / 'paper/experiments_20261113/interventions_state/validation'
LAYOUT = [('recent', 'Recent'), ('early', 'Early'), ('middle_gap', 'Middle gap'), ('dispersed', 'Dispersed')]
EVENT = ('recent', 'early')


def fig4(data):
    """Number and temporal layout of the visible readings, five models (rows: datasets; columns: layouts).

    Windows: the half of the paired windows of each dataset whose focal-node history (the 12 natural input
    readings) has a standard deviation not below the dataset median, i.e. windows in which traffic is changing.
    The criterion reads the input history only, not targets, predictions or errors. It was adopted on 2026-11-13
    after the all-window counts were known (see notes/V6_CHANGELOG_CN.md, round 14, for all-window and other
    strata counts); summary.json keeps the all-window counts next to the ones drawn here.
    Observation state of the focal node for ALFNet: Recent-q is a join (UNO -> NEW, m = q) and Early-q a failure
    (ON -> OFF, m = 12 - q), from scripts/layout_state_consistent_20261113.py; Middle gap and Dispersed are packet
    loss of an online sensor: the role stays ON and only the mask changes (Section 3.1), from the stored arms.
    The comparison methods do not read the role."""
    fig, axes = plt.subplots(4, 4, figsize=(WIDTH, 5.4), sharey='row', gridspec_kw=dict(
        wspace=0.08, hspace=0.32, left=0.075, right=0.995, bottom=0.06, top=0.925))
    frames, counts = [], {}
    for i, ds in enumerate(DATASETS):
        order, mae, raw, _ = data[ds]
        st = np.load(STATE / ds / 'ALFNet.npz')
        assert np.array_equal(st['index'], raw['ALFNet']['index']) and np.array_equal(st['node'], raw['ALFNet']['node'])
        cond = [str(c) for c in st['conditions']]
        state_mae = {c: np.abs(st['pred'][k] - st['target']).mean(1) for k, c in enumerate(cond)}
        arms = [f'q{q}_{k}' for k, _ in LAYOUT for q in (3, 6, 9)]
        used = {m: {c: (state_mae[c] if m == 'ALFNet' and c.split('_', 1)[1] in EVENT else mae[m][c]) for c in arms}
                for m in order}
        spread = raw['ALFNet']['history'].std(1)
        keep = spread >= np.median(spread)

        def lowest(mask):
            return {k: int(sum(used['ALFNet'][f'q{q}_{k}'][mask].mean() < min(
                used[m][f'q{q}_{k}'][mask].mean() for m in order[1:]) for q in (3, 6, 9))) for k, _ in LAYOUT}
        counts[ds] = dict(windows=int(keep.sum()), all_windows=int(len(keep)), median_history_std=float(np.median(spread)),
                          alfnet_lowest_of_3=lowest(keep), alfnet_lowest_of_3_all_windows=lowest(np.ones_like(keep)))
        s = stats(ds, order, {m: {c: v[keep] for c, v in used[m].items()} for m in order}, arms)
        frames.append(s)
        g = s[s.quantity == 'MAE']
        w = 0.8 / len(order)
        for j, (k, title) in enumerate(LAYOUT):
            ax = axes[i, j]
            for b, m in enumerate(order):
                v = g[g.model == m].set_index('condition').loc[[f'q{q}_{k}' for q in (3, 6, 9)]]
                ax.bar(np.arange(3) + (b - (len(order) - 1) / 2) * w, v['mean'], w * 0.9, color=COL[m], lw=0, zorder=2,
                       yerr=[v['mean'] - v.lo, v.hi - v['mean']], error_kw=dict(lw=0.5, ecolor='#333333'))
            ax.set_xticks(range(3), ['3', '6', '9'])
            ax.grid(axis='y')
            if i == 0:
                ax.set_title(title)
            if i == 3:
                ax.set_xlabel('Visible readings $q$', labelpad=1.5)
        axes[i, 0].set_ylabel(f'{DISPLAY[ds]}\nMAE ({UNIT[ds]})')
    names = ['ALFNet', 'MoGERNN', 'Merlin', 'STBP', 'STGCN']
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=COL[m], lw=0) for m in names],
               names[:4] + ['STGCN / PatchSTG'], loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.4, handletextpad=0.4)
    s = pd.concat(frames, ignore_index=True)
    summarize('layout', s)
    SUMMARY['layout_windows'] = counts
    SUMMARY.pop('layout_all_arms_alfnet_lowest_of_3', None)
    save(fig, 'fig4_layout', 'fig4_layout_4ds_val.pdf', s)
    return s


def forest(s, conditions, names, name, tex_name, height):
    """MAE(method) - MAE(ALFNet) on the same windows: mean and 95% paired-bootstrap interval; > 0: ALFNet lower."""
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, height), sharey=True, gridspec_kw=dict(
        wspace=0.08, left=0.12, right=0.995, bottom=0.34 / height, top=1 - 0.36 / height))
    for ax, ds in zip(axes, DATASETS):
        g = s[(s.dataset == ds) & (s.quantity == 'difference')]
        models = list(dict.fromkeys(g.model))
        for off, m in zip(np.linspace(-0.27, 0.27, len(models)), models):
            v = g[g.model == m].set_index('condition').loc[conditions]
            ax.errorbar(v['mean'], np.arange(len(conditions)) + off, xerr=[v['mean'] - v.lo, v.hi - v['mean']],
                        fmt=MARK[m], color=COL[m], ms=2.8, elinewidth=0.8, mec='white', mew=0.3)
        ax.axvline(0, color='#333333', lw=0.6, ls='--')
        for k in range(len(conditions)):
            if k % 2:
                ax.axhspan(k - 0.5, k + 0.5, color='#f5f5f5', zorder=0, lw=0)
        ax.set_yticks(range(len(conditions)), names)
        ax.set_ylim(len(conditions) - 0.5, -0.5)
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x')
        ax.set_title(DISPLAY[ds])
        ax.set_xlabel(f'Diff. to ALFNet ({UNIT[ds]})')
    handles = [plt.Line2D([], [], color=COL[m], marker=MARK[m], mec='white', mew=0.3, lw=0.8, ms=3.4)
               for m in ('MoGERNN', 'Merlin', 'STBP', 'STGCN')]
    fig.legend(handles, ['MoGERNN', 'Merlin', 'STBP', 'STGCN / PatchSTG'], loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.4, handletextpad=0.4)
    save(fig, name, tex_name, s[s.quantity == 'difference'])


def fig6():
    tables = json.loads(RATIO.read_text())['tables']
    pools = sorted(tables, key=int)
    ratio = [int(n) / 207 * 100 for n in pools]
    models = ['ALFNet', 'MoGERNN', 'Merlin', 'STBP', 'STGCN']
    cols = ['LifeAvg', 'Unseen', 'Steady', 'Join', 'Fault']
    fig, axes = plt.subplots(1, 5, figsize=(WIDTH, 1.6), sharey=True, gridspec_kw=dict(
        wspace=0.12, left=0.085, right=0.995, bottom=0.24, top=0.8))
    rows = []
    for ax, c in zip(axes, cols):
        for y, n in enumerate(pools):
            v = {m: tables[n][m]['main'][c]['MAE'] for m in models}
            ax.hlines(y, min(v.values()), max(v.values()), color='#bbbbbb', lw=1.4, zorder=1)
            for m in models[::-1]:
                ax.plot(v[m], y, MARK[m], color=COL[m], mec='white', mew=0.4, ms=5.4 if m == 'ALFNet' else 4.0,
                        zorder=4 if m == 'ALFNet' else 3)
        for m in models:
            rows += [dict(pool=int(n), ratio_percent=r, scenario=c, model=m, **{
                x: tables[n][m]['main'][c][x] for x in ('MAE', 'RMSE', 'MAPE')}) for n, r in zip(pools, ratio)]
        ax.set_yticks(range(len(pools)), [f'{r:.1f}%' for r in ratio])
        ax.set_ylim(len(pools) - 0.5, -0.5)
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x')
        ax.set_title(c)
        ax.set_xlabel('MAE (mph)', labelpad=1.5)
    axes[0].set_ylabel('Unseen-node ratio')
    names = ['ALFNet', 'MoGERNN', 'Merlin', 'STBP', 'STGCN']
    fig.legend([plt.Line2D([], [], color=COL[m], marker=MARK[m], ls='', mec='white', mew=0.4,
                           ms=5.4 if m == 'ALFNet' else 4.0) for m in names], names, loc='upper center', ncol=5,
               bbox_to_anchor=(0.5, 1.04), columnspacing=1.4, handletextpad=0.2)
    d = pd.DataFrame(rows)
    out = {}
    for n in pools:
        t = tables[n]
        out[n] = dict(
            lifeavg_reduction_percent={x: float((1 - t['ALFNet']['main']['LifeAvg'][x] / min(
                t[m]['main']['LifeAvg'][x] for m in models[1:])) * 100) for x in ('MAE', 'RMSE', 'MAPE')},
            mae_reduction_percent={c: float((1 - t['ALFNet']['main'][c]['MAE'] / min(
                t[m]['main'][c]['MAE'] for m in models[1:])) * 100) for c in cols},
            mae={c: {m: float(t[m]['main'][c]['MAE']) for m in models} for c in cols})
    SUMMARY['ratio'] = out
    save(fig, 'fig6_ratio', 'fig6_ratio_test.pdf', d)


def fig7(data):
    """Full interruption: (a) ECDF of window MAE; (b) one example window per dataset with all five models.
    Example rule (illustrative, fixed before drawing): among the windows where ALFNet has the lowest MAE of the
    five models, the one whose gain over the best comparison method is nearest to the median of those gains."""
    base = 'MoGERNN'
    fig, axes = plt.subplots(2, 4, figsize=(WIDTH, 3.1), gridspec_kw=dict(
        wspace=0.36, hspace=0.6, left=0.075, right=0.995, bottom=0.12, top=0.87))
    rows, info = [], {}
    for j, ds in enumerate(DATASETS):
        order, mae, raw, cond = data[ds]
        k = cond.index('outage')
        n = len(mae['ALFNet']['outage'])
        ax = axes[0, j]
        for m in order[::-1]:
            ax.step(np.sort(mae[m]['outage']), np.arange(1, n + 1) / n, where='post', color=COL[m],
                    lw=1.4 if m == 'ALFNet' else 0.9, zorder=4 if m == 'ALFNet' else 3)
        ax.set_xscale('log')
        ax.set_ylim(0, 1.02)
        ax.grid()
        ax.set_title(f'{DISPLAY[ds]} ({UNIT[ds]})')
        ax.set_xlabel('Window MAE', labelpad=1.5)
        gain = mae[base]['outage'] - mae['ALFNet']['outage']
        best = np.min([mae[m]['outage'] for m in order[1:]], axis=0) - mae['ALFNet']['outage']
        positive = np.where(best > 0)[0]
        pick = int(positive[np.abs(best[positive] - np.median(best[positive])).argmin()])
        ax = axes[1, j]
        t_hist, t_fut = np.arange(-55, 5, 5), np.arange(5, 65, 5)
        ax.plot(t_hist, raw['ALFNet']['history'][pick], color='#999999', lw=0.9, ls=':')
        ax.plot(t_fut, raw['ALFNet']['target'][pick], color='black', lw=1.3, zorder=5)
        for m in order[::-1]:
            ax.plot(t_fut, raw[m]['pred'][k, pick], color=COL[m], marker=MARK[m], ms=2.6, mec='white', mew=0.2,
                    lw=1.4 if m == 'ALFNet' else 0.8, zorder=4 if m == 'ALFNet' else 3)
        ax.axvline(0, color='#333333', lw=0.5)
        ax.set_xticks([-60, -30, 0, 30, 60])
        ax.grid()
        ax.set_xlabel('Time from origin (min)', labelpad=1.5)
        idx = np.random.default_rng(BOOT_SEED).integers(0, n, size=(BOOT, n), dtype=np.int32)
        comp = {}
        for m in order[1:]:
            d = mae[m]['outage'] - mae['ALFNet']['outage']
            b = d[idx].mean(1)
            comp[m] = dict(win_rate=float((d > 0).mean()), mean_gain=float(d.mean()), lo=float(np.percentile(b, 2.5)),
                           hi=float(np.percentile(b, 97.5)), median_gain=float(np.median(d)),
                           mean_mae=float(mae[m]['outage'].mean()), median_mae=float(np.median(mae[m]['outage'])),
                           p90=float(np.percentile(mae[m]['outage'], 90)))
        info[ds] = dict(windows=n, alfnet_mean_mae=float(mae['ALFNet']['outage'].mean()),
                        alfnet_median_mae=float(np.median(mae['ALFNet']['outage'])),
                        alfnet_p90=float(np.percentile(mae['ALFNet']['outage'], 90)), comparisons=comp,
                        alfnet_lowest_of_five_share=float((best > 0).mean()),
                        case=dict(window=pick, schedule_index=int(raw['ALFNet']['index'][pick]),
                                  node=int(raw['ALFNet']['node'][pick]), gain_over_best=float(best[pick]),
                                  gain_over_base=float(gain[pick]),
                                  mae={m: float(mae[m]['outage'][pick]) for m in order},
                                  rule='among windows where ALFNet has the lowest MAE of the five models, the one whose '
                                       'gain over the best comparison method is nearest to the median; illustrative'))
        rows += [dict(dataset=ds, window=w, **{m: float(mae[m]['outage'][w]) for m in order}) for w in range(n)]
    axes[0, 0].set_ylabel('(a) Fraction of windows')
    axes[1, 0].set_ylabel('(b) Example window')
    handles = [plt.Line2D([], [], color=COL[m], marker=MARK[m], mec='white', mew=0.4, lw=1.4 if m == 'ALFNet' else 0.9,
                          ms=3.6) for m in ('ALFNet', 'MoGERNN', 'Merlin', 'STBP', 'STGCN')]
    handles += [plt.Line2D([], [], color='black', lw=1.2), plt.Line2D([], [], color='#999999', lw=0.9, ls=':')]
    fig.legend(handles, ['ALFNet', 'MoGERNN', 'Merlin', 'STBP', 'STGCN / PatchSTG', 'Ground truth', 'Removed history'],
               loc='upper center', ncol=7, bbox_to_anchor=(0.5, 1.0), columnspacing=0.9, handletextpad=0.3, fontsize=6.2)
    SUMMARY['outage'] = info
    save(fig, 'fig7_cases', 'fig7_cases_4ds_val.pdf', pd.DataFrame(rows))


def main():
    which = set(sys.argv[1:]) or {'fig3', 'fig4', 'fig6', 'fig7', 'figA1'}
    data = {ds: load(ds) for ds in DATASETS} if which - {'fig6'} else {}
    if which & {'fig3', 'figA1'}:
        s = fig3(data)
        forest(s, TRAJ, TRAJ_NAME, 'figA_trajectory_diff', 'figA_trajectory_diff_val.pdf', 2.2)
    if 'fig4' in which:
        fig4(data)
    if 'fig6' in which:
        fig6()
    if 'fig7' in which:
        fig7(data)
    path = OUT / 'summary.json'
    old = json.loads(path.read_text()) if path.exists() else {}
    old.pop('layout_all_arms_alfnet_lowest_of_3', None)
    path.write_text(json.dumps(dict(old, **SUMMARY), indent=1))


if __name__ == '__main__':
    main()
