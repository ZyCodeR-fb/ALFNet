"""2026-11-12: experiment figures of the paper in one style (no training, no inference; reads saved results only).

  fig2  phase_gain      relative reduction of ALFNet vs the best comparison method per phase cell
                        (test, mean over seeds 32/42/52; tables/phase_gain_3seed.csv from make_tables.py)
  fig3  trajectory      METR-LA validation, 64 paired windows: arrival / interruption arms + paired differences
  fig4  layout          METR-LA validation, same windows: number and temporal layout of the visible readings
  fig5  hparams         one-factor sensitivity (validation; paper/final_figures_20261017/validation/fig5_values.csv)
  fig6  cases           METR-LA validation, full interruption: window-error ECDF, paired gains, one example
  figA1 router          OS router weights per observation role (paper/experiments_20261112/router/router_gates.csv)

Representative comparison methods of fig3/4/6: the best method of each of the four groups by METR-LA full
validation LifeAvg MAE (MoGERNN, Merlin, STBP, STGCN). Intervals are 95% percentile intervals of a paired
bootstrap over the 64 windows (10,000 resamples, seed 20261112). Every figure is written as pdf/svg/png to
paper/experiments_20261112/figures/ with its plotted values as csv; the pdf is copied to the LaTeX figures folder.
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
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/experiments_20261112/figures'
TEX = ROOT / 'paper/latex_neurips_20261020/figures'
OLD = ROOT / 'paper/final_figures_20261017/validation'
NEW = ROOT / 'paper/experiments_20261112/interventions/validation/METRLA'
DATASETS = ['METRLA', 'PEMSBAY', 'PEMS04', 'PEMS08']
DISPLAY = {'METRLA': 'METR-LA', 'PEMSBAY': 'PEMS-BAY', 'PEMS04': 'PEMS04', 'PEMS08': 'PEMS08'}
MODELS = ['ALFNet', 'MoGERNN', 'Merlin', 'STBP', 'STGCN']
COL = {'ALFNet': '#0072B2', 'MoGERNN': '#009E73', 'Merlin': '#E69F00', 'STBP': '#CC79A7', 'STGCN': '#7F7F7F'}
MARK = {'ALFNet': 'o', 'MoGERNN': 's', 'Merlin': '^', 'STBP': 'D', 'STGCN': 'v'}
METRIC_COL = {'MAE': '#0072B2', 'RMSE': '#E69F00', 'MAPE': '#009E73'}
WIDTH = 5.5  # NeurIPS text width in inches
BOOT, BOOT_SEED = 10000, 20261112

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'lines.linewidth': 1.1, 'lines.markersize': 3.6, 'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'savefig.dpi': 300, 'figure.dpi': 150,
    'grid.linewidth': 0.4, 'grid.alpha': 0.35, 'legend.frameon': False,
})


def save(fig, name, tex_name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    data.to_csv(OUT / f'{name}_values.csv', index=False)
    shutil.copyfile(OUT / f'{name}.pdf', TEX / tex_name)
    print('saved', name, '->', tex_name)


def interventions():
    frames = [pd.read_csv((OLD / 'METRLA' if m in ('ALFNet', 'MoGERNN', 'STBP') else NEW) / f'{m}.csv') for m in MODELS]
    d = pd.concat(frames, ignore_index=True)
    assert len(d) == 1280 * len(MODELS) and set(d.model) == set(MODELS)
    wide = d.pivot(index=['condition', 'pair_id'], columns='model', values='MAE')
    assert not wide.isna().any().any()
    return d, wide


def boot_index(n):
    return np.random.default_rng(BOOT_SEED).integers(0, n, size=(BOOT, n))


def mean_ci(values, idx):
    """values [n] -> mean, 2.5%, 97.5% of the bootstrap means (windows resampled with replacement)."""
    means = values[idx].mean(1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def arm_stats(wide, conditions):
    idx = boot_index(64)
    rows = []
    for c in conditions:
        w = wide.loc[c]
        assert len(w) == 64
        for m in MODELS:
            mean, lo, hi = mean_ci(w[m].to_numpy(), idx)
            rows.append(dict(condition=c, model=m, quantity='MAE', mean=mean, lo=lo, hi=hi, win_rate=np.nan))
        for m in MODELS[1:]:
            g = (w[m] - w['ALFNet']).to_numpy()
            mean, lo, hi = mean_ci(g, idx)
            rows.append(dict(condition=c, model=m, quantity='MAE difference (method - ALFNet)', mean=mean, lo=lo,
                             hi=hi, win_rate=float((g > 0).mean())))
    return pd.DataFrame(rows)


def lines(ax, stats, conditions, labels):
    """Mean MAE of every model over the paired windows (intervals are shown on the paired differences)."""
    for m in MODELS:
        s = stats[(stats.model == m) & (stats.quantity == 'MAE')].set_index('condition').loc[conditions]
        ax.plot(range(len(conditions)), s['mean'], color=COL[m], marker=MARK[m], label=m, mec='white', mew=0.4,
                lw=1.5 if m == 'ALFNet' else 1.0, ms=4.4 if m == 'ALFNet' else 3.8, zorder=4 if m == 'ALFNet' else 3)
    ax.set_xticks(range(len(conditions)), labels)
    ax.set_xlim(-0.35, len(conditions) - 0.65)
    ax.grid(axis='y')


def differences(ax, stats, conditions, labels, horizontal=False):
    """MAE(method) - MAE(ALFNet) on the same windows, mean and 95% paired-bootstrap interval; > 0: ALFNet lower."""
    offsets = np.linspace(-0.24, 0.24, 4)
    for off, m in zip(offsets, MODELS[1:]):
        s = stats[(stats.model == m) & stats.quantity.str.startswith('MAE diff')].set_index('condition').loc[conditions]
        pos = np.arange(len(conditions)) + off
        err = [s['mean'] - s.lo, s.hi - s['mean']]
        kw = dict(fmt=MARK[m], color=COL[m], ms=3.2, elinewidth=0.8, mec='white', mew=0.3, label=m)
        if horizontal:
            ax.errorbar(s['mean'], pos, xerr=err, **kw)
        else:
            ax.errorbar(pos, s['mean'], yerr=err, **kw)
    if horizontal:
        ax.axvline(0, color='#333333', lw=0.6, ls='--')
        ax.set_yticks(range(len(conditions)), labels)
        ax.set_ylim(len(conditions) - 0.5, -0.5)
        ax.grid(axis='x')
    else:
        ax.axhline(0, color='#333333', lw=0.6, ls='--')
        ax.set_xticks(range(len(conditions)), labels)
        ax.set_xlim(-0.5, len(conditions) - 0.5)
        ax.grid(axis='y')


def fig2():
    d = pd.read_csv(ROOT / 'paper/latex_neurips_20261020/tables/phase_gain_3seed.csv')
    phases = ['Cold', 'Warm', 'Ready', 'Onset', 'Decay', 'Outage']
    metrics = ['MAE', 'RMSE', 'MAPE']
    ypos = [0, 1, 2, 3.35, 4.35, 5.35]
    lim = float(np.ceil(d.gain_percent.abs().max() / 5) * 5)
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 2.25), sharey=True)
    fig.subplots_adjust(left=0.13, right=0.9, wspace=0.08, top=0.9, bottom=0.1)
    for ax, ds in zip(axes, DATASETS):
        g = d[d.dataset == ds].pivot(index='phase', columns='metric', values='gain_percent').loc[phases, metrics]
        for i, p in enumerate(phases):
            for j, x in enumerate(metrics):
                v = g.loc[p, x]
                ax.add_patch(plt.Rectangle((j - 0.5, ypos[i] - 0.5), 1, 1, fc=plt.cm.RdBu(norm(v)), ec='white', lw=0.8))
                ax.text(j, ypos[i], f'{v:.1f}', ha='center', va='center', fontsize=6.3,
                        color='white' if abs(v) > 0.6 * lim else '#222222')
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(5.9, -0.55)
        ax.set_xticks(range(3), metrics)
        ax.set_yticks(ypos, phases)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(DISPLAY[ds], pad=4)
    for label, lo, hi in (('Join', -0.4, 2.4), ('Fault', 2.95, 5.75)):
        axes[0].annotate('', xy=(-0.62, lo), xytext=(-0.62, hi), xycoords=('axes fraction', 'data'),
                         textcoords=('axes fraction', 'data'), arrowprops=dict(arrowstyle='-', lw=0.8, color='#444444'),
                         annotation_clip=False)
        axes[0].text(-0.7, (lo + hi) / 2, label, transform=axes[0].get_yaxis_transform(), rotation=90, ha='right',
                     va='center', fontsize=7.5, fontweight='bold')
    cax = fig.add_axes([0.915, 0.14, 0.012, 0.72])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='RdBu'), cax=cax)
    cb.set_label('Error reduction vs. best baseline (%)', fontsize=6.5)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=2)
    save(fig, 'fig2_phase_gain', 'fig2_phase_gain_3seed.pdf', d)


def legend(fig, ax, y):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, y), columnspacing=1.4, handletextpad=0.4)


def fig3(wide):
    arrival, interruption = ['join_3', 'join_6', 'full'], ['full', 'fault_3', 'fault_6', 'outage']
    unique = ['join_3', 'join_6', 'full', 'fault_3', 'fault_6', 'outage']
    name = ['Join 3', 'Join 6', 'Full', 'Fault 3', 'Fault 6', 'Outage']
    stats = arm_stats(wide, unique)
    fig = plt.figure(figsize=(WIDTH, 1.95))
    gs = fig.add_gridspec(1, 3, width_ratios=[3, 4, 4.2], left=0.075, right=0.995, bottom=0.2, top=0.86, wspace=0.42)
    axes = [fig.add_subplot(g) for g in gs]
    lines(axes[0], stats, arrival, ['3', '6', '12'])
    axes[0].set(xlabel='Observed recent steps', ylabel='MAE (mph)')
    axes[0].set_title('(a) Arrival', loc='left')
    lines(axes[1], stats, interruption, ['0', '3', '6', '12'])
    axes[1].set(xlabel='Missing recent steps')
    axes[1].set_title('(b) Interruption', loc='left')
    lo = min(a.get_ylim()[0] for a in axes[:2])
    hi = max(a.get_ylim()[1] for a in axes[:2])
    for a in axes[:2]:
        a.set_ylim(lo, hi)
    differences(axes[2], stats, unique, name, horizontal=True)
    axes[2].set_xlabel('MAE diff. to ALFNet (mph)')
    axes[2].set_title('(c) Paired difference', loc='left')
    legend(fig, axes[0], 1.06)
    save(fig, 'fig3_trajectory', 'fig3_trajectory_metrla_val.pdf', stats)
    return stats


def fig4(wide):
    layouts = [('recent', 'Recent'), ('early', 'Early'), ('dispersed', 'Dispersed'), ('middle_gap', 'Middle gap')]
    conditions = ['q0_recent', 'q12_recent'] + [f'q{q}_{k}' for q in (3, 6, 9) for k, _ in layouts]
    stats = arm_stats(wide, conditions)
    fig, axes = plt.subplots(2, 4, figsize=(WIDTH, 3.0), sharey='row', sharex=True,
                             gridspec_kw=dict(wspace=0.1, hspace=0.16, left=0.085, right=0.995, bottom=0.12, top=0.88))
    for j, (key, label) in enumerate(layouts):
        arms = ['q0_recent', f'q3_{key}', f'q6_{key}', f'q9_{key}', 'q12_recent']
        lines(axes[0, j], stats, arms, ['0', '3', '6', '9', '12'])
        differences(axes[1, j], stats, arms, ['0', '3', '6', '9', '12'])
        for ax in axes[:, j]:
            ax.set_xlim(-0.5, 4.5)
            ax.axvspan(-0.5, 0.5, color='#f0f0f0', zorder=0, lw=0)  # q = 0 and q = 12 have one layout only
            ax.axvspan(3.5, 4.5, color='#f0f0f0', zorder=0, lw=0)
        axes[0, j].set_title(label)
        axes[1, j].set_xlabel('Visible steps $q$')
    axes[0, 0].set_ylabel('MAE (mph)')
    axes[1, 0].set_ylabel('MAE difference\nto ALFNet (mph)')
    legend(fig, axes[0, 0], 1.0)
    save(fig, 'fig4_layout', 'fig4_layout_metrla_val.pdf', stats)
    return stats


def fig5():
    d = pd.read_csv(OLD / 'fig5_values.csv')
    axes_order = list(dict.fromkeys(d.axis))
    label = {'capacity': 'Partition capacity', 'lr_multiplier': 'Learning-rate multiplier',
             'dropout': 'ALFTemporal dropout', 'weight_decay': 'Weight decay'}
    fig, axs = plt.subplots(4, 4, figsize=(WIDTH, 4.7), sharey='row',  # x values differ per dataset
                            gridspec_kw=dict(wspace=0.1, hspace=0.34, left=0.1, right=0.995, bottom=0.085, top=0.9))
    for i, ds in enumerate(DATASETS):
        for j, axis in enumerate(axes_order):
            ax = axs[i, j]
            g = d[(d.dataset == ds) & (d.axis == axis)]
            values = sorted(g.parameter.unique())
            for k, (metric, off) in enumerate(zip(('MAE', 'RMSE', 'MAPE'), (-0.22, 0, 0.22))):
                s = g[g.metric == metric].set_index('parameter').loc[values]
                x = np.arange(len(values)) + off
                ax.vlines(x, 0, s.change, color=METRIC_COL[metric], lw=0.9)
                ax.plot(x, s.change, 'o', color=METRIC_COL[metric], ms=2.8, label=metric)
            ax.axhline(0, color='#333333', lw=0.6)
            anchor = [k for k, v in enumerate(values) if (g[g.parameter == v].change.abs() < 1e-12).all()]
            for k in anchor:
                ax.axvspan(k - 0.45, k + 0.45, color='#eeeeee', zorder=0, lw=0)
            fmt = (lambda v: f'{int(v)}') if axis == 'capacity' else \
                (lambda v: f'$10^{{{int(round(np.log10(v)))}}}$') if axis == 'weight_decay' else (lambda v: f'{v:g}')
            ax.set_xticks(range(len(values)), [fmt(v) for v in values])
            ax.set_xlim(-0.6, len(values) - 0.4)
            ax.grid(axis='y')
            if i == 0:
                ax.set_title(label.get(axis, axis))
            if j == 0:
                ax.set_ylabel(f'{DISPLAY[ds]}\nchange (%)')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0), columnspacing=1.6, handletextpad=0.2)
    save(fig, 'fig5_hparams', 'fig5_hparams_val.pdf', d)


def fig6(d, wide):
    w = wide.loc['outage']
    base = w[MODELS[1:]].mean().idxmin()  # representative method with the lowest mean MAE on this arm
    gain = (w[base] - w['ALFNet'])
    positive = gain[gain > 0]
    pick = int((positive - positive.median()).abs().idxmin())
    idx = boot_index(64)
    rows = []
    for m in MODELS[1:]:
        g = (w[m] - w['ALFNet']).to_numpy()
        mean, lo, hi = mean_ci(g, idx)
        rows.append(dict(baseline=m, win_rate=float((g > 0).mean()), mean_gain=mean, lo=lo, hi=hi,
                         median_gain=float(np.median(g)), p90_alfnet=float(w['ALFNet'].quantile(0.9)),
                         p90_baseline=float(w[m].quantile(0.9))))
    info = dict(arm='outage', lowest_mean_baseline=base, case_pair_id=pick, case_gain=float(gain.loc[pick]),
                case_rule='nearest to the median positive MAE gain over the lowest-mean representative method; '
                          'illustrative, not random', comparisons=rows)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 1.95), gridspec_kw=dict(
        width_ratios=[3, 3, 3.4], wspace=0.5, left=0.075, right=0.995, bottom=0.2, top=0.86))
    ax = axes[0]
    for m in MODELS:
        v = np.sort(w[m].to_numpy())
        ax.step(v, np.arange(1, 65) / 64, where='post', color=COL[m], lw=1.3 if m == 'ALFNet' else 0.9, label=m)
    ax.set(xlabel='Window MAE (mph)', ylabel='Fraction of windows', ylim=(0, 1.02))
    ax.set_xscale('log')
    ax.grid()
    ax.set_title('(a) Error distribution', loc='left')
    ax = axes[1]
    g = np.sort(gain.to_numpy())[::-1]
    ax.bar(np.arange(64), g, width=0.85, color=np.where(g > 0, COL['ALFNet'], COL[base]), lw=0)
    ax.axhline(0, color='#333333', lw=0.6)
    ax.set(xlabel='Windows, sorted by gain', ylabel=f'MAE gain over {base} (mph)')
    ax.set_title('(b) Paired gain', loc='left')
    ax.text(0.97, 0.95, f'ALFNet lower in {int((gain > 0).sum())}/64', transform=ax.transAxes, ha='right', va='top', fontsize=6.3)
    ax.grid(axis='y')
    ax = axes[2]
    case = d[(d.condition == 'outage') & (d.pair_id == pick)].set_index('model')
    t = np.arange(5, 65, 5)
    truth = np.array(json.loads(case.loc['ALFNet', 'target']))
    for m in MODELS[::-1]:
        ax.plot(t, json.loads(case.loc[m, 'prediction']), color=COL[m], marker=MARK[m], ms=2.6,
                lw=1.3 if m == 'ALFNet' else 0.8, mec='white', mew=0.2)
    ax.plot(t, truth, color='black', lw=1.2, label='Ground truth')
    ax.set(xlabel='Forecast horizon (min)', ylabel='Speed (mph)')
    ax.set_xticks([15, 30, 45, 60])
    ax.grid()
    ax.set_title('(c) Example window', loc='left')
    handles, labels = axes[0].get_legend_handles_labels()
    extra = ax.get_legend_handles_labels()
    fig.legend(handles + extra[0], labels + extra[1], loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.06),
               columnspacing=1.2, handletextpad=0.3)
    save(fig, 'fig6_cases', 'fig6_cases_metrla_val.pdf',
         pd.DataFrame(dict(pair_id=w.index, **{m: w[m].to_numpy() for m in MODELS})))
    (OUT / 'fig6_case_selection.json').write_text(json.dumps(info, indent=1))
    return info


def figA1():
    d = pd.read_csv(ROOT / 'paper/experiments_20261112/router/router_gates.csv')
    roles = ['ON', 'NEW', 'OFF', 'UNO']
    experts = ['identity', 'mean', 'weighted_mean', 'forward_diffusion', 'backward_diffusion']
    short = ['Identity', 'Mean', 'Weighted', 'Forward', 'Backward']
    per_seed = d.groupby(['dataset', 'seed', 'role', 'expert']).gate.mean().reset_index()
    g = per_seed.groupby(['dataset', 'role', 'expert']).gate.agg(['mean', 'std']).reset_index()
    span = float((g['mean'] - 0.2).abs().max())
    norm = TwoSlopeNorm(vmin=0.2 - span, vcenter=0.2, vmax=0.2 + span)
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 1.65), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.9, wspace=0.07, top=0.88, bottom=0.27)
    for ax, ds in zip(axes, DATASETS):
        m = g[g.dataset == ds].pivot(index='role', columns='expert', values='mean').loc[roles, experts]
        ax.imshow(m.to_numpy(), cmap='PuOr_r', norm=norm, aspect='auto')
        for i in range(4):
            for j in range(5):
                v = m.iloc[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6,
                        color='white' if abs(v - 0.2) > 0.6 * span else '#222222')
        ax.set_xticks(range(5), short, rotation=40, ha='right', rotation_mode='anchor')
        ax.set_yticks(range(4), roles)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(DISPLAY[ds], pad=4)
    cax = fig.add_axes([0.915, 0.27, 0.012, 0.61])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='PuOr_r'), cax=cax)
    cb.set_label('Mean expert weight', fontsize=6.5)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=2)
    save(fig, 'figA1_router_gates', 'figA1_router_gates.pdf', g)


def main():
    which = set(sys.argv[1:]) or {'fig2', 'fig3', 'fig4', 'fig5', 'fig6', 'figA1'}
    d, wide = interventions()
    if 'fig2' in which:
        fig2()
    if 'fig3' in which:
        fig3(wide)
    if 'fig4' in which:
        fig4(wide)
    if 'fig5' in which:
        fig5()
    if 'fig6' in which:
        print(json.dumps(fig6(d, wide), indent=1))
    if 'figA1' in which:
        figA1()


if __name__ == '__main__':
    main()
