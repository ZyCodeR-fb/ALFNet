"""Known-node manuscript views of existing results; no inference or test access."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/rq_experiments_20260922'
MODELS = ['ALFNet', 'UIGNN', 'MoGERNN', 'STBP']
COLORS = ['#0072B2', '#009E73', '#D55E00', '#CC79A7']


def main():
    OUT.mkdir(exist_ok=True)
    source = ROOT / 'paper/support_metric_audit_20260922'
    s = pd.read_csv(source / 'all_metrics.csv')
    rows = []
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})

    def values(model, conditions, figure):
        result = []
        for c in conditions:
            r = s[(s.track == 'fault') & (s.model == model) & (s.condition == c)]
            assert len(r) == 1
            v = float(r.MAE.iloc[0]); result.append(v)
            rows.append(dict(figure=figure, model=model, condition=c, track='fault', MAE=v))
        return result

    def save(fig, name):
        for ext in ['png', 'pdf']:
            fig.savefig(OUT / f'{name}.{ext}', dpi=240, bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True, layout='constrained')
    for ax, cs, xs, title, label in [
        (axes[0], ['join_3', 'join_6', 'full'], [3, 6, 12],
         'Known nodes: controlled arrival', 'Recent observations (12 = full reference)'),
        (axes[1], ['full', 'fault_3', 'fault_6', 'outage'], [0, 3, 6, 12],
         'Known nodes: interruption', 'Missing recent steps')]:
        for m, color, marker in zip(MODELS, COLORS, ['o', 's', '^', 'D']):
            ax.plot(xs, values(m, cs, 3), color=color, marker=marker, label=m,
                    lw=2.4 if m == 'ALFNet' else 1.5)
        ax.set(title=title, xlabel=label, xticks=xs); ax.grid(alpha=.18)
    axes[0].set_ylabel('MAE (mph)'); axes[0].legend(ncol=2, fontsize=8)
    save(fig, 'trajectory_known')

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True, layout='constrained')
    qs = [0, 3, 6, 9, 12]
    for ax, layout in zip(axes, ['recent', 'early', 'dispersed', 'middle_gap']):
        cs = [f'q{q}_{"recent" if q in [0, 12] else layout}' for q in qs]
        for m, color, marker in zip(MODELS, COLORS, ['o', 's', '^', 'D']):
            ax.plot(qs, values(m, cs, 4), color=color, marker=marker, ms=4,
                    label=m, lw=2.4 if m == 'ALFNet' else 1.5)
        ax.set(title=layout.replace('_', ' ').title(), xlabel='Available observations q', xticks=qs)
        ax.grid(alpha=.18)
    axes[0].set_ylabel('Known nodes: MAE (mph)'); axes[0].legend(fontsize=8)
    save(fig, 'layout_known')

    z = np.load(source / 'residual_matrices.npz')
    # Preserve the previous full-cohort color range, not a model-specific range.
    maximum = max(float(np.abs(z[k]).max()) for k in z.files)
    norm = SymLogNorm(linthresh=2, vmin=-maximum, vmax=maximum, base=10)
    fig, axes = plt.subplots(4, 3, figsize=(10, 9), sharex=True, sharey=True, layout='constrained')
    stats = []
    for i, m in enumerate(MODELS):
        for j, (c, title) in enumerate([('full', 'Full history'), ('fault_6', 'Six-step interruption'), ('outage', 'Complete interruption')]):
            a = z[m + '__' + c]; assert a.shape == (64, 12)
            ax = axes[i, j]
            im = ax.imshow(a, aspect='auto', cmap='RdBu_r', norm=norm, interpolation='nearest')
            if i == 0: ax.set_title(title)
            if j == 0: ax.set_ylabel(m + '\nWindow order')
            if i == 3: ax.set_xlabel('Horizon (minutes)')
            ax.set_xticks([0, 5, 11], [5, 30, 60]); ax.set_yticks([0, 31, 63], [1, 32, 64])
            stats.append(dict(model=m, condition=c, MAE=float(abs(a).mean()),
                              absolute_error_p90=float(np.percentile(abs(a), 90)),
                              absolute_error_p95=float(np.percentile(abs(a), 95))))
    cb = fig.colorbar(im, ax=axes, shrink=.65, pad=.025)
    cb.set_label('Prediction − truth (mph); shared symmetric-log scale')
    save(fig, 'residual_known')
    pd.DataFrame(rows).to_csv(OUT / 'plot_values.csv', index=False)
    pd.DataFrame(stats).to_csv(OUT / 'residual_summary.csv', index=False)
    (OUT / 'provenance.json').write_text(json.dumps(dict(
        source=str(source.relative_to(ROOT)), scope='all 64 known-node paired validation windows',
        arrival='counterfactual arrival on known nodes; not unseen-node cold start',
        residual_cells=4*3*64*12, color_max_absolute=maximum,
        previous_held_out_results_retained=True, new_inference=False, new_test=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
