import marimo

__generated_with = "0.11.9"
app = marimo.App(width="full")


@app.cell
def _(mo):
    allow_overwrite = mo.cli_args().get("overwrite") or True
    return (allow_overwrite,)


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.transforms import Affine2D
    from matplotlib.transforms import ScaledTranslation
    import seaborn as sns
    import scienceplots
    import numpy as np
    import re
    import os
    import h5py
    from collections import defaultdict
    from tqdm.auto import tqdm
    from itertools import product
    from scipy.stats import skew, pearsonr
    from pyhere import here
    from utils import get_pdf_dimensions_in_cm, journal_plotting_ctx 
    import config
    import marimo as mo
    return (
        Affine2D,
        GridSpec,
        GridSpecFromSubplotSpec,
        ScaledTranslation,
        config,
        defaultdict,
        get_pdf_dimensions_in_cm,
        h5py,
        here,
        journal_plotting_ctx,
        mo,
        mpl,
        np,
        os,
        pd,
        pearsonr,
        plt,
        product,
        re,
        scienceplots,
        skew,
        sns,
        tqdm,
    )


@app.cell
def _(here):
    DATA_PATH = here("data", "results", "predictions")
    return (DATA_PATH,)


@app.cell
def _():
    def rgb_to_hex(r, g, b):
        return '#{:02X}{:02X}{:02X}'.format(r, g, b)

    BRIGHT_PALETTE = [rgb_to_hex(218, 142, 192), rgb_to_hex(249, 218, 86), rgb_to_hex(145, 159, 199), rgb_to_hex(176, 215, 103)]
    return BRIGHT_PALETTE, rgb_to_hex


@app.cell
def _(BRIGHT_PALETTE, pd):
    MODELS_PROP=[
        ('RiboMIMO', 'RiboMIMO', 'RM', False, BRIGHT_PALETTE[0]),
        #('BiLSTM-CSH', 'BiLSTM [SH]', 'BLSH', False, BRIGHT_PALETTE[1]),
        ('BiLSTM-DH', 'Riboclette BiLSTM', 'BLDH', True, BRIGHT_PALETTE[1]),
        #('XLNet-CSH', 'Riboclette [SH]', 'RSH', False, BRIGHT_PALETTE[3]),
        ('XLNet-DH', 'Riboclette Tr', 'RDH', True,BRIGHT_PALETTE[2]),
        ('XLNet-PLabelDH_exp1', 'Riboclette Tr-Pl', 'RDHPLT', True,BRIGHT_PALETTE[3]),
        #('XLNet-PLabelDH_exp2', 'Riboclette MHA-PLG', 'RDHPLG', True,BRIGHT_PALETTE[6]),
    ]
    MODELS_PROP = pd.DataFrame(MODELS_PROP).rename(columns={0:'fname', 1:'model', 2:'abbr',3:'is_dh',4:'color'})
    MODELS_PROP
    return (MODELS_PROP,)


@app.cell
def _(mo):
    mo.md(r"""# Compute Metrics""")
    return


@app.cell
def _(np):
    from abc import ABC, abstractmethod

    class Metric(ABC):
        @abstractmethod
        def compute(self, y_true, y_pred):
            pass

        def masked_compute(self, y_true, y_pred):
            nan_mask = ~np.isnan(y_true)
            y_true = y_true[nan_mask]
            y_pred = y_pred[nan_mask]
            return self.compute(y_true, y_pred)
    return ABC, Metric, abstractmethod


@app.cell
def _(Metric, np, pearsonr):
    class PCC(Metric):
        def compute(self, y_true, y_pred):
            return pearsonr(y_true, y_pred)[0]

    class MAE(Metric):
        def compute(self, y_true, y_pred):
            return np.nanmean(np.abs(y_true - y_pred))

    class MAAPE(Metric):
        def compute(self, y_true, y_pred, eps=1e-9):
            return 100*np.nanmean(np.arctan(np.abs((y_true - y_pred) / (y_true + eps))))
    return MAAPE, MAE, PCC


@app.cell
def _(MAAPE, MAE, PCC):
    metric_to_func = dict(
        PCC=PCC(),
        MAE=MAE(),
        MAAPE=MAAPE()
    )
    return (metric_to_func,)


@app.cell
def _(defaultdict, h5py, metric_to_func, np, pd, re):
    def compute_performance_metrics(fpath, abbr, is_dh, seed):
        sample_perf = defaultdict(list)
        with h5py.File(fpath, 'r') as f:
            if is_dh:
                ctrl_pred = f['y_pred_ctrl'][:]
                depr_pred = f['y_pred_depr_diff'][:]
                ctrl_true = f['y_true_ctrl'][:]
                depr_true = f['y_true_dd'][:]

            cond_pred = f['y_pred_full'][:]
            cond_true = f['y_true_full'][:]

            conditions = f['condition'][:].astype('U')
            conditions = np.char.replace(conditions, '-', '_')

            transcripts = [re.sub(r'[^A-Za-z0-9.]', '', t) for t in f['transcript'][:].astype('U')]

            for sample_idx in range(cond_true.shape[0]):
                condition = conditions[sample_idx]
                for m, f in metric_to_func.items():

                    if is_dh and condition != 'CTRL':
                        depr_val = f.masked_compute(y_true=depr_true[sample_idx], y_pred=depr_pred[sample_idx])
                    else:
                        depr_val = None
                    sample_perf[f"depr_{m}"].append(depr_val)

                    if is_dh:
                        ctrl_val = f.masked_compute(y_true=ctrl_true[sample_idx], y_pred=ctrl_pred[sample_idx])
                    else:
                        ctrl_val = None
                    sample_perf[f"ctrl_{m}"].append(ctrl_val)

                    sample_perf[f"cond_{m}"].append(
                       f.masked_compute(y_true=cond_true[sample_idx], y_pred=cond_pred[sample_idx]))

                sample_perf["condition"].append(condition)
                sample_perf["n_codons"].append(len(cond_true[sample_idx]))

                sample_perf["transcript"].append(transcripts[sample_idx])

        return pd.DataFrame(sample_perf).assign(abbr=abbr, seed=seed)
    return (compute_performance_metrics,)


@app.cell
def _(
    DATA_PATH,
    MODELS_PROP,
    compute_performance_metrics,
    os,
    pd,
    product,
    tqdm,
):
    performance_metrics_df = pd.concat([
        compute_performance_metrics(os.path.join(DATA_PATH, f'{fname}_S{seed}.h5'), abbr, is_dh, seed)
        for (_, (fname, _, abbr, is_dh, _)), seed in tqdm(list(product(MODELS_PROP.iterrows(), [1,2,3,4,42])))
    ])
    return (performance_metrics_df,)


@app.cell
def _(MODELS_PROP, config, performance_metrics_df):
    _pm_df = performance_metrics_df.replace(dict(condition=config.CONDITIONS_FIXNAME))
    df_overall_pcc = _pm_df.groupby(['abbr', 'condition', 'seed']).cond_PCC.mean().groupby(['abbr', 'seed']).mean().groupby('abbr').agg(['mean', 'std']).reset_index().merge(MODELS_PROP, on='abbr', how='right')
    df_overall_pcc
    return (df_overall_pcc,)


@app.cell
def _(mo):
    mo.md(r"""# Make Performance Panel""")
    return


@app.cell
def _(
    MODELS_PROP,
    allow_overwrite,
    config,
    here,
    journal_plotting_ctx,
    mo,
    np,
    os,
    pd,
    performance_metrics_df,
    plt,
):
    def _():
        fig = plt.figure(figsize=(config.TEXTWIDTH_CM * config.CM_TO_INCH, 8 * config.CM_TO_INCH), layout="constrained")
        gs = fig.add_gridspec(nrows=2, ncols=2)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 0])
        subgs = gs[1,1].subgridspec(nrows=1, ncols=3, wspace=0.01)
        ax3 = fig.add_subplot(subgs[0])
        ax4 = fig.add_subplot(subgs[1])
        ax5 = fig.add_subplot(subgs[2])
        CONDITION_ORDER = list(config.CONDITIONS_FIXNAME.values())
        _pm_df = performance_metrics_df.replace(dict(condition=config.CONDITIONS_FIXNAME))
        df_overall_pcc = _pm_df.groupby(['abbr', 'condition', 'seed']).cond_PCC.mean().groupby(['abbr', 'seed']).mean().groupby('abbr').agg(['mean', 'std']).reset_index().merge(MODELS_PROP, on='abbr', how='right')
        ax = ax0
        ax.axis('off')
        ax.set_title('Pseudo-Labeling')
        ax.text(x=-0.08, y=1.15, s='a', fontsize=8, ha='right', va='center', transform=ax.transAxes)
        ax = ax1
        width = 0.15
        multiplier = -1.5
        x_ticks = np.arange(_pm_df.condition.nunique() + 1)
        seed_cond_mean_1 = _pm_df.groupby(['abbr', 'condition', 'seed']).cond_PCC.mean().reset_index()
        for idx, (abbr, color) in enumerate(MODELS_PROP[['abbr', 'color']].values):
            group = seed_cond_mean_1.query('abbr == @abbr').groupby('condition').cond_PCC.agg(['mean', 'std']).reset_index().set_index('condition').loc[CONDITION_ORDER]
            group = pd.concat([group, df_overall_pcc.query('abbr == @abbr').get(['mean', 'std'])])
            model = MODELS_PROP.query('abbr == @abbr').model.values[0]
            offset = width * multiplier
            rects = ax.bar(height=group['mean'], yerr=group['std'], x=x_ticks + offset, width=width, color=color, label=model)
            multiplier = multiplier + 1
        ax.axvline(x=5.5, linestyle='--', color='gray', linewidth=0.5)
        ax.set_xticks(x_ticks, CONDITION_ORDER + ['MacroAvg'])
        ax.set_ylim(0.3, 0.8)
        ax.xaxis.grid(False)
        ax.xaxis.set_ticks_position('none')
        ax.set_ylabel('PCC')
        ax.set_xlabel('Condition')
        ax.set_title('Condition-wise deprivation condition (DC)')
        ax.text(x=-0.05, y=1.15, s='b', fontsize=8, ha='right', va='center', transform=ax.transAxes)
        ax = ax2
        DEPR_CONDITION_ORDER = CONDITION_ORDER[1:]
        width = 0.12
        multiplier = -1
        x_ticks = np.arange(_pm_df.condition.nunique() - 1)
        seed_cond_mean_1 = _pm_df.groupby(['abbr', 'condition', 'seed']).depr_PCC.mean().reset_index().dropna()
        for abbr in MODELS_PROP['abbr'].values:
            if abbr not in seed_cond_mean_1.abbr.values:
                continue
            group = seed_cond_mean_1.query('abbr == @abbr').groupby('condition').depr_PCC.agg(['mean', 'std']).reset_index().set_index('condition')
            group = group.loc[DEPR_CONDITION_ORDER]
            color = MODELS_PROP.query('abbr == @abbr').color
            model = MODELS_PROP.query('abbr == @abbr').model.values[0]
            offset = width * multiplier
            rects = ax.bar(height=group['mean'], yerr=group['std'], x=x_ticks + offset, width=width, color=color, label=model)
            multiplier = multiplier + 1
        ax.set_xticks(x_ticks, DEPR_CONDITION_ORDER)
        ax.set_yticks([0.2, 0.4, 0.6], [0.2, 0.4, 0.6])
        ax.set_ylim(0, 0.8)
        ax.xaxis.grid(False)
        ax.xaxis.set_ticks_position('none')
        ax.set_ylabel('PCC')
        ax.set_xlabel('Condition')
        ax.set_title('Condition-wise deprivation difference (\u0394D)')
        ax.text(x=-0.08, y=1.15, s='c', fontsize=8, ha='right', va='center', transform=ax.transAxes)
        ax = ax3

        selected_best_seeds = performance_metrics_df.groupby(['abbr','seed','condition']).cond_PCC.mean().groupby(['abbr', 'seed']).mean().groupby('abbr').idxmax().values
        trans_wise_data = _pm_df.set_index(['abbr', 'seed']).loc[selected_best_seeds].query('abbr in ["RDHPLT", "RM"]').reset_index().pivot(index=['transcript', 'condition'], columns=['abbr'], values='cond_PCC')
        ax.scatter(y=trans_wise_data['RDHPLT'], x=trans_wise_data['RM'], s=1, color='#332288', alpha=0.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('RiboMIMO')
        ax.set_ylabel('Riboclette Tr-Pl')
        ax.set_xticks([0.25, 0.5, 0.75], [0.25, 0.5, 0.75])
        ax.set_yticks([0.25, 0.5, 0.75], [0.25, 0.5, 0.75])
        ax.text(x=-0.25, y=1.15, s='d', fontsize=8, ha='right', va='center', transform=ax.transAxes)
        ax = ax4
        trans_wise_data =  _pm_df.set_index(['abbr', 'seed']).loc[selected_best_seeds].query('abbr in ["RDHPLT", "BLDH"]').reset_index().pivot(index=['transcript', 'condition'], columns=['abbr'], values='cond_PCC')
        ax.scatter(y=trans_wise_data['RDHPLT'], x=trans_wise_data['BLDH'], s=1, color='#332288', alpha=0.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Riboclette BiLSTM')
        ax.set_xticks([0.25, 0.5, 0.75], [0.25, 0.5, 0.75])
        ax.set_yticks([0.25, 0.5, 0.75], [])
        ax.set_title('Pairwise Model Comparison')
        ax = ax5
        trans_wise_data = _pm_df.set_index(['abbr', 'seed']).loc[selected_best_seeds].query('abbr in ["RDHPLT", "RDH"]').reset_index().pivot(index=['transcript', 'condition'], columns=['abbr'], values='cond_PCC')
        ax.scatter(y=trans_wise_data['RDHPLT'], x=trans_wise_data['RDH'], s=1, color='#332288', alpha=0.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0.25, 0.5, 0.75], [0.25, 0.5, 0.75])
        ax.set_yticks([0.25, 0.5, 0.75], [])
        ax.set_xlabel('Riboclette Tr')
        fig.legend(*ax1.get_legend_handles_labels(), bbox_transform=fig.transFigure, loc='center', bbox_to_anchor=(0.5, -0.05), borderaxespad=0.0, frameon=False, ncols=5)

        output_fpath = mo.cli_args().get("output_dirpath") or here('data', 'results', 'figures', 'figure2', 'figure2.svg')
        if allow_overwrite or not os.path.isfile(output_fpath):
            plt.savefig(output_fpath, **config.SAVEFIG_KWARGS)
        plt.show()

    with plt.style.context(['grid', 'nature', 'no-latex']), journal_plotting_ctx():
        _()
    return


@app.cell
def _(mo):
    mo.md(r"""# Supplementary Figures""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Pairwise Model Comparison""")
    return


@app.cell
def _(
    MODELS_PROP,
    config,
    here,
    journal_plotting_ctx,
    mo,
    performance_metrics_df,
    plt,
    sns,
):
    def _():
        selected_best_seeds = performance_metrics_df.groupby(['abbr','seed','condition']).cond_PCC.mean().groupby(['abbr', 'seed']).mean().groupby('abbr').idxmax().values
        plot_data = performance_metrics_df.set_index(['abbr', 'seed']).loc[selected_best_seeds].reset_index().pivot(index=['transcript', 'condition'], columns='abbr', values='cond_PCC').rename(columns=dict(zip(MODELS_PROP['abbr'], MODELS_PROP['model']))).reindex(columns=MODELS_PROP['model'])
        g = sns.pairplot(plot_data, markers="+", diag_kws=dict(bins=50, color='#332288'), plot_kws=dict(s=3, color='#332288', alpha=0.4), height=config.TEXTWIDTH_INCH / 4.9)
        for ax in g.axes.flatten():
            ax.set_xlim(0, 1)
            ax.set_xticks([.25,.5, .75])
            ax.set_ylim(0, 1)
            ax.set_yticks([.25,.5, .75])
        output_fpath = mo.cli_args().get("output_dirpath") or here('data', 'results', 'figures', 'supplementary', 'supp_model_pairplot.pdf')
        plt.savefig(output_fpath, **config.SAVEFIG_KWARGS)
        plt.show()
    with plt.style.context(['grid', 'nature', 'no-latex']), journal_plotting_ctx():
        _()
    return


@app.cell
def _(mo):
    mo.md(r"""## MAAPE results""")
    return


@app.cell
def _(
    MODELS_PROP,
    allow_overwrite,
    config,
    here,
    journal_plotting_ctx,
    mo,
    np,
    os,
    pd,
    performance_metrics_df,
    plt,
):
    def _():
        fig = plt.figure(figsize=(config.TEXTWIDTH_CM * config.CM_TO_INCH, 4 * config.CM_TO_INCH), layout="constrained")
        gs = fig.add_gridspec(nrows=1, ncols=2)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])

        CONDITION_ORDER = list(config.CONDITIONS_FIXNAME.values())
        _pm_df = performance_metrics_df.replace(dict(condition=config.CONDITIONS_FIXNAME))
        df_overall_maape = _pm_df.groupby(['abbr', 'condition', 'seed']).cond_MAAPE.mean().groupby(['abbr', 'seed']).mean().groupby('abbr').agg(['mean', 'std']).reset_index().merge(MODELS_PROP, on='abbr', how='right')

        ax = ax0
        width = 0.15
        multiplier = -1.5
        x_ticks = np.arange(_pm_df.condition.nunique() + 1)
        seed_cond_mean_1 = _pm_df.groupby(['abbr', 'condition', 'seed']).cond_MAAPE.mean().reset_index()
        for idx, (abbr, color) in enumerate(MODELS_PROP[['abbr', 'color']].values):
            group = seed_cond_mean_1.query('abbr == @abbr').groupby('condition').cond_MAAPE.agg(['mean', 'std']).reset_index().set_index('condition').loc[CONDITION_ORDER]
            group = pd.concat([group, df_overall_maape.query('abbr == @abbr').get(['mean', 'std'])])
            model = MODELS_PROP.query('abbr == @abbr').model.values[0]
            offset = width * multiplier
            rects = ax.bar(height=group['mean'], yerr=group['std'], x=x_ticks + offset, width=width, color=color, label=model)
            multiplier = multiplier + 1
        ax.axvline(x=5.5, linestyle='--', color='gray', linewidth=0.5)
        ax.set_xticks(x_ticks, CONDITION_ORDER + ['MacroAvg'])
        ax.set_ylim(50, 90)
        ax.xaxis.grid(False)
        ax.xaxis.set_ticks_position('none')
        ax.set_ylabel('MAAPE')
        ax.set_xlabel('Condition')
        ax.set_title('Condition-wise deprivation condition (DC)')
        ax.text(x=-0.05, y=1.15, s='a', fontweight='bold', fontsize=8, ha='right', va='center', transform=ax.transAxes)

        ax = ax1
        DEPR_CONDITION_ORDER = CONDITION_ORDER[1:]
        width = 0.12
        multiplier = -1
        x_ticks = np.arange(_pm_df.condition.nunique() - 1)
        seed_cond_mean_1 = _pm_df.groupby(['abbr', 'condition', 'seed']).depr_MAAPE.mean().reset_index().dropna()
        for abbr in MODELS_PROP['abbr'].values:
            if abbr not in seed_cond_mean_1.abbr.values:
                continue
            group = seed_cond_mean_1.query('abbr == @abbr').groupby('condition').depr_MAAPE.agg(['mean', 'std']).reset_index().set_index('condition')
            group = group.loc[DEPR_CONDITION_ORDER]
            color = MODELS_PROP.query('abbr == @abbr').color
            model = MODELS_PROP.query('abbr == @abbr').model.values[0]
            offset = width * multiplier
            rects = ax.bar(height=group['mean'], yerr=group['std'], x=x_ticks + offset, width=width, color=color, label=model)
            multiplier = multiplier + 1
        ax.set_xticks(x_ticks, DEPR_CONDITION_ORDER)
        #ax.set_yticks([0.2, 0.4, 0.6], [0.2, 0.4, 0.6])
        ax.set_ylim(80, 100)
        ax.xaxis.grid(False)
        ax.xaxis.set_ticks_position('none')
        ax.set_ylabel('MAAPE')
        ax.set_xlabel('Condition')
        ax.set_title('Condition-wise deprivation difference (\u0394D)')
        ax.text(x=-0.08, y=1.15, s='b', fontweight='bold', fontsize=8, ha='right', va='center', transform=ax.transAxes)

        fig.legend(*ax0.get_legend_handles_labels(), bbox_transform=fig.transFigure, loc='center', bbox_to_anchor=(0.5, -0.05), borderaxespad=0.0, frameon=False, ncols=4)

        output_fpath = mo.cli_args().get("output_dirpath") or here('data', 'results', 'figures', 'supplementary', 'supp_perf_maape.svg')
        if allow_overwrite or not os.path.isfile(output_fpath):
            plt.savefig(output_fpath, **config.SAVEFIG_KWARGS)
        plt.show()

    with plt.style.context(['grid', 'nature', 'no-latex']), journal_plotting_ctx():
        _()
    return


@app.cell
def _():
    #get_pdf_dimensions_in_cm(here('data', 'results', 'figures', 'supplementary', 'model_pairplot.pdf'))
    return


if __name__ == "__main__":
    app.run()
