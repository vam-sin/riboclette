import marimo

__generated_with = "0.10.9"
app = marimo.App(width="full")


@app.cell
def _():
    allow_overwrite: bool = True
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
    from permetrics import RegressionMetric
    import h5py
    from collections import defaultdict
    from tqdm.auto import tqdm
    from itertools import product
    from scipy.stats import skew, pearsonr
    from pyhere import here
    from utils import get_pdf_dimensions_in_cm, journal_plotting_ctx 
    import config
    return (
        Affine2D,
        GridSpec,
        GridSpecFromSubplotSpec,
        RegressionMetric,
        ScaledTranslation,
        config,
        defaultdict,
        get_pdf_dimensions_in_cm,
        h5py,
        here,
        journal_plotting_ctx,
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
    BRIGHT_PALETTE = ['#e69f00', '#cc79a7','#009e73','#d55e00','#0072b2', '#f0e442','#56b4e9',]
    return (BRIGHT_PALETTE,)


@app.cell
def _(BRIGHT_PALETTE, pd):
    MODELS_PROP=[
        ('RiboMIMO', 'RiboMIMO', 'RM', False, BRIGHT_PALETTE[0]),
        #('BiLSTM-CSH', 'BiLSTM [SH]', 'BLSH', False, BRIGHT_PALETTE[1]),
        ('BiLSTM-DH', 'BiLSTM [DH]', 'BLDH', True, BRIGHT_PALETTE[2]),
        #('XLNet-CSH', 'Riboclette [SH]', 'RSH', False, BRIGHT_PALETTE[3]),
        ('XLNet-DH', 'Riboclette [DH]', 'RDH', True,BRIGHT_PALETTE[4]),
        ('XLNet-PLabelDH_exp1', 'Riboclette [DH] + PL [T]', 'RDHPLT', True,BRIGHT_PALETTE[5]),
        ('XLNet-PLabelDH_exp2', 'Riboclette [DH] + PL [G]', 'RDHPLG', True,BRIGHT_PALETTE[6]),
    ]
    MODELS_PROP = pd.DataFrame(MODELS_PROP).rename(columns={0:'fname', 1:'model', 2:'abbr',3:'is_dh',4:'color'})
    MODELS_PROP
    return (MODELS_PROP,)


@app.cell
def _():
    CONDITIONS_FIXNAME = {
            'CTRL': 'CTRL',
            'VAL': 'VAL (V)',
            'ILE': 'ILE (I)',
            'LEU': 'LEU (L)',
            'LEU_ILE': '(L, I)', 
            'LEU_ILE_VAL': '(L, I, V)'}
    CONDITIONS_FIXNAME_INV = {v: k for k, v in CONDITIONS_FIXNAME.items()}
    return CONDITIONS_FIXNAME, CONDITIONS_FIXNAME_INV


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
def _(Metric, RegressionMetric):
    class PCC(Metric):
        def compute(self, y_true, y_pred):
            evaluator = RegressionMetric(y_true, y_pred)
            return evaluator.pearson_correlation_coefficient()

    class MAE(Metric):
        def compute(self, y_true, y_pred):
            evaluator = RegressionMetric(y_true, y_pred)
            return evaluator.mean_absolute_error()

    class MAAPE(Metric):
        def compute(self, y_true, y_pred, eps=1e-10):
            evaluator = RegressionMetric(y_true+eps, y_pred)
            return evaluator.mean_arctangent_absolute_percentage_error()*100
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

                # sample_perf["depr_skew"].append(skew(depr_true[sample_idx], nan_policy="omit"))
                # sample_perf["depr_kurtosis"].append(kurtosis(depr_true[sample_idx], nan_policy="omit"))
                # sample_perf["depr_var"].append(np.nanvar(depr_true[sample_idx]))
                # sample_perf["depr_std"].append(np.nanstd(depr_true[sample_idx]))
                # sample_perf["depr_mean"].append(np.nanmean(np.abs(depr_true[sample_idx])))
                # sample_perf["depr_max"].append(np.nanmax(depr_true[sample_idx]))

                # sample_perf["ctrl_skew"].append(skew(ctrl_true[sample_idx], nan_policy="omit"))
                # sample_perf["ctrl_kurtosis"].append(kurtosis(ctrl_true[sample_idx], nan_policy="omit"))
                # sample_perf["ctrl_var"].append(np.nanvar(ctrl_true[sample_idx]))
                # sample_perf["ctrl_std"].append(np.nanstd(ctrl_true[sample_idx]))
                # sample_perf["ctrl_mean"].append(np.nanmean(ctrl_true[sample_idx]))

                # sample_perf["cond_mean"].append(np.nanmean(cond_true[sample_idx]))

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
    data=pd.concat([
        compute_performance_metrics(os.path.join(DATA_PATH, f'{fname}_S{seed}.h5'), abbr, is_dh, seed)
        for (_, (fname, _, abbr, is_dh, _)), seed in tqdm(list(product(MODELS_PROP.iterrows(), [1,2,3,4,42])))
    ])
    return (data,)


@app.cell
def _(DATA_PATH, defaultdict, h5py, np, os, re, skew):
    def compute_stats(fpath):

        sample_perf = defaultdict(list)
        with h5py.File(fpath, 'r') as f:
            ctrl_pred = f['y_pred_ctrl'][:]
            depr_pred = f['y_pred_depr_diff'][:]
            ctrl_true = f['y_true_ctrl'][:]
            depr_true = f['y_true_dd'][:]

            conditions = f['condition'][:].astype('U')
            conditions = np.char.replace(conditions, '-', '_')

            transcripts = [re.sub(r'[^A-Za-z0-9.]', '', t) for t in f['transcript'][:].astype('U')]

        for sample_idx in range(ctrl_true.shape[0]):
            condition = conditions[sample_idx]
            sample_perf["condition"].append(condition)
            sample_perf["n_codons"].append(len(ctrl_true[sample_idx]))

            sample_perf["transcript"].append(transcripts[sample_idx])

            sample_perf["ctrl_mean"].append(np.nanmean(ctrl_true[sample_idx]))
            sample_perf["ctrl_var"].append(np.nanvar(ctrl_true[sample_idx]))
            sample_perf["ctrl_skew"].append(skew(ctrl_true[sample_idx], nan_policy='omit'))

            sample_perf["depr_mean"].append(np.nanmean(depr_true[sample_idx]))
            sample_perf["depr_var"].append(np.nanvar(depr_true[sample_idx]))
            sample_perf["depr_skew"].append(skew(depr_true[sample_idx], nan_policy='omit'))

        return sample_perf
    stast_best_model = compute_stats(os.path.join(DATA_PATH, f'XLNet-PLabelDH_exp1_S1.h5'))
    return compute_stats, stast_best_model


@app.cell
def _(pd, stast_best_model):
    pd.DataFrame(stast_best_model).query('condition == VAL"')
    return


@app.cell
def _(data, np, pd, sns, stast_best_model):
    plot_data = (pd.DataFrame(stast_best_model).query('condition == "VAL"').merge(data.query('abbr == "RDHPLT" and condition == "VAL" and seed == 1'), on='transcript')
                 .assign(is_liver=lambda df: df.transcript.isin(np.load('/nfs_home/nallapar/final/riboclette/riboclette/models/data/extras/liver_transcripts.npz')['arr_0']))
                 .get(['depr_PCC', 'depr_skew', 'is_liver'])
                 .astype(float))

    sns.scatterplot(data=plot_data.query('depr_PCC > 0'), x='depr_PCC', y='depr_skew', hue='is_liver')
    return (plot_data,)


@app.cell
def _(data, np, pd, sns, stast_best_model):
    plot_data_1 = pd.DataFrame(stast_best_model).query('condition == "CTRL"').merge(data.query('abbr == "RDHPLT" and condition == "CTRL" and seed == 1'), on='transcript').assign(is_liver=lambda df: df.transcript.isin(np.load('/nfs_home/nallapar/final/riboclette/riboclette/models/data/extras/liver_transcripts.npz')['arr_0'])).get(['ctrl_PCC', 'ctrl_skew', 'is_liver']).astype(float)
    sns.scatterplot(data=plot_data_1.query('ctrl_PCC > 0'), x='ctrl_PCC', y='ctrl_skew', hue='is_liver')
    return (plot_data_1,)


@app.cell
def _(plot_data_1):
    plot_data_1
    return


@app.cell
def _(data):
    data_1 = data.replace({'condition': {'VAL': 'VAL (V)', 'ILE': 'ILE (I)', 'LEU': 'LEU (L)', 'LEU_ILE': '(L, I)', 'LEU_ILE_VAL': '(L, I, V)'}})
    return (data_1,)


@app.cell
def _():
    # data = data.query('condition != "CTRL"')
    return


@app.cell
def _():
    TEXTWIDTH_CM = 18.3
    CM_TO_INCH = 1/2.54  # centimeters in inches
    CONDITION_ORDER = ['CTRL', 'ILE (I)', 'LEU (L)', 'VAL (V)', '(L, I)', '(L, I, V)']
    return CM_TO_INCH, CONDITION_ORDER, TEXTWIDTH_CM


@app.cell
def _():
    # df_condition_pcc = (pd.read_csv('Performance.csv')
    #  .melt(id_vars='Condition', var_name='Model', value_name='PCC')
    #  .assign(
    #      Seed=lambda df: [[el for el in re.findall(r"S\d+", x)] for x in df.PCC],
    #      PCC=lambda df: [[float(el) for el in re.findall(r"\d+\.\d+", x)] for x in df.PCC])
    #  .explode(['Seed','PCC'])
    #  .replace({
    #      'Model': {
    #       ' XL-Net 1 (64+6) DH Seed Best': 'XLNet DH',
    #       'XL-Net 1 DH (PLabel)': 'XLNet DH+PL',
    #       'XL-Net 1 SH (L: MAE+PCC)': 'XLNet SH', 
    #       'LSTM DH (L: MAE+PCC)': 'LSTM DH',
    #       'LSTM SH (L: MAE+PCC)': 'LSTM SH'}, 
    #     'Condition': {
    #         'CTRL + Liver': 'CTRL', 
    #         'LEU_ILE': '(LEU, ILE)', 
    #         'LEU_ILE_VAL': '(LEU, ILE, VAL)'
    #     }}
    # ))
    # df_condition_pcc
    return


@app.cell
def _(np, pd):
    palette_df = pd.DataFrame.from_dict(dict(
        Model = ['Riboclette DH+IM', 'Riboclette DH', 'LSTM DH', 'RiboMIMO'],
        Palette = np.array(['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB'])[[0,1,4,5]]
    ))
    palette_dict = dict(zip(palette_df['Model'], palette_df['Palette']))
    return palette_df, palette_dict


@app.cell
def _():
    # df_condition_pcc = (
    #     pd.DataFrame.from_dict({
    #         'Order': [1, 3, 2, 5, 6, 4],
    #         'Condition': ['CTRL', 'LEU', 'ILE', '(LEU, ILE)', '(LEU, ILE, VAL)', 'VAL'],
    #         'Riboclette DH': [0.5979, 0.6893, 0.6382, 0.6798, 0.689, 0.6997],
    #         'LSTM DH': [0.525, 0.6092, 0.5163, 0.6005, 0.656, 0.6602],
    #         'RiboMIMO': [0.3898, 0.5958, 0.5421, 0.5899, 0.5888, 0.6129]
    #     })
    #     .melt(id_vars=['Condition', 'Order'], var_name='Model', value_name='PCC')
    #     .assign(Model=lambda df: pd.Categorical(df.Model, ['RiboMIMO', 'LSTM DH', 'Riboclette DH']))
    #     .sort_values('Model'))
    # df_condition_pcc
    return


@app.cell
def _():
    # PCCs =[0.6816, 0.6759, 0.68, 0.6788, 0.6789] + [0.659, 0.6438, 0.6557, 0.6209, 0.6531] + [0.5831, 0.58, 0.5808, 0.5806, 0.5811] + [0.5532]
    # MAEs = [.218, .2213, .2176, .2168, .2198] + [0.2231, 0.231, 0.2242, 0.2328, 0.2232] + [np.nan] * 6
    # Models = ['Riboclette DH+IM'] * 5 + ['Riboclette DH'] * 5 + ['LSTM DH'] * 5 + ['RiboMIMO']

    # df_overall_pcc = (
    #     pd.DataFrame(np.array([PCCs, MAEs]).T, columns=['PCC', 'MAE']).assign(Model=Models)
    #     .assign(Model=lambda df: pd.Categorical(df.Model, ['RiboMIMO', 'LSTM DH', 'Riboclette DH', 'Riboclette DH+IM']))
    #     .sort_values('Model'))
    # df_overall_pcc
    return


@app.cell
def _():
    # df_imputation = (
    #     pd.DataFrame.from_dict({
    #         'Trainset Size': [17897] * 5 + [17897] * 5 + [92700] * 5 + [128808] * 5,
    #         'PCC': [0.659, 0.6438, 0.6557, 0.6209, 0.6531] + [0.6644, 0.6301, 0.6605, 0.6303, 0.6338] + [0.6793, 0.6768, 0.6795, 0.6795, 0.6817] + [0.6757, 0.6754, 0.6802, 0.6732, 0.6797],
    #         'MAE': [0.2231, 0.231, 0.2242, 0.2328, 0.2232] + [0.2198, 0.2273, 0.2226, 0.2198, 0.2198] + [0.2236, 0.2243, 0.2209, 0.2253, 0.2217] + [0.2198, 0.2271, 0.2197, 0.2269, 0.2264],
    #         'Imputed': ['None'] * 5 + ['T'] * 5 + ['(T, D)'] * 5 + ['(T, D, M)'] * 5,
    #         'Color': [PALETTE_TO_MODEL["Riboclette DH"]] * 5 + [BRIGHT_PALETTE[6]] * 5 + [PALETTE_TO_MODEL["Riboclette DH+IM"]] * 5 + [BRIGHT_PALETTE[5]] * 5
    #     }))
    # df_imputation
    return


@app.cell
def _(CONDITION_ORDER, data_1):
    seed_cond_mean = data_1.groupby(['abbr', 'condition', 'seed']).cond_PCC.mean().reset_index()
    seed_cond_mean.query('abbr == "RM"').groupby('condition').cond_PCC.agg(['mean', 'std']).reset_index().set_index('condition').loc[CONDITION_ORDER]
    return (seed_cond_mean,)


@app.cell
def _():
    # (data
    #         .groupby(['abbr', 'condition', 'seed'])
    #         .cond_PCC
    #         .mean()
    #         .groupby(['abbr', 'seed'])
    #         .mean()
    #         .groupby('abbr')
    #         .agg(['mean', 'std'])
    #         .reset_index()
    #         .merge(MODELS_PROP, on='abbr'))
    return


@app.cell
def _():
    # for a in MODELS_PROP[['abbr', 'color']].values:
    #     print(a)
    return


@app.cell
def _(CONDITION_ORDER):
    CONDITION_ORDER
    return


@app.cell
def _(
    CM_TO_INCH,
    CONDITION_ORDER,
    MODELS_PROP,
    TEXTWIDTH_CM,
    allow_overwrite,
    config,
    data_1,
    here,
    journal_plotting_ctx,
    np,
    os,
    pd,
    plt,
):
    with plt.style.context(['grid', 'nature', 'no-latex']), journal_plotting_ctx():
        plt.rcParams['svg.fonttype'] = 'none'
        fig = plt.figure(figsize=(TEXTWIDTH_CM * CM_TO_INCH, 8 * CM_TO_INCH), layout="constrained")
        gs = fig.add_gridspec(nrows=2, ncols=2)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 0])
        subgs = gs[1,1].subgridspec(nrows=1, ncols=3, wspace=0.01)
        ax3 = fig.add_subplot(subgs[0])
        ax4 = fig.add_subplot(subgs[1])
        ax5 = fig.add_subplot(subgs[2])
        df_overall_pcc = data_1.groupby(['abbr', 'condition', 'seed']).cond_PCC.mean().groupby(['abbr', 'seed']).mean().groupby('abbr').agg(['mean', 'std']).reset_index().merge(MODELS_PROP, on='abbr', how='right')
        ax = ax0
        ax.axis('off')
        ax.set_title('Pseudo-Labeling')
        ax.text(x=-0.08, y=1.15, s='a.', fontweight='bold', fontsize=8, ha='right', va='center', transform=ax.transAxes)
        ax = ax1
        width = 0.15
        multiplier = -2
        x_ticks = np.arange(data_1.condition.nunique() + 1)
        seed_cond_mean_1 = data_1.groupby(['abbr', 'condition', 'seed']).cond_PCC.mean().reset_index()
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
        ax.set_title('Condition-wise Model Performance')
        ax.text(x=-0.05, y=1.15, s='b.', fontweight='bold', fontsize=8, ha='right', va='center', transform=ax.transAxes)
        ax = ax2
        DEPR_CONDITION_ORDER = CONDITION_ORDER[1:]
        width = 0.12
        multiplier = -1.5
        x_ticks = np.arange(data_1.condition.nunique() - 1)
        seed_cond_mean_1 = data_1.groupby(['abbr', 'condition', 'seed']).depr_PCC.mean().reset_index().dropna()
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
        ax.set_title('Condition-wise Deprivation Performance')
        ax.text(x=-0.08, y=1.15, s='c.', fontweight='bold', fontsize=8, ha='right', va='center', transform=ax.transAxes)
        ax = ax3
        trans_wise_data = data_1.query('(abbr == "RDHPLT" and seed == 1) or (abbr == "RM" and seed == 1)').pivot(index=['transcript', 'condition'], columns=['abbr'], values='cond_PCC')
        ax.scatter(y=trans_wise_data['RDHPLT'], x=trans_wise_data['RM'], s=1, color='#332288', alpha=0.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('RiboMIMO')
        ax.set_ylabel('Riboclette')
        ax.set_xticks([0.25, 0.5, 0.75], [0.25, 0.5, 0.75])
        ax.set_yticks([0.25, 0.5, 0.75], [0.25, 0.5, 0.75])
        ax.text(x=-0.25, y=1.15, s='d.', fontweight='bold', fontsize=8, ha='right', va='center', transform=ax.transAxes)
        ax = ax4
        trans_wise_data = data_1.query('(abbr == "RDHPLT" and seed == 1) or (abbr == "BLDH" and seed == 1)').pivot(index=['transcript', 'condition'], columns=['abbr'], values='cond_PCC')
        ax.scatter(y=trans_wise_data['RDHPLT'], x=trans_wise_data['BLDH'], s=1, color='#332288', alpha=0.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('BiLSTM [DH]')
        ax.set_xticks([0.25, 0.5, 0.75], [0.25, 0.5, 0.75])
        ax.set_yticks([0.25, 0.5, 0.75], [])
        ax.set_title('Pairwise Model Comparison')
        ax = ax5
        trans_wise_data = data_1.query('(abbr == "RDHPLT" and seed == 1) or (abbr == "RDH" and seed == 1)').pivot(index=['transcript', 'condition'], columns=['abbr'], values='cond_PCC')
        ax.scatter(y=trans_wise_data['RDHPLT'], x=trans_wise_data['RDH'], s=1, color='#332288', alpha=0.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0.25, 0.5, 0.75], [0.25, 0.5, 0.75])
        ax.set_yticks([0.25, 0.5, 0.75], [])
        ax.set_xlabel('Riboclette [DH]')
        fig.legend(*ax1.get_legend_handles_labels(), bbox_transform=fig.transFigure, loc='center', bbox_to_anchor=(0.5, -0.05), borderaxespad=0.0, frameon=False, ncols=5)
        output_fpath = here('data', 'results', 'figures', 'figure2', 'figure2.pdf')
        if allow_overwrite or not os.path.isfile(output_fpath):
            plt.savefig(output_fpath, **config.SAVEFIG_KWARGS)
    return (
        DEPR_CONDITION_ORDER,
        abbr,
        ax,
        ax0,
        ax1,
        ax2,
        ax3,
        ax4,
        ax5,
        color,
        df_overall_pcc,
        fig,
        group,
        gs,
        idx,
        model,
        multiplier,
        offset,
        output_fpath,
        rects,
        seed_cond_mean_1,
        subgs,
        trans_wise_data,
        width,
        x_ticks,
    )


@app.cell
def _(get_pdf_dimensions_in_cm):
    get_pdf_dimensions_in_cm()
    return


@app.cell
def _(seed_cond_mean_1):
    seed_cond_mean_1
    return


@app.cell
def _():
    #pd.read_csv(os.path.join(DATA_PATH, "replicates_pcc.csv"), index_col=0).loc[]
    return


@app.cell
def _(df_imputation):
    data_2 = df_imputation.groupby('Experiment').agg({'PCC': ['mean', 'std'], 'Trainset Size': 'mean'}).reset_index()
    data_2.columns = ['Experiment', 'PCC_mean', 'PCC_std', 'Trainset Size']
    data_2
    return (data_2,)


@app.cell
def _(
    Affine2D,
    CM_TO_INCH,
    TEXTWIDTH_CM,
    ax1,
    ax3,
    df_imputation,
    fig,
    plt,
):
    with plt.style.context(['science', 'nature', 'grid', 'bright']):
        ax_1 = plt.figure(constrained_layout=True, figsize=(TEXTWIDTH_CM * CM_TO_INCH, 6 * CM_TO_INCH))
        data_3 = df_imputation.groupby('Experiment').agg({'PCC': ['mean', 'std'], 'Trainset Size': 'mean'}).reset_index()
        data_3.columns = data_3.columns.map(lambda x: '_'.join([str(i) for i in x]) if x[0] == 'PCC' else x[0])
        for idx_1, row in data_3.sort_values('Trainset Size').iterrows():
            if 'GC' in row.Experiment:
                sign = 1 if 'IM' in row.Experiment else -1
                trans = Affine2D().translate(sign * 1000.0, 0.0) + ax3.transData
                ax1.errorbar(x=row['Trainset Size'], y=row['PCC_mean'], yerr=row['PCC_std'], fmt='o', markersize=4, transform=trans, capsize=3, label=row.Experiment)
            else:
                ax1.errorbar(x=row['Trainset Size'], y=row['PCC_mean'], yerr=row['PCC_std'], fmt='o', markersize=4, capsize=3, label=row.Experiment)
        ax1.legend()
        ax1.set_xticks([0, 50000.0, 100000.0, 150000.0])
        ax1.text(x=0.05, y=0.45, s='c.', fontweight='bold', fontsize=12, ha='center', va='center', transform=fig.transFigure)
        ax1.set_title('Imputation')
        ax1.set_xlabel('Trainset Size')
        ax1.set_ylabel('PCC')
    return ax_1, data_3, idx_1, row, sign, trans


@app.cell
def _(data_3):
    data_3.sort_values('Trainset Size_mean')
    return


@app.cell
def _(data_3):
    data_3
    return


@app.cell
def _(data_3):
    data_3
    return


@app.cell
def _(df_imputation):
    df_imputation.groupby('Experiment').PCC.agg(['Mean', 'Std']).reset_index()
    return


@app.cell
def _(df_imputation):
    df_imputation.groupby('Experiment').std()
    return


@app.cell
def _(df_imputation):
    df_imputation.groupby('Experiment').mean()
    return


@app.cell
def _(df_imputation):
    df_imputation['Trainset Size'] = df_imputation['Trainset Size'].astype('float64')
    return


@app.cell
def _(df_imputation):
    df_imputation
    return


@app.cell
def _(df_overall_pcc):
    df_overall_pcc.groupby('Model').mean().dropna()
    return


@app.cell
def _(mpl):
    mpl.rcParams.keys()
    return


if __name__ == "__main__":
    app.run()
