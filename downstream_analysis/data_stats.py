import marimo

__generated_with = "0.11.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import config
    import utils
    import matplotlib.pyplot as plt
    import h5py
    from pyhere import here
    import numpy as np
    import scienceplots
    import pandas as pd
    return config, h5py, here, mo, np, pd, plt, scienceplots, utils


@app.cell
def _(mo):
    allow_overwrite = mo.cli_args().get("overwrite") or True
    return (allow_overwrite,)


@app.cell
def _(config):
    config.SPLIT_TO_FNAMES
    return


@app.cell
def _(allow_overwrite, config, h5py, here, mo, np, os, plt, utils):
    def _():
        fig, axs = plt.subplots(3, 4, figsize=(config.TEXTWIDTH_INCH, 3), constrained_layout=True)

        split_to_data = {}
        for split, fnames in config.SPLIT_TO_FNAMES.items():
            split_data = []
            conditions = []
            for f in fnames:
                f = h5py.File(here("data", "results", "interpretability", f), "r")
                split_data.append(f['y_true_full'][:])
                conditions.append([c.decode('utf8') for c in f['condition'][:]])
                f.close()
            
            split_to_data[split] = dict(
                rpf = np.concatenate(split_data),
                conditions = np.concatenate(conditions)
            )


        for col, split in enumerate(config.SPLIT_TO_FNAMES.keys()):

            y_true_full = split_to_data[split]['rpf']
            coverage = [(((~np.isnan(x)) & (x != 0)).sum())/((~np.isnan(x)).sum()) for x in y_true_full]
            length = [len(x) for x in y_true_full]
            axs[0][col].hist(coverage)
            axs[1][col].hist(length)

            axs[0][0].set_ylabel('Coverage')
            axs[1][0].set_ylabel('Sequence Length')
            axs[0][col].set_xlim(0,1.1)
            axs[1][col].set_xlim(0,2100)

            axs[0][col].set_title(split)

            unique_values, counts = np.unique(split_to_data[split]['conditions'], return_counts=True)
            ordered_values = [unique_values[list(unique_values).index(cat)] for cat in config.CONDITIONS_FIXNAME.keys() if cat in unique_values]
        
            axs[2][col].bar([config.CONDITIONS_FIXNAME[c] for c in unique_values], counts)
            axs[2][0].set_ylabel('Condition frequency')

        fig.suptitle('Split statistics')
        fig.align_ylabels()

        output_fpath = mo.cli_args().get("output_dirpath") or here('data', 'results', 'figures', 'supplementary', 'supp_dataset_statistics.pdf')
        if allow_overwrite or not os.path.isfile(output_fpath):
            plt.savefig(output_fpath, **config.SAVEFIG_KWARGS)
        plt.show()

    with plt.style.context(['grid', 'nature', 'no-latex']), utils.journal_plotting_ctx():
        _()
    return


@app.cell
def _(allow_overwrite, config, h5py, here, mo, np, os, plt, utils):
    def _():

        y_diff_pred = []
        y_diff_true = []
        condition = []
        for split, fnames in config.SPLIT_TO_FNAMES.items():
            for f in fnames:
                f = h5py.File(here("data", "results", "interpretability", f), "r")
                y_diff_pred.extend(f['y_pred_full'][:]-f['y_pred_ctrl'][:])
                y_diff_true.extend(f['y_true_full'][:]-f['y_true_ctrl'][:])
                condition.extend([c.decode('utf8') for c in f['condition'][:]])
            
        condition = np.array(condition)

        fig,(ax0, ax1) = plt.subplots(1, 2, figsize=(config.TEXTWIDTH_INCH *.6, 1.5), constrained_layout=True)

        conds_dict = config.CONDITIONS_FIXNAME.copy()
        conds_dict.pop('CTRL')
    
        positive_mean = [[np.nanmean(y_diff_pred[idx][np.where(y_diff_pred[idx]>0)]) for idx in np.where(condition == cond)[0]] for cond in conds_dict.keys() if cond != 'CTRL']
        ax1.violinplot(positive_mean)
        ax1.set_xticks(np.arange(1,6), conds_dict.values())
        ax1.set_title('Predicted Values')

        positive_mean = [[np.nanmean(y_diff_true[idx][np.where(y_diff_true[idx]>0)]) for idx in np.where(condition == cond)[0]] for cond in conds_dict.keys() if cond != 'CTRL']
        ax0.violinplot(positive_mean)
        ax0.set_xticks(np.arange(1,6), conds_dict.values())
        ax0.set_title('True Values')

        fig.suptitle('Average positive \u0394RPF gene-wise')

        output_fpath = mo.cli_args().get("output_dirpath") or here('data', 'results', 'figures', 'supplementary', 'supp_positive_rpf.pdf')
        if allow_overwrite or not os.path.isfile(output_fpath):
            plt.savefig(output_fpath, **config.SAVEFIG_KWARGS)
        plt.show()

    with plt.style.context(['grid', 'nature', 'no-latex']), utils.journal_plotting_ctx():
        _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
