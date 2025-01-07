import marimo

__generated_with = "0.10.9"
app = marimo.App(width="small")


@app.cell
def _():
    import config
    import h5py
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from config import ATTR_FNAMES, ENSEMBL_FPATH, TEXTWIDTH_INCH
    from pyhere import here
    from utils import read_ensembl
    import pandas as pd
    import re
    return (
        ATTR_FNAMES,
        ENSEMBL_FPATH,
        TEXTWIDTH_INCH,
        config,
        h5py,
        here,
        np,
        pd,
        plt,
        re,
        read_ensembl,
        sns,
    )


@app.cell
def _(plt):
    from contextlib import contextmanager

    @contextmanager
    def journal_plotting_ctx():
        """
        A context manager to temporarily apply specific plotting parameters for matplotlib.
        """
        # Save the current rcParams
        original_rc = plt.rcParams.copy()

        # Define the desired parameters
        SMALL_SIZE = 5
        MEDIUM_SIZE = 6
        BIGGER_SIZE = 7

        try:
            # Apply the desired rc settings
            plt.rc(
                "font", size=SMALL_SIZE, family="arial"
            )  # Default font sizes and family
            plt.rc("axes", titlesize=BIGGER_SIZE, labelsize=SMALL_SIZE)  # Axes sizes
            plt.rc("xtick", labelsize=SMALL_SIZE)  # Tick label sizes
            plt.rc("ytick", labelsize=SMALL_SIZE)
            plt.rc(
                "legend", fontsize=SMALL_SIZE, title_fontsize=BIGGER_SIZE
            )  # Legend sizes
            plt.rc("figure", titlesize=BIGGER_SIZE)  # Figure title size
            plt.rcParams["svg.fonttype"] = "none"  # SVG fonttype

            # Yield control to the block of code
            yield
        finally:
            # Restore original rcParams
            plt.rcParams.update(original_rc)
    return contextmanager, journal_plotting_ctx


@app.cell
def _(config, pd, read_ensembl):
    ensembl_df = read_ensembl(config.ENSEMBL_FPATH)
    gencode_df = pd.read_csv(config.GENCODE_FPATH)
    return ensembl_df, gencode_df


@app.cell
def _(config, ensembl_df, h5py, here):
    def search_fname_w_gene(gene_symbol: str = "Col1a2"):
        for fname in config.ATTR_FNAMES:
            with h5py.File(
                here("data", "results", "interpretability", fname), "r"
            ) as h5:
                sel_transc = ensembl_df.set_index("transcript").loc[h5["transcript"][:].astype(str)]
                if gene_symbol in sel_transc.symbol.values:
                    sequence = sel_transc.query('symbol == @gene_symbol').index[0]
                    return fname, sequence
    return (search_fname_w_gene,)


@app.cell
def _(
    TEXTWIDTH_INCH,
    config,
    ensembl_df,
    gencode_df,
    h5py,
    here,
    journal_plotting_ctx,
    np,
    plt,
    re,
    search_fname_w_gene,
    sns,
):
    def plot_transcript(
        gene_symbol = "Col1a2",
        cond_ama = ['Val'],
        asites = [166, 466, 1000, 1320]
    ):
        condition = '_'.join([str.upper(a) for a in cond_ama])
        cond_codons = gencode_df.query('AminoAcid in @cond_ama')
        fname, transcript = search_fname_w_gene(gene_symbol)
        fpath = here(
            "data", "results", "interpretability", fname
        )
        print(fname)
        fhandle = h5py.File(fpath, "r")
        sequence = ensembl_df.query("transcript == @transcript").sequence.values[0]
        codons = re.findall('...', sequence)
        transc_idx = (
            np.isin(fhandle["transcript"][:].astype(str), transcript)
            & (fhandle["condition"][:].astype("str") == condition)
        ).nonzero()[0][0]
        y_true_ctrl = fhandle["y_true_ctrl"][transc_idx]
        y_pred_ctrl = fhandle["y_pred_ctrl"][transc_idx]
        y_true_depr = fhandle["y_true_dd"][transc_idx]
        y_pred_depr = fhandle["y_pred_depr_diff"][transc_idx]
        y_true_full = fhandle["y_true_full"][transc_idx]
        y_pred_full = fhandle["y_pred_full"][transc_idx]
        y_min = np.nanmin(np.concatenate((y_true_depr, y_pred_depr)))
        y_max = np.nanmax(np.concatenate((y_true_depr, y_pred_depr)))
        n_codons = len(y_true_ctrl)
        attr_ctrl = fhandle["lig_ctrl"][transc_idx].reshape(n_codons, n_codons)
        attr_depr = fhandle["lig_dd"][transc_idx].reshape(n_codons, n_codons)
        ratio = 3.2
        width = TEXTWIDTH_INCH
        height = TEXTWIDTH_INCH / ratio
        print(height * 2.54)
        with journal_plotting_ctx():
            fig = plt.figure(figsize=(width, height), constrained_layout=True)
            gs = fig.add_gridspec(3, 4, hspace=0.1, height_ratios=[0.2, 0.2, 0.6])
            ax = fig.add_subplot(gs[0, :])
            ax.set_ylabel('Ctrl')
            ax.fill_between(
                np.arange(n_codons), np.nan_to_num(y_pred_ctrl, 0)/np.nanmax(y_pred_ctrl), color=config.PRED_COL
            )
            ax.set_xlim(-10, n_codons+10)
            ax.set_xticklabels([])
            for a in asites:
                plt.axvspan(a - 0.5, a + 0.5, color="red", alpha=0.2)
            ax = fig.add_subplot(gs[1, :])
            ax.set_ylabel('Depr. Diff.')
            ax.fill_between(
                np.arange(n_codons), np.nan_to_num(y_pred_depr, 0)/np.nanmax(y_pred_depr), color=config.PRED_COL
            )
            ax.set_xlim(-10, n_codons+10)
            for a in asites:
                plt.axvspan(a - 0.5, a + 0.5, color="red", alpha=0.2)
            ylim = (-0.3, 1)
            nup_cdn = 11
            ndown_cdn = 5
            for idx, asite in enumerate(asites):
                ax = fig.add_subplot(gs[2, idx])
                codon_attr = attr_depr[asite, :]
                ax.plot(
                    np.arange(n_codons),
                    np.nan_to_num(codon_attr, 0)/np.nanmax(codon_attr[:-1]),
                    color=config.ATTR_COL,
                    linewidth=1,
                    label='Attribution'
                )
                ax.scatter(
                    np.arange(n_codons),
                    np.nan_to_num(codon_attr, 0)/np.nanmax(codon_attr[:-1]),
                    s=1.5,
                    marker="s",
                    color=config.ATTR_COL,
                )
                ax.plot(
                    np.arange(n_codons),
                    np.nan_to_num(y_pred_depr, 0)/np.nanmax(y_pred_depr),
                    color=config.PRED_COL,
                    linewidth=1,
                    label='Predicted'
                )
                ax.scatter(
                    np.arange(n_codons),
                    np.nan_to_num(y_pred_depr, 0)/np.nanmax(y_pred_depr),
                    s=1.5,
                    marker="s",
                    color=config.PRED_COL,
                )
                ax.plot(
                    np.arange(n_codons),
                    np.nan_to_num(y_true_depr, 0)/np.nanmax(y_true_depr),
                    color=config.TRUE_COL,
                    linewidth=1,
                    label='True'
                )
                ax.scatter(
                    np.arange(n_codons),
                    np.nan_to_num(y_true_depr, 0)/np.nanmax(y_true_depr),
                    s=1.5,
                    marker="s",
                    color=config.TRUE_COL,
                )
                ax.axhline(
                    0,
                    color='black',
                    linewidth=.6,
                    zorder=1
                )
                ax.axvspan(asite - 0.5, asite + 0.5, color="red", alpha=0.1, linewidth=0, label="A-site")
                ax.text(asite+0.1, -1, codons[asite], rotation=90, ha='center', va='bottom')
                cond_idxs = np.where(np.isin(codons, cond_codons))[0]
                cond_idxs = cond_idxs[(cond_idxs > asite - nup_cdn) & (cond_idxs < asite + ndown_cdn)]
                legend_label="Depr. codon"
                for cidx in cond_idxs: 
                    ax.axvspan(
                        cidx - 0.5, cidx + 0.5, color="blue", linewidth=0, alpha=0.1, 
                        label=legend_label)
                    legend_label=""
                    ax.text(cidx+0.1, -1, codons[cidx], rotation=90, ha='center', va='bottom')
                ax.set_xlim(asite - nup_cdn-0.5, asite + ndown_cdn+0.5)
                ax.set_ylim(-1.1, 1.1)
                ax.set_xticks(
                    [asite - nup_cdn, asite, asite + ndown_cdn], 
                    [f"{asite - nup_cdn}", f"{asite}\nA-site", f"{asite + ndown_cdn}"]
                )
                if idx > 0:
                    ax.set_yticklabels([])
                else:
                    ax.set_ylabel('Attr. / Depr. Diff.')
                if idx == 3:
                    _lines, _labels = ax.get_legend_handles_labels()
            sns.despine()
            fig.legend(_lines, _labels, ncol=5, bbox_to_anchor=(1,0), loc='center right', frameon=False)
            fig.suptitle(f"{gene_symbol}, {config.CONDITIONS_FIXNAME[condition]}", fontsize=config.FSB)
            fig.align_ylabels()
            fig.supxlabel('Pos in CDS')
            plt.savefig(
                here("data", "results", "figures", f"figure5_{gene_symbol}_{condition}.svg"), **config.SAVEFIG_KWARGS
            )
            plt.show()
        fhandle.close()
    return (plot_transcript,)


@app.cell
def _(plot_transcript):
    plot_transcript(
        gene_symbol = "Col1a2",
        cond_ama = ['Val'],
        asites = [166, 466, 1000, 1320]
    )
    return


@app.cell
def _(plot_transcript):
    plot_transcript(
        'Psmd3',
        ['Ile'],
        [291, 296, 393, 439]
    )
    return


@app.cell
def _():
    # plot_transcript(
    #     'Slc38a4',
    #     ['Ile'],
    #     [11, 165, 211, 461]
    # )
    return


@app.cell
def _(plot_transcript):
    plot_transcript(
        'Col6a1',
        ['Leu', 'Ile', 'Val'],
        [87, 156, 726, 971]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
