import marimo

__generated_with = "0.10.9"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pyhere import here
    import scienceplots
    import itertools
    from adjustText import adjust_text
    from scipy.stats import pearsonr
    import marsilea as ma
    import palettable
    import matplotlib as mpl
    import config
    import utils
    from pyhere import here
    return (
        adjust_text,
        config,
        here,
        itertools,
        ma,
        mpl,
        np,
        palettable,
        pd,
        pearsonr,
        plt,
        scienceplots,
        utils,
    )


@app.cell
def _():
    TEXTWIDTH_CM = 18.3
    CM_TO_INCH = 1/2.54  # centimeters in inches
    CONDITION_ORDER = ['CTRL', 'ILE (I)', 'LEU (L)', 'VAL (V)', '(L, I)', '(L, I, V)']
    return CM_TO_INCH, CONDITION_ORDER, TEXTWIDTH_CM


@app.cell
def _(here, np):
    ctrl_data = np.load(here("data/results/plotting/globl_attr_plot_True.npy"))
    dd_data = np.load(here("data/results/plotting/globl_attr_plot_False.npy"))
    return ctrl_data, dd_data


@app.cell
def _(dd_data):
    dd_data
    return


@app.cell
def _():
    def global_attr_plot(ax,data, title:str, xlabel=False, ylabel=False, yticks=False, xticks=False):
        ax.hist(data, bins=21, color='#e74c3c', edgecolor='#ffffff', linewidth=1, range=(-10, 10), density=True)
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        if xticks: ax.set_xticks([-10, -4.75, 0, 4.75, 10], [-10, -5, 'A', 5, 10])
        if yticks: ax.set_yticks([0,.05,.1])
        if title: ax.set_title(title)
        if xlabel: ax.set_xlabel('Codon Distance from A-site')
        if ylabel: ax.set_ylabel('Frequency')
    return (global_attr_plot,)


@app.cell
def _(adjust_text, config, itertools, np, pearsonr):
    def plot_condition(ax, condition, stall_mean_sorted, attr_mean_sorted, codons_to_depr, ctrl_tagged_codons, colors_depr, texts, annotation_kwargs=dict(fontsize=5)):
        for codon in stall_mean_sorted[condition]:
            x = stall_mean_sorted[condition][codon]
            y = attr_mean_sorted[condition][codon]

            # Determine the color and label for the codon
            if len(codons_to_depr[codon]) != 0:
                ax.scatter(x, y, label=codon, color=colors_depr[codons_to_depr[codon][0]])#, s=40)
                if condition in ['ILE', 'LEU_ILE'] and codon in ['ATC', 'ATT']:
                    texts.append(ax.text(x, y, codon, **annotation_kwargs))
                elif condition in ['VAL', 'LEU_ILE_VAL'] and codon in ['GTC', 'GTT', 'GTA', 'GTG']:
                    texts.append(ax.text(x, y, codon, **annotation_kwargs))
            elif codon in ctrl_tagged_codons:
                ax.scatter(x, y, label=codon, color=colors_depr['CTRL'], alpha=1)#, s=40)
                if condition == 'CTRL':
                    texts.append(ax.text(x, y, codon, **annotation_kwargs))
            else:
                ax.scatter(x, y, label=codon, color='black', linewidth=0, alpha=0.5)#, s=10)

            ax.set_ylim(-0.3,+0.6)
        return texts


    def global_stalling(
        axs,
        genetic_code,
        condition_codon_stall_mean_sorted,
        condition_codon_attr_peaks_mean_sorted,
        condition_codon_attr_full_mean_sorted,
        mode='peaks'
    ):

        # Get the codons for each deprivation condition
        deprivation_conditions = ['Ile', 'Leu', 'Val', 'CTRL']
        depr_codons = {}

        for condition in deprivation_conditions:
            amino_acids = condition.split('_')
            codons_dep_cond = []
            for amino_acid in amino_acids:
                df_aa = genetic_code[genetic_code['AminoAcid'] == amino_acid]
                codons_dep_cond += df_aa['Codon'].tolist()

            depr_codons[condition.upper()] = codons_dep_cond
        depr_codons['LEU_ILE'] = []
        depr_codons['LEU_ILE_VAL'] = []

        # Ensure CTRL is the first key
        depr_codons = {k: depr_codons[k] for k in ['CTRL', 'ILE', 'LEU', 'VAL', 'LEU_ILE', 'LEU_ILE_VAL']}

        id_to_codon = {idx: ''.join(el) for idx, el in enumerate(itertools.product(['A', 'T', 'C', 'G'], repeat=3))}
        codons_to_depr = {codon: [depr for depr, codons in depr_codons.items() if codon in codons] for codon in id_to_codon.values()}
        ctrl_tagged_codons = ['GAC', 'GAA', 'GAT']

        attr_mean_sorted = (
            condition_codon_attr_peaks_mean_sorted
            if mode == 'peaks' else
            condition_codon_attr_full_mean_sorted
        )

        for i, condition in enumerate(['CTRL', 'ILE', 'VAL']):#enumerate(depr_codons.keys()):
            texts = []
            texts = plot_condition(
                axs[i],
                condition,
                condition_codon_stall_mean_sorted,
                attr_mean_sorted,
                codons_to_depr,
                ctrl_tagged_codons,
                config.COND_COL,
                texts
            )

            # Adjust the text
            adjust_text(
                texts,
                ax=axs[i], 
                #arrowprops=dict(arrowstyle='->', color='red'), 
                expand_points=(1.2, 1.2), 
                expand_text=(1.2, 1.2), 
                force_text=(0.5, 0.5)
            )

            # Calculate PCC
            x = [condition_codon_stall_mean_sorted[condition][codon] for codon in condition_codon_stall_mean_sorted[condition]]
            y = [attr_mean_sorted[condition][codon] for codon in condition_codon_stall_mean_sorted[condition]]
            corr, _ = pearsonr(x, y)

            # Fit a line to the data
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            axs[i].plot(x, p(x), "r--", color='black', alpha=0.5)

            #axs[i].set_xlabel('Mean Ribosome Counts')
            if i == 0: axs[i].set_ylabel('Mean Attribution',labelpad=-2)
            c_text = (
                'LEU + ILE' if condition == 'LEU_ILE' else
                'LEU + ILE + VAL' if condition == 'LEU_ILE_VAL' else
                condition
            )
            axs[i].set_title(f"{c_text} (PCC: {corr:.2f})")
            axs[i].tick_params(axis='both', which='major')
    return global_stalling, plot_condition


@app.cell
def _(config, ma, np, plt):
    def topk_attributions(data, genetic_code, width: float, height=float, fontsizes: list[int] = [5,6,7]):
        AA = ['Val', 'Ile', 'Leu', 'Lys', 'Asn', 'Thr', 'Arg', 'Ser', 'Met', 'Gln', 'His', 'Pro', 'Glu', 'Asp', 'Ala', 'Gly', 'Tyr', 'Cys', 'Trp', 'Phe']
        AA_corr = [config.AMINO_ACID_MAP[a] for a in AA]
        DEPR_NAMES = {'CTRL':'CTRL', 'ILE':'ILE (I)', 'LEU':'LEU (L)', 'VAL':'VAL (V)', 'LEU_ILE':'(L, I)', 'LEU_ILE_VAL':'(L,I,V)'}

        data = data.rename(columns=DEPR_NAMES)[DEPR_NAMES.values()]
        h = ma.Heatmap(data.T, width=width, height=height, cmap='RdBu_r', cbar_kws=dict(title="Normalized Mean Attribution", orientation="horizontal", height=1.5, fontsize=config.FSS, width=12, title_fontproperties=dict(weight='normal', size=config.FSM)))

        cmap = plt.get_cmap('tab20c')
        colors = np.array(cmap.colors)
        np.random.seed(42)
        np.random.shuffle(colors)

        h.group_cols(genetic_code.AminoAcid, spacing=0.002, order=AA)
        h.add_top(
                ma.plotter.Chunk(
                    AA_corr,
                    colors[:len(AA_corr)],
                    #padding=10,
                ),
                pad=0.025
            )
        h.add_bottom(ma.plotter.Labels(data.index, rotation=45), name='Codon')
        h.add_left(ma.plotter.Labels(data.columns,align='center'))
        h.group_rows([1,0,0,0,0,0], spacing=0.03, order=[1,0])
        #h.add_left(
        #       ma.plotter.Chunk(
        #           ["CH", "DH"],
        #           colors[:2],
        #       ),
        #       pad=0.1
        #   )
        h.add_legends("bottom")
        h.add_title("Codon-wise mean attribution in high stalling positions", fontsize=config.FSB)
        h.render()

        #out_width, out_height = h.figure.get_size_inches()
        #h.figure.set_size_inches(width, width * out_height / out_width)

        return h
    return (topk_attributions,)


@app.cell
def _(
    config,
    ctrl_data,
    dd_data,
    global_attr_plot,
    global_stalling,
    here,
    mpl,
    pd,
    plt,
    utils,
):
    with plt.style.context(['grid', 'nature', 'no-latex']), utils.journal_plotting_ctx():
        corrected_width = config.TEXTWIDTH_CM + 3.75
        aspect_ratio = 6
        fig = plt.figure(figsize=(corrected_width * config.CM_TO_INCH, corrected_width / aspect_ratio * config.CM_TO_INCH))
        gs = fig.add_gridspec(nrows=1, ncols=5, wspace=0.6, hspace=3)
        sub_gs = gs[0, :2].subgridspec(1, 2, wspace=0.1)
        ax = fig.add_subplot(sub_gs[0, 0])
        ax.text(-0.3, 1.1, "a", transform=ax.transAxes, fontsize=8)
        global_attr_plot(ax, ctrl_data, title='Control', ylabel=True, yticks=True, xticks=True)
        ax = fig.add_subplot(sub_gs[0, 1], sharex=ax, sharey=ax)
        global_attr_plot(ax, dd_data, title='Deprivation Difference', yticks=False, xticks=True)
        plt.tick_params('y', labelleft=False)
        subfig_coords = sub_gs.get_grid_positions(fig)
        ax.text(subfig_coords[2][1] - 0.02, subfig_coords[0] - 0.22, 'Codon Distance from A-site', transform=fig.transFigure, ha='center', fontsize=config.FSM)
        sub_gs = gs[0, 2:].subgridspec(1, 3, wspace=0.1)
        axs = [fig.add_subplot(sub_gs[row, col]) for row in range(1) for col in range(0, 3)]
        _genetic_code = pd.read_csv(here('data', 'data', 'genetic_code.csv'))
        condition_codon_attr_full_mean_sorted = pd.read_csv(here('data', 'results', 'plotting', 'condition_codon_attr_full_mean_sorted.zip'), index_col=0).to_dict()
        condition_codon_attr_peaks_mean_sorted = pd.read_csv(here('data', 'results', 'plotting', 'condition_codon_attr_peaks_mean_sorted.zip'), index_col=0).to_dict()
        condition_codon_stall_mean_sorted = pd.read_csv(here('data', 'results', 'plotting', 'condition_codon_stall_mean_sorted.zip'), index_col=0).to_dict()
        global_stalling(axs, genetic_code=_genetic_code, condition_codon_attr_full_mean_sorted=condition_codon_attr_full_mean_sorted, condition_codon_attr_peaks_mean_sorted=condition_codon_attr_peaks_mean_sorted, condition_codon_stall_mean_sorted=condition_codon_stall_mean_sorted)
        axs[0].text(-0.3, 1.1, "b", transform=axs[0].transAxes, fontsize=8)
        axs[0].set_xlabel(r'Mean RPF Counts')
        axs[1].set_xlabel(u'Mean \u0394RPF Counts')
        axs[2].set_xlabel(u'Mean \u0394RPF Counts')
        for ax in axs[1:]:
            ax.set_yticklabels([])
        legend_elements = [mpl.lines.Line2D([0], [0], marker='o', color='w', label='CTRL', markerfacecolor=config.COND_COL['CTRL'], markersize=5), mpl.lines.Line2D([0], [0], marker='o', color='w', label='ILE', markerfacecolor=config.COND_COL['ILE'], markersize=5), mpl.lines.Line2D([0], [0], marker='o', color='w', label='LEU', markerfacecolor=config.COND_COL['LEU'], markersize=5), mpl.lines.Line2D([0], [0], marker='o', color='w', label='VAL', markerfacecolor=config.COND_COL['VAL'], markersize=5), mpl.lines.Line2D([0], [0], marker='o', color='w', label='Other', markerfacecolor='black', markersize=5)]
        axs[1].legend(handles=legend_elements, loc='center', ncols=5, bbox_to_anchor=(0.5, -0.4), frameon=False)
        plt.savefig(here('data', 'results', 'figures', 'figure3_1.svg'), **config.SAVEFIG_KWARGS)
    return (
        aspect_ratio,
        ax,
        axs,
        condition_codon_attr_full_mean_sorted,
        condition_codon_attr_peaks_mean_sorted,
        condition_codon_stall_mean_sorted,
        corrected_width,
        fig,
        gs,
        legend_elements,
        sub_gs,
        subfig_coords,
    )


@app.cell
def _(
    CM_TO_INCH,
    TEXTWIDTH_CM,
    config,
    here,
    pd,
    plt,
    topk_attributions,
    utils,
):
    data = pd.read_csv(here('data', 'results', 'plotting', 'topk_attr_condition_wise.zip'), index_col=0)
    _genetic_code = pd.read_csv(config.GENCODE_FPATH, index_col=0)
    _genetic_code = _genetic_code.set_index('Codon').drop(index=['TAA', 'TAG', 'TGA'])
    with plt.style.context(['grid', 'nature', 'no-latex']), utils.journal_plotting_ctx():
        f = topk_attributions(data=data.loc[_genetic_code.index], genetic_code=_genetic_code, width=(TEXTWIDTH_CM - 1) * CM_TO_INCH, height=1.8 * CM_TO_INCH)
        plt.text(.05, .875, "c", transform=f.figure.transFigure, fontsize=8)
        f.figure.savefig(here('data', 'results', 'figures', 'figure3_2.svg'), **config.SAVEFIG_KWARGS)
    return data, f


@app.cell
def _(here, utils):
    utils.get_pdf_dimensions_in_cm(here('data', 'results', 'figures', 'figure3_2.pdf'))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
