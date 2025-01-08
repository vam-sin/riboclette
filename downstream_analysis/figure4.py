import marimo

__generated_with = "0.10.9"
app = marimo.App(width="full")


@app.cell
def _():
    import itertools
    import marsilea as ma
    from pyhere import here
    import numpy as np
    from tqdm.auto import tqdm
    import pandas as pd
    import scienceplots
    import matplotlib.pyplot as plt
    from utils import get_pdf_dimensions_in_cm
    import marimo as mo
    import config
    return (
        config,
        get_pdf_dimensions_in_cm,
        here,
        itertools,
        ma,
        mo,
        np,
        pd,
        plt,
        scienceplots,
        tqdm,
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
    return contextmanager, journal_plotting_ctx


@app.cell
def _(mo):
    mo.md(r"""# Setup""")
    return


@app.cell
def _(here, np):
    mutations_everything = np.load(here('data', 'results', 'motifs', 'raw_bms', 'motifswAF_addStall_1000.npz'), allow_pickle=True)['mutations_everything'].item()
    keys = list(mutations_everything.keys())
    print(keys)
    return keys, mutations_everything


@app.cell
def _(keys, mutations_everything, np, tqdm):
    motif_str = []
    motif_len = []
    condition = []
    perc_increase = []
    orig_density_list = []
    new_density_list = []
    for k in tqdm(keys):
        start = k[2]
        orig_density = k[5]
        for mot in mutations_everything[k]:
            condition.append(k[4])
            new_density = mutations_everything[k][mot]
            orig_density_list.append(orig_density)
            new_density_list.append(new_density)
            try:
                perc_increase.append(np.abs((new_density - orig_density) / orig_density))
            except:
                perc_increase.append(0)
            motif_len.append(int(len(mot) / 2))
            motif_sample_dict = {}
            for _i in range(0, len(mot), 2):
                motif_sample_dict[mot[_i] - (start + 10)] = mot[_i + 1]
            motif_sample_dict = dict(sorted(motif_sample_dict.items()))
            motif_str_sample = ''
            for k1, v1 in motif_sample_dict.items():
                motif_str_sample = motif_str_sample + (str(k1) + '_' + str(v1) + '_')
            motif_str.append(motif_str_sample)
    return (
        condition,
        k,
        k1,
        mot,
        motif_len,
        motif_sample_dict,
        motif_str,
        motif_str_sample,
        new_density,
        new_density_list,
        orig_density,
        orig_density_list,
        perc_increase,
        start,
        v1,
    )


@app.cell
def _(
    condition,
    motif_len,
    motif_str,
    new_density_list,
    orig_density_list,
    pd,
    perc_increase,
):
    # make a dataframe
    df = pd.DataFrame({'motif': motif_str, 'motif_len': motif_len, 'perc_increase': perc_increase, 'condition': condition, 'orig_density': orig_density_list, 'new_density': new_density_list})
    df
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""# Create Motif Positional Heatmap""")
    return


@app.cell
def _(itertools):
    id_to_codon = {str(idx):''.join(el) for idx, el in enumerate(itertools.product(['A', 'T', 'C', 'G'], repeat=3))}
    codon_to_id = {v:k for k,v in id_to_codon.items()}
    id_to_codon['X'] = '[SKIP]'

    # Function to fill missing integers with zeros
    def create_exp_motif(lst):

        pos = lst['positions']
        motif_codons = iter(lst['motif_codons'])

        # Create a range from the minimum to the maximum value in the list
        full_range = range(min(pos), max(pos) + 1)
        # Replace missing integers with 0
        return [next(motif_codons) if i in pos else 'X' for i in full_range]
    return codon_to_id, create_exp_motif, id_to_codon


@app.cell
def _(mo):
    mo.md(r"""Define filled motifs and their starting positions""")
    return


@app.cell
def _(create_exp_motif, df, np):
    filled_motifs_df = (df
        .assign(motif_exp=lambda df: df.motif.str[:-1].str.split('_'))
        .assign(positions=lambda df: df.motif_exp.apply(lambda lst: list(map(int,lst[0::2]))))
        .assign(motif_codons=lambda df: df.motif_exp.apply(lambda lst: lst[1::2]))
        .assign(filled_motif_codons=lambda df: df.apply(create_exp_motif, axis=1))
        .assign(start_pos=lambda df: df.positions.apply(lambda lst: int(lst[0])))
        .assign(MAAPE=lambda df: df.apply(lambda row: 100*np.arctan(np.abs((row.orig_density - row.new_density) / (row.orig_density + 1e-9))), axis=1))
        .drop(columns=['motif', 'motif_len', 'perc_increase', 'orig_density', 'new_density', 'motif_exp', 'positions', 'motif_codons'])
        .assign(motif_key=lambda df: df.filled_motif_codons.astype(str))
        .groupby(['condition', 'motif_key'])
        .agg(dict(start_pos=lambda x: list(x), MAAPE='mean', filled_motif_codons='first'))
        .reset_index()
        .drop(columns=['motif_key'])
        )
    filled_motifs_df
    return (filled_motifs_df,)


@app.cell
def _(mo):
    mo.md(r"""Aggregate filled motifs""")
    return


@app.cell
def _(filled_motifs_df, id_to_codon, np):
    exclude_conditions = ['LEU_ILE', 'LEU_ILE_VAL']
    agg_filled_motifs_df = (filled_motifs_df
     .assign(freq=lambda df: df.start_pos.apply(lambda x: len(x)))
     .groupby('condition', group_keys=False)
     .apply(lambda x: x.nlargest(10, 'freq'))
     .assign(filled_motif_codons=lambda df: df.filled_motif_codons.apply(lambda x: ' '.join([id_to_codon[key] for key in x])))
     .assign(bincount=lambda df: df.start_pos.apply(lambda x: np.bincount(np.array(x)+10, minlength=16)/len(x))))
    agg_filled_motifs_df = agg_filled_motifs_df.query('not condition.isin(@exclude_conditions)')
    return agg_filled_motifs_df, exclude_conditions


@app.cell
def _(create_exp_motif, df, np):
    result_df2 = (df
     .assign(motif_exp=lambda df: df.motif.str[:-1].str.split('_'))
     .assign(positions=lambda df: df.motif_exp.apply(lambda lst: list(map(int,lst[0::2]))))
     .assign(motif_codons=lambda df: df.motif_exp.apply(lambda lst: lst[1::2]))
     .assign(filled_motif_codons=lambda df: df.apply(create_exp_motif, axis=1))
     .assign(start_pos=lambda df: df.positions.apply(lambda lst: int(lst[0])))
     .assign(MAAPE=lambda df: df.apply(lambda row: 100*np.arctan(np.abs((row.orig_density - row.new_density) / (row.orig_density + 1e-9))), axis=1))
     .drop(columns=['motif', 'motif_len', 'perc_increase', 'orig_density', 'new_density', 'motif_exp', 'positions', 'motif_codons'])
     .assign(motif_key=lambda df: df.filled_motif_codons.astype(str))
     .groupby(['condition', 'motif_key', 'start_pos'])
     .agg(dict(MAAPE='mean', filled_motif_codons='first'))
    )
    result_df2.head()
    return (result_df2,)


@app.cell
def _(agg_filled_motifs_df, result_df2):
    maape = result_df2.reset_index().pivot(index=['condition', 'motif_key'],columns='start_pos',values='MAAPE').iloc[agg_filled_motifs_df.index].fillna(0).values
    maape
    return (maape,)


@app.cell
def _(agg_filled_motifs_df, np):
    pos_freq = np.stack(agg_filled_motifs_df.bincount.values)
    return (pos_freq,)


@app.cell
def _(agg_filled_motifs_df, config, here, ma, maape, np, plt, pos_freq):
    from matplotlib.ticker import FuncFormatter, FixedLocator
    _SMALL_SIZE = 5
    _MEDIUM_SIZE = 6
    _BIGGER_SIZE = 7
    plt.rc('font', size=_SMALL_SIZE)
    plt.rc('axes', titlesize=_BIGGER_SIZE)
    plt.rc('axes', labelsize=_SMALL_SIZE)
    plt.rc('xtick', labelsize=_SMALL_SIZE)
    plt.rc('ytick', labelsize=_SMALL_SIZE)
    plt.rc('legend', fontsize=_SMALL_SIZE, title_fontsize=_BIGGER_SIZE)
    plt.rc('figure', titlesize=_BIGGER_SIZE)
    plt.rc('font', family='arial')
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    cmap = plt.get_cmap('Set3')
    colors = np.array(cmap.colors)
    fmt = FuncFormatter(lambda x, _: f'{x:.0%}')
    cells_proportion = ma.plotter.SizedMesh(
        pos_freq, 
        color=maape, 
        sizes=(0, 25), 
        linewidth=0,
        size_legend_kws=dict(title='% of occurrences', title_fontproperties=dict(weight='normal', size=config.FSM), fontsize=config.FSS, show_at=[0.25, 0.5, 0.75, 1], fmt=fmt, ncols=4), 
        color_legend_kws=dict(title='MAAPE', orientation='horizontal', height=1.5, fontsize=config.FSS, title_fontproperties=dict(weight='normal', size=config.FSM)))
    _h = ma.base.ClusterBoard(np.zeros(pos_freq.shape), width=1.4, height=3.5)
    _h.add_layer(cells_proportion)
    _h.add_left(ma.plotter.Labels(agg_filled_motifs_df.filled_motif_codons, align='center'))
    _h.group_rows(agg_filled_motifs_df.condition)
    conditions = agg_filled_motifs_df.condition.unique()
    _h.add_left(ma.plotter.Chunk(conditions, fill_colors=[config.COND_COL[c] for c in conditions]), pad=0.05)
    _h.add_top(ma.plotter.Labels(list(range(-10, -2)) + ['E', 'P', 'A'] + list(range(1, 6)), align='center', rotation=0))
    _h.add_legends('bottom', align_stacks='center', align_legends='center', stack_by='col', pad=0.2)
    _h.render()
    plt.savefig(here('data', 'results', 'figures', f'figure4_filled_motif_pos.svg'), dpi=600, bbox_inches='tight', pad_inches=0.0)
    plt.show()
    return (
        FixedLocator,
        FuncFormatter,
        cells_proportion,
        cmap,
        colors,
        conditions,
        fmt,
    )


@app.cell
def _(get_pdf_dimensions_in_cm, here):
    get_pdf_dimensions_in_cm(here('data', 'results', 'figures', f'filled_motif_pos.pdf'))
    return


@app.cell
def _(mo):
    mo.md(r"""# Create Codon Stats on Motifs""")
    return


@app.cell
def _(itertools):
    id_to_codon_1 = {idx: ''.join(el) for idx, el in enumerate(itertools.product(['A', 'T', 'C', 'G'], repeat=3))}
    codon_to_id_1 = {v: k for k, v in id_to_codon_1.items()}
    stop_codons = ['TAA', 'TAG', 'TGA']
    codonid_list = []
    for _i in range(64):
        codon = id_to_codon_1[_i]
        if codon not in stop_codons:
            codonid_list.append(_i)
    print('Number of codons:', len(codonid_list))
    condition_dict_values = {64: 'CTRL', 65: 'ILE', 66: 'LEU', 67: 'LEU_ILE', 68: 'LEU_ILE_VAL', 69: 'VAL'}
    condition_dict = {v: k for k, v in condition_dict_values.items()}
    return (
        codon,
        codon_to_id_1,
        codonid_list,
        condition_dict,
        condition_dict_values,
        id_to_codon_1,
        stop_codons,
    )


@app.cell
def _(motif_str):
    pos = []
    for _i in range(len(motif_str)):
        x = motif_str[_i].split('_')[:-1]
        x = [int(el) for el in x]
        for _j in range(0, len(x), 2):
            pos.append(x[_j])
    return pos, x


@app.cell
def _(df, id_to_codon_1, tqdm):
    condition_wise_dfs = {}
    for _c in df['condition'].unique():
        _df_c = df[df['condition'] == _c]
        _df_c = _df_c['motif'].value_counts(normalize=True).reset_index()
        _df_c.columns = ['motif', 'perc']
        c_list_fin = [[] for _ in range(21)]
        for m in tqdm(_df_c['motif']):
            m_s = m.split('_')[:-1]
            for _i in range(0, len(m_s), 2):
                c_list_fin[int(m_s[_i]) + 10].append(id_to_codon_1[int(m_s[_i + 1])])
            pos_motif = [int(x) for x in m_s[::2]]
            for _i in range(-10, 11):
                if _i not in pos_motif:
                    c_list_fin[_i + 10].append('-')
        for _i in range(-10, 11):
            _df_c['codon_' + str(_i)] = c_list_fin[_i + 10]
        _df_c.columns = ['motif', 'perc_counts'] + ['-10', '-9', '-8', '-7', '-6', '-5', '-4', '-3', 'E', 'P', 'A', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        condition_wise_dfs[_c] = _df_c
    return c_list_fin, condition_wise_dfs, m, m_s, pos_motif


@app.cell
def _(
    codon_to_id_1,
    condition_wise_dfs,
    config,
    here,
    id_to_codon_1,
    journal_plotting_ctx,
    ma,
    np,
    pd,
    plt,
):
    colors_full = plt.cm.tab20c(np.linspace(0, 1, 64))
    colors_aa = np.array(colors_full).repeat(2, 0)
    np.random.seed(42)
    np.random.shuffle(colors_aa)
    max_motif_val = 28383
    _SMALL_SIZE = 5
    _MEDIUM_SIZE = 6
    _BIGGER_SIZE = 7
    with journal_plotting_ctx():
        for _c in ['CTRL', 'ILE', 'LEU', 'VAL', 'LEU_ILE', 'LEU_ILE_VAL']:
            AA = ['Val', 'Ile', 'Leu', 'Lys', 'Asn', 'Thr', 'Arg', 'Ser', 'Met', 'Gln', 'His', 'Pro', 'Glu', 'Asp', 'Ala', 'Gly', 'Tyr', 'Cys', 'Trp', 'Phe', 'Stp']
            AA_1 = ['V', 'I', 'L', 'K', 'N', 'T', 'R', 'S', 'M', 'Q', 'H', 'P', 'E', 'D', 'A', 'G', 'Y', 'C', 'W', 'F', 'S']
            aa_3_1 = {AA[i]: AA_1[i] for i in range(len(AA))}
            _df_c = condition_wise_dfs[_c]
            _df_c = _df_c.drop(columns=['motif', 'perc_counts'])
            df_c_mat = _df_c.to_numpy()
            df_c_mat_perc = np.zeros((64, 21))
            for _i in range(21):
                codon_counts = df_c_mat[:, _i]
                num_non_dash = np.sum(codon_counts != '-')
                for _j in range(64):
                    df_c_mat_perc[_j, _i] = np.sum(codon_counts == id_to_codon_1[_j]) / num_non_dash * 100 if num_non_dash != 0 else 0
            stack_data = pd.DataFrame(df_c_mat_perc, index=[id_to_codon_1[i] for i in range(64)])
            stack_data = stack_data.sort_values(by=stack_data.columns.tolist(), ascending=False)
            stack_data_t = stack_data[stack_data.sum(axis=1) > stack_data.sum(axis=1).mean()]
            stack_data_thresh = stack_data_t / stack_data_t.sum(axis=0)
            square_side = 2
            _h = ma.Heatmap(stack_data_thresh.T, linewidth=0.5, width=square_side, height=square_side, cmap='Blues', label='Frequency', vmin=0, vmax=1, cbar_kws=dict(orientation='horizontal', height=1.5, title_fontproperties=dict(weight='normal', size=config.FSM), fontsize=config.FSS))
            colors_set = [colors_full[codon_to_id_1[i]] for i in list(stack_data_thresh.index)]
            genetic_code = pd.read_csv(config.GENCODE_FPATH, index_col=0).set_index('Codon')
            genetic_code['AminoAcid_1'] = [aa_3_1[i] for i in genetic_code['AminoAcid']]
            genetic_code = genetic_code.loc[stack_data_thresh.index]
            _h.group_cols(group=genetic_code.AminoAcid_1, order=list(set(genetic_code.AminoAcid_1)), spacing=0.007)
            colors_aa_c = []
            for _i in list(set(genetic_code.AminoAcid_1)):
                colors_aa_c.append(colors_aa[AA_1.index(_i)])
            _h.add_top(ma.plotter.Chunk(list(set(genetic_code.AminoAcid_1)), colors_aa_c, fontsize=_MEDIUM_SIZE), pad=0.025)
            _h.add_bottom(ma.plotter.Labels(list(stack_data_thresh.index), fontsize=_SMALL_SIZE, rotation=45), name='Codon')
            num_motifs_list = stack_data_t.sum(axis=0).values / 100
            for _i in range(21):
                codon_counts = df_c_mat[:, _i]
                num_non_dash = np.sum(codon_counts != '-')
                num_motifs_list[_i] = num_non_dash * num_motifs_list[_i]
            num_motifs_list = np.log(num_motifs_list + 1)
            cm = ma.plotter.ColorMesh(num_motifs_list.reshape(1, -1), cmap='Reds', vmin=0, vmax=np.log(max_motif_val), cbar_kws=dict(orientation='horizontal', title='Num. Motifs (log)', height=1.5, title_fontproperties=dict(weight='normal', size=config.FSM), fontsize=config.FSS))
            _h.add_right(cm, pad=0.05, size=0.075)
            c_text = _c
            if _c == 'LEU_ILE':
                c_text = 'LEU + ILE'
            if _c == 'LEU_ILE_VAL':
                c_text = 'LEU + ILE + VAL'
            _h.add_title(c_text, fontsize=_BIGGER_SIZE)
            pos_labels_list = ['-10', '-9', '-8', '-7', '-6', '-5', '-4', '-3', 'E', 'P', 'A', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
            _h.add_left(ma.plotter.Labels(list(pos_labels_list), fontsize=_SMALL_SIZE))
            if _c == 'CTRL':
                _h.add_legends(pad=0.025)
            _h.render()
            plt.savefig(here('data', 'results', 'figures', f'figure4_motifswAF_addStall_1000_HeatMap_{_c}.svg'), dpi=600, bbox_inches='tight', pad_inches=0.0)
    return (
        AA,
        AA_1,
        aa_3_1,
        c_text,
        cm,
        codon_counts,
        colors_aa,
        colors_aa_c,
        colors_full,
        colors_set,
        df_c_mat,
        df_c_mat_perc,
        genetic_code,
        max_motif_val,
        num_motifs_list,
        num_non_dash,
        pos_labels_list,
        square_side,
        stack_data,
        stack_data_t,
        stack_data_thresh,
    )


@app.cell
def _(get_pdf_dimensions_in_cm, here):
    _c = 'ILE'
    get_pdf_dimensions_in_cm(here('data', 'results', 'figures', f'figure4_motifswAF_addStall_1000_HeatMap_{_c}.pdf'))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
