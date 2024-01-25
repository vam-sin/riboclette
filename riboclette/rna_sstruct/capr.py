import subprocess
from tqdm.auto import tqdm
import pandas as pd
import os


class CapRInterface:
    def __init__(self, capr_in_fname="capr_in.txt", capr_out_fname="capr_out.txt"):
        self.capr_in_fname = capr_in_fname
        self.capr_out_fname = capr_out_fname

    def get_categories(self, df: pd.DataFrame, maximal_span: int = 2000):
        """
        Input df is supposed to have transcript and sequence columns.
        """
        results = []
        for trans, seq in tqdm(zip(df.transcript, df.sequence), total=len(df.sequence)):
            capr_cat = self._get_capr_categories(trans, seq, maximal_span=maximal_span)
            results.append((seq, capr_cat))

        # Remove files
        os.remove(self.capr_in_fname)
        os.remove(self.capr_out_fname)
        return pd.DataFrame(results)

    def _get_capr_categories(
        self,
        trans: str,
        seq: str,
        maximal_span: int = 2000,
    ):
        f = open(self.capr_in_fname, "w")
        f.writelines([f">{trans}\n", seq])
        f.close()

        capr_path = os.environ.get("CAPR_PATH")
        if capr_path is None:
            raise KeyError("CAPR_PATH not defined in envirorment.")

        subprocess.run(
            [capr_path, self.capr_in_fname, self.capr_out_fname, f"{maximal_span}"]
        )

        df = pd.read_csv(
            self.capr_out_fname,
            skiprows=1,
            header=None,
            delim_whitespace=True,
            index_col=0,
        )
        df = df.T
        df.columns = [c[0] for c in df.columns]
        df = df.assign(nucleotide_idx=df.index)
        df = pd.melt(
            df, id_vars="nucleotide_idx", value_vars=df.columns, var_name="cat"
        )
        df.iloc[df.groupby("nucleotide_idx").value.idxmax()].reset_index()

        return "".join(df.cat.values)
