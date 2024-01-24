import os
from typing import Dict, List, Optional, Tuple

import h5py
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from nucleotide_transformer.constants import EXTRA_NUCLEOTIDES, NUCLEOTIDES
from nucleotide_transformer.pretrained import get_pretrained_model
from nucleotide_transformer.tokenizers import StandardTokenizer, _compute_k_mers
from tqdm.auto import trange


class CodonTokenizer(StandardTokenizer):
    def __init__(
        self,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        class_token: str = "<cls>",
        eos_token: str = "<eos>",
        bos_token: str = "<bos>",
        prepend_bos_token: bool = False,
        prepend_cls_token: bool = False,
        append_eos_token: bool = False,
        tokens_to_ids: Optional[Dict[str, int]] = None,
    ):
        kmers_tokens = _compute_k_mers(6)
        standard_tokens = kmers_tokens + NUCLEOTIDES + EXTRA_NUCLEOTIDES

        StandardTokenizer.__init__(
            self,
            standard_tokens=standard_tokens,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            class_token=class_token,
            eos_token=eos_token,
            bos_token=bos_token,
            prepend_bos_token=prepend_bos_token,
            prepend_cls_token=prepend_cls_token,
            append_eos_token=append_eos_token,
            tokens_to_ids=tokens_to_ids,
        )

    def tokenize(self, sequence: str) -> Tuple[List[str], List[int]]:
        len_sequence = len(sequence)

        tokens = [sequence[i * 3 : i * 3 + 6] for i in range(len_sequence // 3 - 1)]

        if self._prepend_cls_token:
            tokens = [self._class_token] + tokens

        if self._prepend_bos_token:
            tokens = [self._bos_token] + tokens

        if self._append_eos_token:
            tokens.append(self._eos_token)

        tokens_ids = [self.token_to_id(tok) for tok in tokens]

        return tokens, tokens_ids


def instadeep_preprocessing(
    data: pd.DataFrame,
    model_name: str = "500M_1000G",
    batch_size: int = 25,
    embeddings_layer_to_save: int = 20,
    out_fpath: str = "",
) -> (List[np.array], List[List[int]]):
    # Get pretrained model
    parameters, forward_fn, tokenizer, config = get_pretrained_model(
        model_name=model_name,
        mixed_precision=False,
        embeddings_layers_to_save=(embeddings_layer_to_save,),
        max_positions=1000,
    )
    forward_fn = hk.transform(forward_fn)

    # Initialize random key
    random_key = jax.random.PRNGKey(0)

    n_samples = len(data)
    f = h5py.File(out_fpath, "w")
    dt = h5py.vlen_dtype(np.dtype("float32"))
    emb_dset = f.create_dataset("embeddings", (n_samples,), dtype=dt)

    dt = h5py.vlen_dtype(np.dtype("float32"))
    counts_dset = f.create_dataset("counts", (n_samples,), dtype=dt)

    embeddings = []
    counts = []
    pbar = trange(0, n_samples, batch_size)
    for start_idx in pbar:
        # Get batch_infos
        sequences = data.sequence[start_idx : start_idx + batch_size].values
        codon_idxs = data.codon_idx[start_idx : start_idx + batch_size].values
        seq_counts = data.norm_counts[start_idx : start_idx + batch_size].values

        # Tokenize sequence and extract token_ids
        tokens_ids = [b[1] for b in CodonTokenizer().batch_tokenize(sequences)]
        tokens_ids = jnp.asarray(tokens_ids, dtype=jnp.int32)

        # Extract embedding
        embs = forward_fn.apply(parameters, random_key, tokens_ids)[
            f"embeddings_{embeddings_layer_to_save}"
        ]
        embs = np.asarray(embs)

        # Save non-padded embeddings
        for idx in range(len(embs)):
            # Save embedding
            temp_emb = embs[idx][np.where(tokens_ids[idx] != 1)]
            emb_dset[start_idx + idx] = temp_emb.flatten()

            # Save counts
            counts = np.zeros(
                len(sequences[idx]) // 3 - 1
            )  # Last codon skipped given 6mer kernel!
            np.put(counts, codon_idxs[idx], seq_counts[idx])
            counts_dset[start_idx + idx] = counts

    return embeddings, counts
