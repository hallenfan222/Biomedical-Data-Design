"""Reproducible pseudo-target splits for cell-type/drug DE prediction.

Save pair identities as JSON and export matching training/validation data files.
Dependencies: numpy, pandas, and pyarrow (for reading the source parquet file).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PAIR_COLUMNS = ["cell_type", "sm_name"]
METADATA_COLUMNS = ["cell_type", "sm_name", "sm_lincs_id", "SMILES", "control"]

## make sure if the required pair columns are present and valid
def _pairs(data):
    missing = set(PAIR_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Missing pair columns: {sorted(missing)}")
    pairs = data[PAIR_COLUMNS]
    if pairs.isna().any().any():
        raise ValueError("Pair identifiers must not contain missing values.")
    if not all(isinstance(x, str) for x in pairs.to_numpy().ravel()):
        raise ValueError("Pair identifiers must be strings.")
    if pairs.duplicated().any():
        raise ValueError("Expected one DE row per (cell_type, sm_name) pair.")
    return sorted(pairs.itertuples(index=False, name=None))

# Sparse-cell folds following XGBoost.ipynb.
SOURCE_CELLS = ["NK cells", "T cells CD4+", "T cells CD8+", "T regulatory cells"]

def make_sparse_splits(data):
    _pairs(data)
    anchors_b = set(data.loc[data.cell_type.eq("B cells") & ~data.control, "sm_name"])
    anchors_m = set(data.loc[data.cell_type.eq("Myeloid cells") & ~data.control, "sm_name"])
    assert anchors_b == anchors_m and len(anchors_b) == 15
    folds = {}
    for target_cell in SOURCE_CELLS:
        is_target = data.cell_type.eq(target_cell)
        held_out = is_target & ~data.control & ~data.sm_name.isin(anchors_b)
        train = data.loc[~held_out]
        validation = data.loc[held_out]
        target_anchors = data.loc[is_target & ~data.control & data.sm_name.isin(anchors_b), "sm_name"]
        assert len(target_anchors) in (13, 15)
        assert len(validation) > 100 and not validation.control.any()
        assert set(validation.sm_name).issubset(set(train.sm_name))
        assert len(train) + len(validation) == len(data)
        assert len(train.loc[train.cell_type.eq(target_cell) & ~train.control]) == len(target_anchors)
        folds[target_cell] = {
            "target_cell": target_cell,
            "anchor_drugs": sorted(anchors_b),
            "train_pairs": train[PAIR_COLUMNS].values.tolist(),
            "validation_pairs": validation[PAIR_COLUMNS].values.tolist(),
        }
    return folds

def save_sparse_splits(data, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    folds = make_sparse_splits(data)
    for target_cell, manifest in folds.items():
        filename = {"NK cells": "nk", "T cells CD4+": "cd4", "T cells CD8+": "cd8", "T regulatory cells": "treg"}[target_cell]
        (folder / f"sparse_cell_{filename}.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return folds

def apply_split(data, manifest):
    positions = {pair: i for i, pair in enumerate(data[PAIR_COLUMNS].itertuples(index=False, name=None))}
    return tuple(data.iloc[[positions[tuple(pair)] for pair in manifest[key]]].copy().reset_index(drop=True)
                 for key in ("train_pairs", "validation_pairs"))


def get_gene_columns(data):
    """Identify numeric DE targets, excluding the five competition metadata fields."""
    genes = [column for column in data.columns if column not in METADATA_COLUMNS]
    if not genes or any(not pd.api.types.is_numeric_dtype(data[g]) for g in genes):
        raise ValueError("Expected numeric gene targets after removing metadata.")
    if not np.isfinite(data[genes].to_numpy()).all():
        raise ValueError("DE targets contain missing or infinite values.")
    return genes
