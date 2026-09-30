"""Reproducible pseudo-target splits for cell-type/drug DE prediction.

Only pair identities and split settings are saved; no expression data are copied.
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

## validate the saved training/validation split
def _validate_split(data, manifest):
    all_pairs = set(_pairs(data))
    train = set(map(tuple, manifest["train_pairs"]))
    validation = set(map(tuple, manifest["validation_pairs"]))

    if not train or not validation:
        raise ValueError("Train and validation must both contain pairs.")
    if train & validation:
        raise ValueError("A pair appears in both train and validation.")
    if train | validation != all_pairs:
        raise ValueError("Train and validation do not match the current data.")

    fraction = manifest["validation_fraction"]
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    for cell_type in data["cell_type"].unique():
        cell_pairs = {pair for pair in all_pairs if pair[0] == cell_type}
        val_pairs = {pair for pair in validation if pair[0] == cell_type}
        expected = max(1, int(np.ceil(len(cell_pairs) * fraction)))
        if len(val_pairs) != expected:
            raise ValueError(f"Incorrect validation size for {cell_type}.")
        if len(cell_pairs - val_pairs) == 0:
            raise ValueError(f"No training pairs remain for {cell_type}.")


## create an 80/20 split within every cell type
def make_split(data, validation_fraction=0.2, seed=42):
    """Randomly split each cell type into training and validation pairs."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")

    pairs = _pairs(data)
    rng = np.random.default_rng(seed)
    train_pairs = []
    validation_pairs = []

    for cell_type in sorted({cell for cell, _ in pairs}):
        cell_pairs = [pair for pair in pairs if pair[0] == cell_type]
        n_validation = max(1, int(np.ceil(len(cell_pairs) * validation_fraction)))
        if n_validation >= len(cell_pairs):
            raise ValueError(f"Not enough pairs to split {cell_type} into train and validation.")

        selected = set(rng.choice(len(cell_pairs), size=n_validation, replace=False))
        for i, pair in enumerate(cell_pairs):
            if i in selected:
                validation_pairs.append(pair)
            else:
                train_pairs.append(pair)

    manifest = {
        "validation_fraction": validation_fraction,
        "seed": seed,
        "train_pairs": [list(pair) for pair in train_pairs],
        "validation_pairs": [list(pair) for pair in validation_pairs],
    }
    _validate_split(data, manifest)
    return manifest


def load_or_create_split(data, path, validation_fraction=0.2, seed=42):
    """Load the saved split, or create it if the file does not exist."""
    path = Path(path)

    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("validation_fraction") != validation_fraction or manifest.get("seed") != seed:
            raise ValueError("Saved split settings differ. Use a new JSON filename.")
        _validate_split(data, manifest)
        return manifest

    manifest = make_split(data, validation_fraction, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return manifest


def apply_split(data, manifest):
    """Return train_df, val_df in saved pair order, preserving source columns.

    Pair identities, rather than row positions, also support reordered source data.
    """
    _validate_split(data, manifest)
    positions = {
        pair: i for i, pair in enumerate(
            data[PAIR_COLUMNS].itertuples(index=False, name=None)
        )
    }
    return tuple(
        data.iloc[[positions[tuple(p)] for p in manifest[key]]].copy()
        for key in ("train_pairs", "validation_pairs")
    )


def get_gene_columns(data):
    """Identify numeric DE targets, excluding the five competition metadata fields."""
    genes = [column for column in data.columns if column not in METADATA_COLUMNS]
    if not genes or any(not pd.api.types.is_numeric_dtype(data[g]) for g in genes):
        raise ValueError("Expected numeric gene targets after removing metadata.")
    if not np.isfinite(data[genes].to_numpy()).all():
        raise ValueError("DE targets contain missing or infinite values.")
    return genes
