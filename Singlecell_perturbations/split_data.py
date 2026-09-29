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

## validate splits for training and validation
def _validate_split(data, manifest):
    all_pairs = set(_pairs(data))
    target = manifest["target_cell_type"]
    keep = manifest["n_observed"]
    groups = []
    for name in ("train_pairs", "validation_pairs"):
        records = manifest[name]
        ## validate the format and structure of the records
        if not isinstance(records, list) or not all(
            isinstance(pair, list) and len(pair) == 2
            and all(isinstance(x, str) for x in pair) for pair in records
        ):
            raise ValueError(f"Invalid {name} records.")
        group = set(map(tuple, records))
        ## validate that there are no duplicate pairs in the current group
        if len(group) != len(records):
            raise ValueError(f"Duplicate pairs in {name}.")
        groups.append(group)
    train, validation = groups
    if not train or not validation or train & validation:
        raise ValueError("Train and validation must be nonempty and disjoint.")
    if train | validation != all_pairs:
        raise ValueError("Saved split does not match the current dataset pairs.")
    if any(cell != target for cell, _ in validation):
        raise ValueError("Only the pseudo-target cell type may be held out.")
    if sum(cell == target for cell, _ in train) != keep:
        raise ValueError("Incorrect number of observed pseudo-target pairs.")
    training_drugs = {drug for _, drug in train}
    unseen = sorted({drug for _, drug in validation} - training_drugs)
    if unseen:
        raise ValueError(f"Validation contains drugs absent from training: {unseen}")

## create splits for training and validation, 
def make_split(data, target_cell_type="T cells CD4+", n_observed=17, seed=42):
    """Keep n_observed target pairs, hold out the rest, retain all other cells.

    Pair sorting makes the selection independent of dataframe row order.
    Sampling includes control rows, matching the initial all-pairs protocol.
    """
    pairs = _pairs(data)
    target_pairs = [p for p in pairs if p[0] == target_cell_type]
    if not target_pairs:
        raise ValueError(f"Unknown target cell type: {target_cell_type!r}")
    if isinstance(n_observed, bool) or not isinstance(n_observed, int):
        raise ValueError("n_observed must be an integer.")
    if not 1 <= n_observed < len(target_pairs):
        raise ValueError(f"n_observed must be between 1 and {len(target_pairs) - 1}.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    selected = np.random.default_rng(seed).choice(
        len(target_pairs), size=n_observed, replace=False
    )
    observed = {target_pairs[i] for i in selected}
    validation = set(target_pairs) - observed
    manifest = {
        "version": 1,
        "target_cell_type": target_cell_type,
        "n_observed": n_observed,
        "seed": seed,
        "train_pairs": [list(p) for p in pairs if p not in validation],
        "validation_pairs": [list(p) for p in pairs if p in validation],
    }
    _validate_split(data, manifest)
    return manifest


def load_or_create_split(data, path, target_cell_type="T cells CD4+", n_observed=17, seed=42):
    """Reuse saved pair IDs; reject changed settings/data instead of overwriting.

    Choose a different path when intentionally creating another experiment.
    """
    path = Path(path)
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected = (target_cell_type, n_observed, seed)
        actual = tuple(manifest.get(k) for k in ("target_cell_type", "n_observed", "seed"))
        if actual != expected:
            raise ValueError("Saved split settings differ. Use a new manifest path.")
        _validate_split(data, manifest)
        return manifest
    manifest = make_split(data, target_cell_type, n_observed, seed)
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
