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
def make_split(data, validation_fraction=0.2, seed=42):
    """随机拆分每种 cell type 的组合，约 80% 训练、20% 验证。"""
    pairs = _pairs(data)
    rng = np.random.default_rng(seed)

    train_pairs = []
    validation_pairs = []

    cell_types = sorted({cell_type for cell_type, _ in pairs})

    for cell_type in cell_types:
        cell_pairs = [pair for pair in pairs if pair[0] == cell_type]

        n_validation = max(1, round(len(cell_pairs) * validation_fraction))
        if n_validation >= len(cell_pairs):
            raise ValueError(
                f"Not enough pairs for cell type {cell_type!r} "
                "to create both training and validation data."
            )

        selected = rng.choice(len(cell_pairs), size=n_validation, replace=False)
        validation_indices = set(selected)

        for i, pair in enumerate(cell_pairs):
            if i in validation_indices:
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
    """读取已有划分；如果文件不存在，就生成并保存一份。"""
    path = Path(path)

    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))

        if manifest.get("validation_fraction") != validation_fraction:
            raise ValueError(
                "Saved split uses a different validation fraction. "
                "Use a new manifest path."
            )

        if manifest.get("seed") != seed:
            raise ValueError(
                "Saved split uses a different seed. "
                "Use a new manifest path."
            )

        _validate_split(data, manifest)
        return manifest

    manifest = make_split(
        data,
        validation_fraction=validation_fraction,
        seed=seed,
    )

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
