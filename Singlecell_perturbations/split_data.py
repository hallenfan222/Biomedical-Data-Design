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

## validate splits for training and validation
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
        cell_pairs = {
            pair for pair in all_pairs
            if pair[0] == cell_type
        }
        val_pairs = {
            pair for pair in validation
            if pair[0] == cell_type
        }

        expected = max(1, int(np.ceil(len(cell_pairs) * fraction)))

        if len(val_pairs) != expected:
            raise ValueError(
                f"Incorrect validation size for {cell_type}."
            )

        if len(cell_pairs - val_pairs) == 0:
            raise ValueError(
                f"No training pairs remain for {cell_type}."
            )
        
## create splits for training and validation, 
def make_split(data, validation_fraction=0.2, seed=42):
    """Split data randomly, 80% for training data, and 20% for validation data"""
    pairs = _pairs(data)
    rng = np.random.default_rng(seed)

    train_pairs = []
    validation_pairs = []

    cell_types = sorted({cell_type for cell_type, _ in pairs})

    for cell_type in cell_types:
        cell_pairs = [pair for pair in pairs if pair[0] == cell_type]

        n_validation = max(1, int(np.ceil(len(cell_pairs) * validation_fraction)))
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


def load_or_create_split(data, path, validation_fraction=0.2, seed=42,
                         export_format="parquet"):
    """Reuse/create JSON and export full rows beside it (default: parquet).

    Use export_format="csv" for CSV, or None for JSON only. Exports include
    metadata and all genes, in manifest order, without an extra index column.
    Repeated calls refresh the exported files from the same saved split.
    """
    if export_format not in ("parquet", "csv", None):
        raise ValueError("export_format must be 'parquet', 'csv', or None.")
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
        if export_format is not None:
            export_split(data, manifest, path, export_format)
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

    if export_format is not None:
        export_split(data, manifest, path, export_format)
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


def export_split(data, manifest, manifest_path, file_format="parquet"):
    """Export full split data; return paths keyed by 'train' and 'validation'.

    Example: split.json -> split_train.parquet, split_validation.parquet.
    CSV does not preserve pandas dtypes; use parquet for lossless round trips.
    """
    if file_format not in ("parquet", "csv"):
        raise ValueError("file_format must be 'parquet' or 'csv'.")
    train_df, val_df = apply_split(data, manifest)
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, frame in (("train", train_df), ("validation", val_df)):
        output = manifest_path.with_name(f"{manifest_path.stem}_{name}.{file_format}")
        if file_format == "parquet":
            frame.to_parquet(output, index=False)
        else:
            frame.to_csv(output, index=False)
        paths[name] = output
    return paths


def get_gene_columns(data):
    """Identify numeric DE targets, excluding the five competition metadata fields."""
    genes = [column for column in data.columns if column not in METADATA_COLUMNS]
    if not genes or any(not pd.api.types.is_numeric_dtype(data[g]) for g in genes):
        raise ValueError("Expected numeric gene targets after removing metadata.")
    if not np.isfinite(data[genes].to_numpy()).all():
        raise ValueError("DE targets contain missing or infinite values.")
    return genes
