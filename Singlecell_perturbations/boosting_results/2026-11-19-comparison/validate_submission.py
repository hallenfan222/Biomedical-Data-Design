"""Check a prediction CSV against the official competition sample_submission.csv."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def header(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return next(csv.reader(handle))


def validate(template, prediction):
    columns = header(template)
    if columns != header(prediction) or columns[0] != 'id' or len(set(columns)) != 18212 or len(columns) != 18212:
        raise ValueError('ID column or ordered gene columns differ from the expected template')
    expected = pd.read_csv(template, usecols=['id'], dtype={'id': str})['id'].tolist()
    if len(expected) != 255 or len(set(expected)) != 255:
        raise ValueError('Expected 255 unique official IDs')
    ids = []
    for batch in pd.read_csv(prediction, chunksize=32, dtype={'id': str}):
        ids.extend(batch.pop('id').tolist())
        if not np.isfinite(batch.to_numpy(dtype=np.float64)).all():
            raise ValueError('Missing or non-finite gene prediction')
    if ids != expected:
        raise ValueError('Prediction row count, IDs or order differ from the official template')
    return dict(file=prediction.name, rows=len(ids), genes=len(columns)-1,
                sha256=hashlib.sha256(prediction.read_bytes()).hexdigest(), validated=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('template', type=Path)
    parser.add_argument('predictions', type=Path, nargs='+')
    args = parser.parse_args()
    print(json.dumps([validate(args.template, p) for p in args.predictions], indent=2))
