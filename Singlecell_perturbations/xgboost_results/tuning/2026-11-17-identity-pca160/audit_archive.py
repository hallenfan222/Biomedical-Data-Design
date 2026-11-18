"""Audit saved archive evidence without data access, GPU use, or training."""
import ast
import csv
import hashlib
import json
import math
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
FOLDS = ["NK", "CD4", "CD8", "Treg"]


def csv_rows(name):
    with (ROOT / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12), (a, b)


def check_link(source, target):
    url = urlsplit(target)
    if url.scheme in {"http", "https", "mailto", "data"}:
        return
    assert not url.scheme and not url.netloc, (source, target)
    if url.path:
        assert not url.path.startswith("/"), (source, target)
        assert (source.parent / unquote(url.path)).exists(), (source, target)


class Links(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.source = source

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in {"href", "src", "xlink:href"} and value:
                check_link(self.source, value)
                if tag in {"script", "img", "link"}:
                    assert not urlsplit(value).netloc, (self.source, value)


def main():
    manifest = json.loads((ROOT / "ARCHIVE_SHA256.json").read_text())
    for name, expected in manifest.items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        assert actual == expected, (name, "changed original file")
    configs = json.loads((ROOT / "configs.json").read_text())
    rows = csv_rows("reports/results.csv")
    folds = csv_rows("reports/fold_results.csv")
    assert len(rows) == len(configs) == 15 and len(folds) == 60
    assert [r["trial_id"] for r in rows] == [f"T{i:02d}" for i in range(15)]
    assert len({(r["trial_id"], r["fold"]) for r in folds}) == 60
    saved = {r["trial_id"]: json.loads((ROOT / "results" / (r["trial_id"] + ".json")).read_text()) for r in rows}
    base = saved["T00"]
    for config, row in zip(configs, rows):
        trial = saved[row["trial_id"]]
        assert trial["status"] == row["status"] == "complete"
        assert trial["params"] == {**base["params"], **config.get("overrides", {})}
        for key, value in trial["params"].items():
            if isinstance(value, (int, float)):
                close(row[key], value)
            else:
                assert row[key] == value
        close(row["mean_mrrmse"], trial["mean_mrrmse"])
        close(trial["mean_mrrmse"], sum(f["mrrmse"] for f in trial["folds"]) / 4)
        close(row["model_seconds"], trial["model_total_seconds"])
        close(row["improvement_pct"], 100 * (base["mean_mrrmse"] - trial["mean_mrrmse"]) / base["mean_mrrmse"])
        assert [f["fold"] for f in trial["folds"]] == FOLDS
        job = trial["context"]["job_id"]
        assert row["slurm_job_id"] == job
        excerpt = (ROOT / "logs" / (job + ".excerpt.log")).read_text()
        logged = [json.loads(line.split(" ", 2)[2]) for line in excerpt.splitlines() if line.startswith("FOLD " + trial["id"] + " ")]
        assert logged == trial["folds"]
        assert trial["context"]["runner_sha256"] == manifest["runner.py"]
        assert trial["context"]["split_sha256"] == manifest["source/split_data.py"]
        for fold in trial["folds"]:
            close(row[fold["fold"] + "_mrrmse"], fold["mrrmse"])
            long = next(f for f in folds if f["trial_id"] == trial["id"] and f["fold"] == fold["fold"])
            close(long["mrrmse"], fold["mrrmse"])
            assert long["job_id"] == job and fold["device"].startswith("cuda")
    ranks = sorted(saved, key=lambda key: saved[key]["mean_mrrmse"])
    assert ranks[:3] == ["T09", "T08", "T02"]
    close(base["mean_mrrmse"], 1.0320132131701798)
    close(saved["T09"]["mean_mrrmse"], 1.00740161673428)
    for row in csv_rows("reports/controlled_contrasts.csv"):
        a, b = saved[row["reference"]], saved[row["trial"]]
        actual = {k: [a["params"][k], b["params"][k]] for k in a["params"] if a["params"][k] != b["params"][k]}
        assert json.loads(row["changed_parameters"]) == actual
        close(row["mean_delta"], b["mean_mrrmse"] - a["mean_mrrmse"])
    for path in ROOT.rglob("*.md"):
        for target in re.findall(r"\]\(([^\s)]+)\)", path.read_text()):
            check_link(path, target)
    Links(ROOT / "reports/BDD_XGBoost_Tuning_Report.html").feed((ROOT / "reports/BDD_XGBoost_Tuning_Report.html").read_text())
    for path in (ROOT / "reports").glob("*.svg"):
        ET.parse(path)
    for path in ROOT.rglob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    print(json.dumps({"status": "PASS", "unchanged_original_files": len(manifest), "configurations": len(rows), "validation_fits": len(folds), "top_three": ranks[:3], "csv_json_log_agreement": True, "source_hashes": "match actual run", "local_links_svg_python": "valid", "training_rerun": False}, indent=2))


if __name__ == "__main__":
    main()
