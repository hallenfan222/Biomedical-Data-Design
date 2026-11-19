"""Create a fresh replay directory with the paths used by the Expanse scripts."""
import argparse
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('destination', type=Path)
parser.add_argument('competition_data', type=Path)
parser.add_argument('runtime', type=Path, help='Cluster environment containing activate.sh')
args = parser.parse_args()
archive = Path(__file__).resolve().parent
repo = archive.parents[2]
evidence = archive / 'provenance'
data = args.competition_data.resolve(strict=True)
runtime = args.runtime.resolve(strict=True)
for f in (data/'de_train.parquet', data/'id_map.csv', runtime/'activate.sh',
          repo/'Singlecell_perturbations/xgboost_results/baseline_expression_donor_balanced_v2.csv'):
    if not f.is_file():
        parser.error(f'Missing {f}')
root = args.destination.resolve()
root.mkdir(parents=True, exist_ok=False)
prior = root/'xgb-tuning-20261117'
run = root/'boosting-comparison-20261119'
for folder in (prior/'results', prior/'source', run/'records', run/'logs'):
    folder.mkdir(parents=True)
(prior/'data').symlink_to(data, target_is_directory=True)
(prior/'source/repo').symlink_to(repo, target_is_directory=True)
(root/'bdd-runtime').symlink_to(runtime, target_is_directory=True)
for name in ('runner.py', 'fold_manifests.json'):
    shutil.copyfile(evidence/name, prior/name)
shutil.copyfile(evidence/'split_data.py', prior/'source/split_data.py')
for name in ('T00.json', 'T09.json'):
    shutil.copyfile(evidence/name, prior/'results'/name)
for name in ('fit.py','campaign.py','finish_validation.py','batch.sh','resume.sbatch','reuse.json','validate_submission.py'):
    shutil.copyfile(archive/name, run/name)
(run/'id_map.csv').symlink_to(data/'id_map.csv')
print(run)
