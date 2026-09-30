import os
from getpass import getpass
import kagglehub

os.environ["KAGGLE_API_TOKEN"] = getpass("Paste Kaggle API token: ").strip()

path = kagglehub.competition_download(
    "open-problems-single-cell-perturbations"
)

print("Path to competition files:", path)