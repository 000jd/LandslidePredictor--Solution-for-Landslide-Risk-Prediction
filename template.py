import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "LandslidePredictor"

list_of_files = [
    # Config files for parameters and paths
    "config/config.yaml",
    "config/params.yaml",
    
    # Data folders
    "data/raw/",
    "data/processed/",
    "data/interim/",

    # Notebooks for EDA and experiments
    "notebooks/EDA.ipynb",

    # Source code for ML models and pipelines
    "src/data/load_data.py",
    "src/data/process_data.py",
    "src/data/split_data.py",
    "src/models/train_model.py",
    "src/models/evaluate_model.py",
    "src/models/predict.py",
    "src/utils/helpers.py",

    # Unit tests
    "tests/test_data.py",
    "tests/test_model.py",
    "tests/test_utils.py",

    # DVC configuration for data version control
    "dvc.yaml",

    # Dockerfile for containerization
    "Dockerfile",

    # CI/CD pipeline configuration for GitLab (can be modified for other CI tools)
    ".gitlab-ci.yml",

    # Python dependencies
    "requirements.txt",

    # Project description and documentation
    "README.md",

    # Python package setup
    "setup.py",

    # Makefile for build automation
    "Makefile",

    # Git ignore file
    ".gitignore",

    # Logs folder for logging outputs
    "logs/"
]

# Loop to create directories and files as defined above
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
