# D501 Machine Learning DevOps — FastAPI Model Deployment

This project demonstrates how to serve a trained machine learning model using a RESTful API built with FastAPI. The model is trained locally and loaded for inference through a `POST /predict` endpoint. This repository includes reproducible environment setup, model packaging, version control structure, and basic API-based model inference.

---

## Getting Started

### Clone the repository

git clone https://github.com/abridgers087/Deploying-a-Scalable-ML-Pipeline-with-FastAPI.git
cd Deploying-a-Scalable-ML-Pipeline-with-FastAPI

### Create and activate a virtual environment

python -m venv venv
.\venv\Scripts\activate # Windows

### Install dependencies

pip install -r requirements.txt

### Train the model

python model/train_model.py

### Launch the API

uvicorn app.main:app --reload

---

## Data Versioning

This project uses DVC (Data Version Control) to track the `census.csv` dataset without storing the raw data directly in GitHub. The dataset is added to DVC and referenced via a `.dvc` tracking file so that the data can be reproduced reliably and consistently.

Commands used:

- `dvc init`
- `dvc add data/census.csv`
- `git add data/census.csv.dvc .gitignore`
- `git commit -m "Track census data with DVC"`

The raw dataset remains local, while the tracking file is committed and version-controlled.

Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.

# Environment Set up (pip or conda)

- Option 1: use the supplied file `environment.yml` to create a new environment with conda
- Option 2: use the supplied file `requirements.txt` to create a new environment with pip

## Repositories

- Create a directory for the project and initialize git.
  - As you work on the code, continually commit changes. Trained models you want to use in production must be committed to GitHub.
- Connect your local git repo to GitHub.
- Setup GitHub Actions on your repo. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
  - Make sure you set up the GitHub Action to have the same version of Python as you used in development.

# Data

- Download census.csv and commit it to dvc.
- This data is messy, try to open it in pandas and see what you get.
- To clean it, use your favorite text editor to remove all spaces.

# Model

- Using the starter code, write a machine learning model that trains on the clean data and saves the model. Complete any function that has been started.
- Write unit tests for at least 3 functions in the model code.
- Write a function that outputs the performance of the model on slices of the data.
  - Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.
- Write a model card using the provided template.

# API Creation

- Create a RESTful API using FastAPI this must implement:
  - GET on the root giving a welcome message.
  - POST that does model inference.
