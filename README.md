# D501 Machine Learning DevOps — FastAPI Model Deployment

This project demonstrates deployment of a machine learning pipeline using FastAPI, scikit-learn, GitHub Actions, DVC, and pytest. A classification model is trained on the census.csv dataset to predict whether an individual's income is <=50K or >50K annually based on demographic features from the UCI Census Income dataset.<Br>

The trained model is served via a REST API, which allows inference through a POST endpoint. This repository includes:

- Model training and packaging
- Reproducible data pipeline
- Unit testing and continuous integration
- Slice-based model performance evaluation
- FastAPI model serving<br>

## Getting Started

### Clone the repository

git clone https://github.com/abridgers087/Deploying-a-Scalable-ML-Pipeline-with-FastAPI.git<br>
cd Deploying-a-Scalable-ML-Pipeline-with-FastAPI<br>

### Create and activate a virtual environment

python -m venv venv<br>
.\venv\Scripts\activate # Windows<Br>

### Install dependencies

pip install -r requirements.txt<br>

### Train the model

python model/train_model.py<Br>

### Example training output:

Precision: 0.7419 | Recall: 0.6384 | F1: 0.6863<br>
Reference: screenshots/model_train.png<br>

### Slice-Based Performance

Slice-based metrics evaluate fairness and performance differences for different categorical groups (e.g., education, workclass, marital status). Results are saved to slice_output.txt.<br>
Reference: screenshots/slice_output.png

### Unit Testing

Tests validate:<Br>

- Model successfully trains and returns a model object
- Inference returns predictions of correct length
- Metric calculation returns float values<br>

Run tests: pytest -q
Reference: screenshots/unit_test.png

### Continuous Integration (GitHub Actions)

Pipeline automatically runs on push and validates:<br>

- Environment setup
- flake8 linting
- pytest execution<br>

All checks must pass before merge.
Reference: screenshots/continuous_int.png

### FastAPI Model Inference

Start the API:<Br>

- uvicorn main:app --reload<br>

Open Swagger docs in a browser: http://127.0.0.1:8000/docs<br>

Example inference payload:
{
"age": 37,
"workclass": "Private",
"education": "Masters",
"marital-status": "Married-civ-spouse",
"occupation": "Exec-managerial",
"relationship": "Husband",
"race": "White",
"sex": "Male",
"hours-per-week": 40,
"native-country": "United-States"
}<br>

Example response:
{ "result": ">50K" }<br>

RScreenshot: screenshots/local_api.png
