import pandas as pd
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
from train_model import cat_features


# Create synthetic sample dataset for testing
def sample_dataframe():
    return pd.DataFrame({
        "age": [37],
        "workclass": ["Private"],
        "fnlgt": [178356],
        "education": ["HS-grad"],
        "education-num": [10],
        "marital-status": ["Married-civ-spouse"],
        "occupation": ["Prof-specialty"],
        "relationship": ["Husband"],
        "race": ["White"],
        "sex": ["Male"],
        "capital-gain": [0],
        "capital-loss": [0],
        "hours-per-week": [40],
        "native-country": ["United-States"],
        "salary": ["<=50K"]
    })


# TEST 1: Model training returns valid model object
def test_train_model_returns_model():
    df = sample_dataframe()
    X, y, encoder, lb = process_data(
        df,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    model = train_model(X, y)
    assert hasattr(model, "predict")


# TEST 2: Inference output shape matches input rows
def test_inference_output_length():
    df = sample_dataframe()
    X, y, encoder, lb = process_data(
        df,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    model = train_model(X, y)
    preds = inference(model, X)
    assert len(preds) == len(X)


# TEST 3: compute_model_metrics returns float values
def test_compute_model_metrics_returns_values():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
