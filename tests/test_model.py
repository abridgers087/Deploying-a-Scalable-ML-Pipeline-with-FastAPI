import os
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
from train_model import cat_features  # import list from script


# Resolve path to census.csv from repo root, reliable in GH Actions
def get_data_path():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "data", "census.csv")


# TEST 1
def test_train_model_returns_model():
    data = pd.read_csv(get_data_path())
    train, test = train_test_split(data, test_size=0.20, random_state=42)

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    model = train_model(X_train, y_train)

    assert hasattr(model, "predict")


# TEST 2
def test_inference_output_length():
    data = pd.read_csv(get_data_path())
    train, test = train_test_split(data, test_size=0.20, random_state=42)

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    model = train_model(X_train, y_train)

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    preds = inference(model, X_test)
    assert len(preds) == len(X_test)


# TEST 3
def test_compute_model_metrics_returns_values():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
