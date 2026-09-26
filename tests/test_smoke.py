from src.train import run_experiment
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split



def test_run_experiment():
    X,y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,y,test_size=0.2, random_state=42, stratify = y
    )
    X_train_subset = X_train[:100]
    y_train_subset = y_train[:100]

    X_test_subset = X_test[:40]
    y_test_subset = y_test[:40]

    record = run_experiment("smoke_test", 100, X_train_subset, X_test_subset, y_train_subset, y_test_subset)

    required_keys = {
        "config_name",
        "model",
        "max_depth",
        "seed",
        "accuracy",
        "fit_seconds",
        "predict_seconds",
        "n_train",
            "n_test",
    }

    #Run to check: python -m pytest -q
    
    assert required_keys <= record.keys()
    assert 0.0 <= record["accuracy"] <= 1
    assert record["fit_seconds"] >= 0
    assert record["predict_seconds"] >= 0
    