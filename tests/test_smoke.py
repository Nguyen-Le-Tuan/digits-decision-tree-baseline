import csv
import json

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from src.train import run_experiment, save_results


def test_run_experiment(tmp_path, capsys):
    seed = 7
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_train_subset = X_train[:100]
    y_train_subset = y_train[:100]

    X_test_subset = X_test[:40]
    y_test_subset = y_test[:40]

    record = run_experiment(
        "smoke_test",
        100,
        X_train_subset,
        X_test_subset,
        y_train_subset,
        y_test_subset,
        seed,
    )

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

    assert required_keys <= record.keys()
    assert record["seed"] == seed
    assert 0.0 <= record["accuracy"] <= 1
    assert record["fit_seconds"] >= 0
    assert record["predict_seconds"] >= 0

    save_results([record], tmp_path)
    assert capsys.readouterr().out == ""
    assert (tmp_path / "results.csv").is_file()
    assert (tmp_path / "results.json").is_file()

    with (tmp_path / "results.json").open(encoding="utf-8") as file:
        json_rows = json.load(file)
    assert json_rows[0]["config_name"] == record["config_name"]

    with (tmp_path / "results.csv").open(newline="", encoding="utf-8") as file:
        csv_rows = list(csv.DictReader(file))
    assert csv_rows[0]["config_name"] == record["config_name"]
