from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from time import perf_counter
import pandas as pd
import csv
import json
from pathlib import Path


def run_experiment(config_name, max_depth_size,  X_train, X_test, y_train, y_test):
        model = DecisionTreeClassifier(max_depth=max_depth_size, random_state=42)
    
        start = perf_counter()
    
        # Training stage
        model.fit(X_train, y_train)
    
        fit_seconds = perf_counter() - start
    
        start_pred = perf_counter()
    
        y_pred = model.predict(X_test)
    
        predict_seconds = perf_counter() - start_pred
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)


        results = {
            "config_name" : config_name,
            "model" : "DecisionTreeClassifier",
            "max_depth" : max_depth_size,
            "seed" : 42,
            "accuracy" : accuracy,
            "fit_seconds" : fit_seconds,
            "predict_seconds" : predict_seconds,
            "n_train" : len(X_train),
            "n_test" : len(X_test)
        }

        return results


def save_results(all_results, output_dir):
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok = True)

    csv_path = output_dir / "results.csv"
    
    with csv_path.open("w", newline="",encoding="utf-8") as file:
        writer = csv.DictWriter(file,fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)

    json_path = output_dir / "results.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(all_results, file, indent = 2)

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        csv_rows = list(csv.DictReader(file))

    with json_path.open("r", encoding="utf-8") as file:
        json_rows = json.load(file)

    print(len(csv_rows), len(json_rows))


     
def main():
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print("First five label: ", y[:5])

    print("Start Experiment")
    
    kq1 = run_experiment("tree_depth_5", 5, X_train, X_test, y_train, y_test)
    kq2 = run_experiment("tree_depth_10", 10, X_train, X_test, y_train, y_test)

    all_results = [kq1, kq2]

    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    

    save_results([kq1, kq2], results_dir)

if __name__ == "__main__":
    main()