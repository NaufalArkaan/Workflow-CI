"""
MLProject Training Script with CLI Arguments & MLflow Autolog
Project: Dicoding Final Project - Sistem Machine Learning MLOps (Kriteria 3 MLProject)
Author: NaufalArkaan
File: MLProject/modelling.py
"""

import os
import sys
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
import mlflow
import mlflow.sklearn


def parse_args():
    parser = argparse.ArgumentParser(description="MLProject Model Training")
    parser.add_argument("--data_path", type=str, default="bank-full_preprocessing.csv", help="Path ke processed dataset")
    parser.add_argument("--n_estimators", type=int, default=100, help="Jumlah pohon pada RandomForest")
    parser.add_argument("--random_state", type=int, default=42, help="Random state seed")
    return parser.parse_args()


def resolve_dataset_path(data_path: str) -> str:
    """
    Menemukan lokasi file dataset secara opsional dari working dir atau script dir.
    """
    if os.path.exists(data_path):
        return os.path.abspath(data_path)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, data_path),
        os.path.join(script_dir, "bank-full_preprocessing.csv"),
        os.path.join(os.path.dirname(script_dir), "bank-full_preprocessing.csv"),
        os.path.join(os.path.dirname(script_dir), "preprocessing", "bank-full_preprocessing.csv")
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)

    raise FileNotFoundError(f"Dataset preprocessing tidak ditemukan di: {data_path}")


def main():
    args = parse_args()

    print("=" * 60)
    print(" MLPROJECT TRAINING EXECUTION")
    print("=" * 60)
    print(f"Data Path   : {args.data_path}")
    print(f"n_estimators: {args.n_estimators}")
    print(f"random_state: {args.random_state}")

    # 1. Resolve & Load Dataset
    dataset_path = resolve_dataset_path(args.data_path)
    print(f"\n[1/5] Memuat processed dataset dari: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"      Dimensi dataset: {df.shape[0]} baris, {df.shape[1]} kolom")

    if 'y' not in df.columns:
        raise KeyError("Kolom target 'y' tidak ditemukan pada dataset preprocessing.")

    # 2. Train-Test Split
    X = df.drop(columns=['y'])
    y = df['y']

    print("[2/5] Membagi data (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=args.random_state,
        stratify=y
    )

    # 3. Setup MLflow Tracking & Autolog
    print("[3/5] Mengaktifkan MLflow Autolog...")

    # Jika tidak dipanggil via `mlflow run` (env MLFLOW_RUN_ID kosong), set experiment
    if "MLFLOW_RUN_ID" not in os.environ and mlflow.active_run() is None:
        mlflow.set_experiment("Bank_Marketing_MLProject")

    mlflow.autolog(log_models=True)

    # 4. Train Model
    print("[4/5] Melatih model RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        class_weight='balanced'
    )

    # Kelola context run secara aman (apakah active run dari mlflow run atau membuat baru)
    active_run = mlflow.active_run()
    if active_run is None and "MLFLOW_RUN_ID" not in os.environ:
        run_context = mlflow.start_run(run_name="MLProject_RandomForest_Run")
    else:
        # Panggil start_run tanpa argument untuk mengaitkan ke MLFLOW_RUN_ID aktif
        run_context = mlflow.start_run()

    with run_context as run:
        run_id = run.info.run_id
        print(f"      - MLflow Run ID: {run_id}")

        model.fit(X_train, y_train)

        # Evaluasi
        print("[5/5] Evaluasi performa model...")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)

        print("\n" + "-" * 40)
        print("METRIK PERFORMA MLPROJECT:")
        print("-" * 40)
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1-Score : {f1:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")
        print("-" * 40)

        # Log metrik tambahan
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_precision", prec)
        mlflow.log_metric("test_recall", rec)
        mlflow.log_metric("test_f1_score", f1)
        mlflow.log_metric("test_roc_auc", auc)

        # Validasi re-loadability
        model_uri = f"runs:/{run_id}/model"
        loaded_model = mlflow.sklearn.load_model(model_uri)
        sample_preds = loaded_model.predict(X_test.iloc[:5])
        print(f"      - Validasi Reload Model Berhasil! Prediksi sampel: {sample_preds}")

    print("\n" + "=" * 60)
    print(" MLPROJECT TRAINING SELESAI BERHASIL DILANJUTKAN")
    print("=" * 60)


if __name__ == "__main__":
    main()
