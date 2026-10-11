"""Entrenar y registrar el baseline Ridge del dataset de Walt Disney World."""
import argparse
import hashlib
import json
import random
import re
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_PRIORITY = ["AVG_WAIT_MIN", "AVERAGE_WAIT_MIN", "MEAN_WAIT_MIN", "AVG_WAIT", "AVERAGE_WAIT", "MEAN_WAIT", "MEDIAN_WAIT", "CROWD_INDEX", "CROWD_SCORE"]


def parse_args():
    parser = argparse.ArgumentParser(description="Experimentos reproducibles con Ridge y MLflow para Walt Disney World")
    parser.add_argument("--alpha", type=float, default=1.0, help="Intensidad de regularización L2 (por defecto: 1.0)")
    parser.add_argument("--tol", type=float, default=1e-3, help="Tolerancia de convergencia del solver lsqr (por defecto: 0.001)")
    parser.add_argument("--fit-intercept", action=argparse.BooleanOptionalAction, default=True, help="Ajustar o no intercepto (por defecto: sí)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para división de datos y generadores aleatorios")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fracción fija de prueba (por defecto: 0.2)")
    parser.add_argument("--data-path", type=str, default=None, help="Ruta opcional a calendar_enriched.csv; por defecto descarga desde Kaggle")
    parser.add_argument("--tracking-uri", type=str, default="sqlite:///mlflow.db", help="URI de la base de seguimiento MLflow")
    parser.add_argument("--experiment-name", type=str, default="WDW_Ridge_Hyperparameter_Experiments", help="Nombre común de experimento")
    args = parser.parse_args()
    if args.alpha < 0 or args.tol <= 0 or not 0 < args.test_size < 1:
        parser.error("alpha debe ser >= 0, tol > 0 y test-size entre 0 y 1")
    return args


def load_data(data_path):
    if data_path is None:
        import kagglehub
        dataset_dir = Path(kagglehub.dataset_download("chadwambles/walt-disney-world-wait-times"))
        data_file = dataset_dir / "calendar_enriched.csv"
    else:
        data_file = Path(data_path).expanduser().resolve()
    if not data_file.is_file():
        raise FileNotFoundError(f"No existe calendar_enriched.csv en: {data_file}")
    df = pd.read_csv(data_file)
    df.columns = [re.sub(r"[^A-Z0-9]+", "_", str(name).strip().upper()).strip("_") for name in df.columns]
    data_hash = hashlib.sha256(data_file.read_bytes()).hexdigest()
    return df, data_file, data_hash


def choose_target(df):
    numerics = df.select_dtypes(include=np.number).columns.tolist()
    target = next((name for name in TARGET_PRIORITY if name in numerics), None)
    if target is None:
        target = next((name for name in numerics if ("WAIT" in name or "CROWD" in name) and name not in {"YEAR", "MONTH", "DAY", "PARK_ID"}), None)
    if target is None:
        raise ValueError("No se detectó una columna objetivo numérica WAIT/CROWD - Revisa las columnas del CSV")
    return target


def prepare_data(df, target):
    # replicar las decisiones de limpieza del notebook original
    clean = df.drop_duplicates().copy()
    if "DATE" in clean.columns:
        clean["DATE"] = pd.to_datetime(clean["DATE"], errors="coerce")
        clean["DATE_YEAR"] = clean["DATE"].dt.year
        clean["DATE_MONTH"] = clean["DATE"].dt.month
        clean["DATE_DAY_OF_WEEK"] = clean["DATE"].dt.dayofweek
        clean["DATE_DAY_OF_YEAR"] = clean["DATE"].dt.dayofyear
        clean["DATE_IS_WEEKEND"] = (clean["DATE_DAY_OF_WEEK"] >= 5).astype(int)
    clean[target] = pd.to_numeric(clean[target], errors="coerce")
    clean = clean.dropna(subset=[target]).copy()
    excluded = {target, "DATE"}
    excluded.update(col for col in clean.columns if col.endswith("_ID") or col == "ID")
    if "WAIT" in target:
        excluded.update(col for col in clean.columns if "WAIT" in col and col != target)
    if "CROWD" in target:
        excluded.update(col for col in clean.columns if "CROWD" in col and col != target)
    candidates = [col for col in clean.columns if col not in excluded]
    numeric = [col for col in candidates if pd.api.types.is_numeric_dtype(clean[col])]
    categorical = [col for col in candidates if clean[col].dtype == "object" and 1 < clean[col].nunique(dropna=True) <= 30]
    constant = [col for col in numeric + categorical if clean[col].nunique(dropna=True) <= 1]
    numeric = [col for col in numeric if col not in constant]
    categorical = [col for col in categorical if col not in constant]
    features = numeric + categorical
    if not features:
        raise ValueError("No se encontraron predictores utilizables en el archivo CSV")
    return clean[features].copy(), clean[target].copy(), numeric, categorical

def build_model(numeric, categorical, alpha, tol, fit_intercept):
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical))
    return Pipeline([
        ("preprocessor", ColumnTransformer(transformers=transformers)),
        ("model", Ridge(alpha=alpha, tol=tol, fit_intercept=fit_intercept, solver="lsqr"))
    ])


def save_artifacts(folder, actual, predicted, metrics, args, target, data_hash):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metrics.json").write_text(json.dumps({"metrics": metrics, "target": target, "data_sha256": data_hash, "configuration": vars(args)}, indent=2, ensure_ascii=False), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(actual, predicted, alpha=0.35)
    low = float(min(np.min(actual), np.min(predicted)))
    high = float(max(np.max(actual), np.max(predicted)))
    ax.plot([low, high], [low, high], "--", color="gray")
    ax.set(xlabel="Valor real", ylabel="Predicción", title=f"Valores reales vs. predicciones — {target}")
    fig.tight_layout()
    fig.savefig(folder / "real_vs_predicho.png", dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    df, data_file, data_hash = load_data(args.data_path)
    target = choose_target(df)
    features, labels, numeric, categorical = prepare_data(df, target)
    train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=args.test_size, random_state=args.seed)
    model = build_model(numeric, categorical, args.alpha, args.tol, args.fit_intercept)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=f"ridge_a{args.alpha}_tol{args.tol}_int{int(args.fit_intercept)}") as run:
        model.fit(train_x, train_y)
        predicted = model.predict(test_x)
        metrics = {
            "mae": float(mean_absolute_error(test_y, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(test_y, predicted))),
            "r2": float(r2_score(test_y, predicted)),
        }
        for key, value in {"alpha": args.alpha, "tol": args.tol, "fit_intercept": args.fit_intercept, "seed": args.seed, "test_size": args.test_size, "model": "Ridge", "solver": "lsqr", "target": target, "dataset_file": data_file.name, "data_sha256": data_hash, "training_rows": len(train_x), "test_rows": len(test_x), "num_features": features.shape[1]}.items():
            mlflow.log_param(key, value)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir)
            save_artifacts(artifact_dir, test_y.to_numpy(), predicted, metrics, args, target, data_hash)
            mlflow.log_artifacts(str(artifact_dir), artifact_path="evaluation")
        mlflow.sklearn.log_model(sk_model=model, name="ridge_model")
        print(f"Run ID: {run.info.run_id} | Target: {target} | MAE: {metrics['mae']:.3f} | RMSE: {metrics['rmse']:.3f} | R²: {metrics['r2']:.3f}")


if __name__ == "__main__":
    main()
