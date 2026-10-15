# ============================================================
# VehicleVision - Best-known model (reported Kaggle score 0.90600)
# 70% Logistic Regression + 30% CatBoost depth 7
# CatBoost noise augmentation: 2.5%
# Same blend weights and noise method as the best 0.90400 model;
# only CatBoost tree depth / associated depth-7 hyperparameters change.
# ============================================================

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

ID = "vehicle_id"
TARGET = "vehicle_class"
OUTPUT_NAME = "vehicle_submission_lr70_cat30_depth7_noise025.csv"


def log(message):
    print(message, flush=True)


def locate_file(filenames):
    if isinstance(filenames, str):
        filenames = [filenames]
    roots = [
        Path("/kaggle/input/competitions/vehicle-type-classification"),
        Path("/kaggle/input"), Path.cwd(), Path("/kaggle/working"),
        Path("/mnt/data"), Path("/content")
    ]
    checked = set()
    for root in roots:
        if not root.is_dir():
            continue
        root_id = str(root.resolve())
        if root_id in checked:
            continue
        checked.add(root_id)
        for filename in filenames:
            direct = root / filename
            if direct.is_file():
                log(f"Found {filename}: {direct}")
                return direct
        for filename in filenames:
            try:
                for candidate in root.rglob(filename):
                    if candidate.is_file():
                        log(f"Found {filename}: {candidate}")
                        return candidate
            except (PermissionError, OSError):
                pass
    raise FileNotFoundError(
        "Could not find any of: " + ", ".join(filenames) +
        ". In Kaggle, attach the competition data via Add Input."
    )


def align_probs(probabilities, classes):
    aligned = np.zeros((len(probabilities), 3), dtype=float)
    for j, cls in enumerate(classes):
        aligned[:, int(cls)] = probabilities[:, j]
    return aligned


def main():
    started = time.time()
    log("VehicleVision best-known model: 70% Logistic Regression + 30% CatBoost depth 7")
    log("STEP 1/7: Finding inputs...")
    train_path = locate_file(["vehicle_train_10000(1).csv", "vehicle_train_10000.csv", "vehicle_train.csv"])
    test_path = locate_file(["vehicle_test(1).csv", "vehicle_test.csv", "vehicle_public_test.csv"])
    sample_path = locate_file(["vehicle_sample_submission(1).csv", "vehicle_sample_submission.csv"])

    log("STEP 2/7: Reading CSV files...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    if ID not in train.columns or TARGET not in train.columns:
        raise ValueError("Training file must contain vehicle_id and vehicle_class.")
    features = [c for c in train.columns if c not in [ID, TARGET]]
    expected_test_columns = set(features) | {ID}
    if set(test.columns) != expected_test_columns:
        raise ValueError(
            f"Test columns mismatch. Missing: {expected_test_columns-set(test.columns)}; "
            f"extra: {set(test.columns)-expected_test_columns}"
        )
    if list(sample.columns) != [ID, TARGET]:
        raise ValueError("Sample submission must have columns vehicle_id, vehicle_class.")
    if not np.array_equal(test[ID].to_numpy(), sample[ID].to_numpy()):
        raise ValueError("Test IDs/order differ from the sample submission.")
    if train[TARGET].isna().any():
        raise ValueError("Training target has missing values.")
    for col in features:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")

    log("STEP 3/7: Checking duplicates and preparing features...")
    if train.groupby(features, dropna=False)[TARGET].nunique().max() > 1:
        raise ValueError("Identical measurement profiles have conflicting labels.")
    # Match the preprocessing used for the successful depth-6 blend.
    train = train.drop_duplicates(subset=features).reset_index(drop=True)
    X = train[features].copy()
    y = train[TARGET].astype(int).to_numpy()
    X_test = test[features].copy()
    medians = X.median()
    X = X.fillna(medians)
    X_test = X_test.fillna(medians)
    if not set(np.unique(y)).issubset({0, 1, 2}):
        raise ValueError("Expected labels 0, 1 and 2.")
    log(f"Unique training profiles: {len(X)}; test rows: {len(X_test)}")

    log("STEP 4/7: Training Logistic Regression...")
    lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=42))
    ])
    lr.fit(X, y)
    p_lr = align_probs(lr.predict_proba(X_test), lr.named_steps["clf"].classes_)
    log("Logistic Regression complete.")

    log("STEP 5/7: Creating 2.5% noise-augmented CatBoost training rows...")
    sd = X.std().fillna(0).to_numpy(dtype=float)  # pandas sample standard deviation, matching previous generator
    values = X.to_numpy(dtype=float).copy()
    rng = np.random.default_rng(17777)
    values += rng.normal(0.0, 0.025 * sd, size=values.shape)
    cols = list(X.columns)
    for j, col in enumerate(cols):
        if col == "engine_cylinders":
            values[:, j] = X[col].to_numpy(dtype=float)
        elif col != "drag_coefficient":
            values[:, j] = np.maximum(values[:, j], 0.0)
    X_aug = pd.DataFrame(values, columns=cols)
    X_fit = pd.concat([X, X_aug], ignore_index=True)
    y_fit = np.concatenate([y, y])
    log(f"Augmented CatBoost training rows: {len(X_fit)}")

    log("STEP 6/7: Training CatBoost depth 7 (650 iterations)...")
    cat = CatBoostClassifier(
        iterations=650,
        depth=7,
        learning_rate=0.035,
        l2_leaf_reg=5.0,
        random_strength=1.0,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        random_seed=42,
        verbose=100,
        thread_count=2,
        allow_writing_files=False
    )
    cat.fit(X_fit, y_fit)
    p_cat = align_probs(cat.predict_proba(X_test), cat.classes_)
    log("CatBoost depth-7 complete.")

    log("STEP 7/7: Blending and saving submission...")
    p_blend = 0.70 * p_lr + 0.30 * p_cat
    predictions = p_blend.argmax(axis=1).astype(int)
    submission = pd.DataFrame({ID: test[ID].to_numpy(), TARGET: predictions})
    assert list(submission.columns) == list(sample.columns)
    assert len(submission) == len(sample)
    assert np.array_equal(submission[ID].to_numpy(), sample[ID].to_numpy())
    assert set(submission[TARGET].unique()).issubset({0, 1, 2})
    outdir = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
    outpath = outdir / OUTPUT_NAME
    submission.to_csv(outpath, index=False)
    log("=" * 60)
    log("SUCCESS: depth-7 candidate generated")
    log(f"Path: {outpath.resolve()}")
    log(f"Rows: {len(submission)}")
    log(f"Class counts: {submission[TARGET].value_counts().sort_index().to_dict()}")
    log(f"Elapsed minutes: {(time.time()-started)/60:.2f}")
    log("The matching submission was reported by the user at Kaggle score 0.90600.")
    log("The Kaggle score is evaluated by the competition and is not calculated locally.")
    log("=" * 60)


if __name__ == "__main__":
    main()
