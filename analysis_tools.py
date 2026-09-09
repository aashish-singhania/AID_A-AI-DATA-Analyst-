from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _ensure_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    return df


def _infer_problem_type_from_target(y: pd.Series) -> str:
    y_nonnull = y.dropna()
    if y_nonnull.empty:
        return "unknown"
    nunique = int(y_nonnull.nunique(dropna=True))
    if y_nonnull.dtype == "bool":
        return "binary"
    if pd.api.types.is_numeric_dtype(y_nonnull.dtype):
        if nunique <= 2:
            return "binary"
        return "regression"
    if nunique <= 2:
        return "binary"
    return "multiclass"


def _prep_xy(
    df: pd.DataFrame,
    target: str,
    features: Optional[Iterable[str]] = None,
    dropna: bool = True,
):
    _ensure_df(df)
    if not target or target not in df.columns:
        raise ValueError(f"target column not found: {target!r}")

    if features is None:
        features = [c for c in df.columns if c != target]
    else:
        features = list(features)

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"feature columns not found: {missing}")

    X = df.loc[:, features].copy()
    y = df.loc[:, target].copy()

    if dropna:
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X.loc[mask]
        y = y.loc[mask]

    if X.empty or y.empty:
        raise ValueError("No rows available after filtering NA values.")

    return X, y


def _one_hot_encode(X: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    other_cols = [c for c in X.columns if c not in numeric_cols]

    X_num = X.loc[:, numeric_cols].copy()
    if X_num.shape[1]:
        X_num = X_num.astype(float)

    if other_cols:
        X_cat = pd.get_dummies(X.loc[:, other_cols].astype("string"), dummy_na=True, drop_first=False)
        X_out = pd.concat([X_num, X_cat], axis=1)
    else:
        X_out = X_num

    if X_out.shape[1] == 0:
        raise ValueError("No usable feature columns after encoding.")
    return X_out


def logistic_regression(
    df: pd.DataFrame,
    target: str,
    features: Optional[Iterable[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 2000,
):
    """
    Binary (and multiclass) classification using scikit-learn LogisticRegression.
    Returns a dict with metrics and coefficient table.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X_raw, y_raw = _prep_xy(df, target=target, features=features)
    X = _one_hot_encode(X_raw)

    problem = _infer_problem_type_from_target(y_raw)
    if problem not in {"binary", "multiclass"}:
        raise ValueError(
            f"Target {target!r} does not look like a classification target. "
            f"Detected problem type: {problem}"
        )

    y = y_raw
    if y.dtype == "bool":
        y = y.astype(int)

    strat = y if int(pd.Series(y).nunique()) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            (
                "clf",
                LogisticRegression(
                    max_iter=max_iter,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)

    report = classification_report(y_test, y_pred, output_dict=False, zero_division=0)
    out = {
        "problem_type": problem,
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "test_accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    # AUC (binary only)
    try:
        if problem == "binary" and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
            out["roc_auc"] = float(roc_auc_score(y_test, proba))
    except Exception:
        pass

    # Coefficients
    clf = model.named_steps["clf"]
    coefs = getattr(clf, "coef_", None)
    if coefs is not None:
        coef_df = pd.DataFrame(coefs, columns=X.columns)
        if coef_df.shape[0] == 1:
            coef_df = coef_df.T.rename(columns={0: "coef"}).sort_values("coef", ascending=False)
        else:
            coef_df.index = [f"class_{i}" for i in range(coef_df.shape[0])]
        out["coefficients"] = coef_df

    return out


def linear_regression(
    df: pd.DataFrame,
    target: str,
    features: Optional[Iterable[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Linear regression with one-hot encoding and train/test evaluation.
    Returns a dict with metrics and coefficient table.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    X_raw, y_raw = _prep_xy(df, target=target, features=features)
    if not pd.api.types.is_numeric_dtype(y_raw.dtype):
        raise ValueError(f"Target {target!r} must be numeric for linear regression.")

    X = _one_hot_encode(X_raw)
    y = y_raw.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    out = {
        "problem_type": "regression",
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "coefficients": pd.Series(model.coef_, index=X.columns).sort_values(ascending=False).to_frame("coef"),
        "intercept": float(model.intercept_),
    }
    return out


def kmeans_clusters(
    df: pd.DataFrame,
    features: Iterable[str],
    n_clusters: int = 3,
    random_state: int = 42,
):
    """
    KMeans clustering for numeric features.
    Returns a dict including cluster labels and centers.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    _ensure_df(df)
    features = list(features)
    if not features:
        raise ValueError("features must be provided for clustering.")
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"feature columns not found: {missing}")

    X = df.loc[:, features].copy()
    X = X.dropna()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c].dtype):
            raise ValueError(f"Feature {c!r} must be numeric for KMeans.")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values.astype(float))

    km = KMeans(n_clusters=int(n_clusters), random_state=random_state, n_init="auto")
    labels = km.fit_predict(Xs)
    centers = scaler.inverse_transform(km.cluster_centers_)

    out = {
        "n_rows_used": int(X.shape[0]),
        "n_clusters": int(n_clusters),
        "labels": pd.Series(labels, index=X.index, name="cluster"),
        "centers": pd.DataFrame(centers, columns=features),
    }
    return out


def ttest_independent(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    group_a,
    group_b,
    equal_var: bool = False,
):
    """
    Two-sample t-test on value_col between two groups in group_col.
    """
    from scipy.stats import ttest_ind

    _ensure_df(df)
    if value_col not in df.columns or group_col not in df.columns:
        raise ValueError("value_col or group_col not found in DataFrame.")

    x = df.loc[df[group_col] == group_a, value_col].dropna()
    y = df.loc[df[group_col] == group_b, value_col].dropna()
    if x.empty or y.empty:
        raise ValueError("Not enough data in one or both groups.")

    stat, pval = ttest_ind(x.astype(float), y.astype(float), equal_var=bool(equal_var))
    return {
        "group_a": group_a,
        "group_b": group_b,
        "n_a": int(x.shape[0]),
        "n_b": int(y.shape[0]),
        "t_stat": float(stat),
        "p_value": float(pval),
        "mean_a": float(x.mean()),
        "mean_b": float(y.mean()),
    }

