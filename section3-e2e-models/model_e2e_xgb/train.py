import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV


def load_data(path: str) -> pd.DataFrame:
    # adapt to how you actually store final_df
    return pd.read_parquet(path)  # or read_csv


def train_model(df: pd.DataFrame, artifact_path: str = "model_e2e_xgb.joblib"):

    df = df.copy()
    y = df["final_cost"].astype(float)
    X = df.drop(columns=["final_cost"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    xgb_log = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        eval_metric="rmse",
        early_stopping_rounds=50,
        reg_lambda=1.0,
        reg_alpha=0.0,
    )

    param_dist = {
        "n_estimators":      randint(600, 2000),
        "learning_rate":     uniform(0.02, 0.10),
        "max_depth":         randint(3, 9),
        "min_child_weight":  randint(1, 7),
        "subsample":         uniform(0.7, 0.3),
        "colsample_bytree":  uniform(0.7, 0.3),
        "reg_lambda":        uniform(0.5, 2.0),
        "reg_alpha":         uniform(0.0, 0.5),
    }

    cv = KFold(n_splits=3, shuffle=True, random_state=42)

    rs = RandomizedSearchCV(
        estimator=xgb_log,
        param_distributions=param_dist,
        n_iter=25,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
        random_state=42,
    )

    rs.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    best_model_xgb = rs.best_estimator_

    predict_kwargs = {}
    if hasattr(best_model_xgb, "best_iteration") and best_model_xgb.best_iteration is not None:
        predict_kwargs["iteration_range"] = (0, best_model_xgb.best_iteration + 1)

    # Simple test RMSE so they see performance
    from math import sqrt
    y_pred_test = best_model_xgb.predict(X_test, **predict_kwargs)
    rmse = sqrt(mean_squared_error(y_test, y_pred_test))
    print(f"Test RMSE: {rmse:.3f}")

    artifact = {
        "model": best_model_xgb,
        "feature_names": X.columns.tolist(),
        "predict_kwargs": predict_kwargs,
    }
    joblib.dump(artifact, artifact_path)
    print(f"Saved artifact to {artifact_path}")


if __name__ == "__main__":
    df = load_data("final_df.parquet")  # adjust as needed
    train_model(df)
