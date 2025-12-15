import joblib
import pandas as pd
from typing import Union, List, Dict, Any

# Load artifact once at import time
_ARTIFACT_PATH = "model_e2e_xgb.joblib"
_artifact = joblib.load(_ARTIFACT_PATH)

_model = _artifact["model"]
_feature_names = _artifact["feature_names"]
_predict_kwargs = _artifact.get("predict_kwargs", {})


def _prepare_input(X: Union[pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    Ensure that the input has exactly the same columns and order
    as the training data.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Reindex to training feature set; any missing cols become 0
    X = X.reindex(columns=_feature_names, fill_value=0)

    return X


def predict_cost(X: Union[pd.DataFrame, List[Dict[str, Any]]]) -> pd.Series:
    """
    Predict final_cost for one or multiple call records.

    Parameters
    ----------
    X : DataFrame or list of dicts
        Input data with feature columns.

    Returns
    -------
    Pandas Series with predicted final_cost.
    """
    X_prepared = _prepare_input(X)
    y_pred = _model.predict(X_prepared, **_predict_kwargs)
    return pd.Series(y_pred, index=X_prepared.index, name="pred_final_cost")


if __name__ == "__main__":
    # Example usage: dummy row with feature names
    example = [{col: 0 for col in _feature_names}]
    preds = predict_cost(example)
    print("Example prediction:", preds.iloc[0])
