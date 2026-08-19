# model.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# ---------------------------------------------------------
# FROZEN FEATURE CONTRACT
# Do not rename or reorder these columns.
# ---------------------------------------------------------

FEATURE_COLUMNS = [
    "queue_length",
    "staff_available",
    "avg_service_time",
    "appointments_next_hour",
    "recent_walkins",
    "complex_request_ratio"
]

TARGET_COLUMN = "wait_time"


# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------

def train_model(df):
    """
    Train the waiting-time prediction model.

    Parameters
    ----------
    df : pandas.DataFrame
        Training data containing all FEATURE_COLUMNS
        and the TARGET_COLUMN.

    Returns
    -------
    RandomForestRegressor
        Trained regression model.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("train_model expects a pandas DataFrame.")

    if df.empty:
        raise ValueError("Training DataFrame cannot be empty.")

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Training data is missing required columns: "
            f"{missing_columns}"
        )

    # Work only with required columns
    data = df[required_columns].copy()

    # Ensure all required columns are numeric
    for column in required_columns:
        try:
            data[column] = pd.to_numeric(
                data[column],
                errors="raise"
            )
        except Exception as exc:
            raise ValueError(
                f"Column '{column}' must contain numeric values."
            ) from exc

    # Check missing values
    if data.isnull().any().any():
        bad_columns = data.columns[
            data.isnull().any()
        ].tolist()

        raise ValueError(
            f"Training data contains missing values in: "
            f"{bad_columns}"
        )

    # Check infinity
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError(
            "Training data contains infinite or invalid numeric values."
        )

    # Operational validity checks
    if (data["queue_length"] < 0).any():
        raise ValueError("queue_length cannot be negative.")

    if (data["staff_available"] <= 0).any():
        raise ValueError("staff_available must be greater than 0.")

    if (data["avg_service_time"] < 0).any():
        raise ValueError("avg_service_time cannot be negative.")

    if (data["appointments_next_hour"] < 0).any():
        raise ValueError(
            "appointments_next_hour cannot be negative."
        )

    if (data["recent_walkins"] < 0).any():
        raise ValueError("recent_walkins cannot be negative.")

    if (
        (data["complex_request_ratio"] < 0)
        | (data["complex_request_ratio"] > 1)
    ).any():
        raise ValueError(
            "complex_request_ratio must be between 0 and 1."
        )

    if (data[TARGET_COLUMN] < 0).any():
        raise ValueError("wait_time cannot be negative.")

    # IMPORTANT:
    # Keep feature order exactly the same everywhere.
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    # Simple, deterministic model suitable for the hackathon MVP
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    return model


# ---------------------------------------------------------
# VALIDATE ONE BRANCH INPUT
# ---------------------------------------------------------

def _validate_feature_dict(feature_dict):
    """
    Validate operational inputs before prediction.
    """

    if not isinstance(feature_dict, dict):
        raise TypeError(
            "feature_dict must be a dictionary."
        )

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in feature_dict
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    values = {}

    for feature in FEATURE_COLUMNS:
        value = feature_dict[feature]

        # bool is technically an int in Python,
        # but it should not be accepted here.
        if isinstance(value, bool):
            raise ValueError(
                f"'{feature}' must be numeric, not boolean."
            )

        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"'{feature}' must be numeric."
            ) from exc

        if not np.isfinite(value):
            raise ValueError(
                f"'{feature}' must contain a finite numeric value."
            )

        values[feature] = value

    # Business constraints

    if values["queue_length"] < 0:
        raise ValueError(
            "queue_length cannot be negative."
        )

    if values["staff_available"] <= 0:
        raise ValueError(
            "staff_available must be greater than 0."
        )

    if values["avg_service_time"] < 0:
        raise ValueError(
            "avg_service_time cannot be negative."
        )

    if values["appointments_next_hour"] < 0:
        raise ValueError(
            "appointments_next_hour cannot be negative."
        )

    if values["recent_walkins"] < 0:
        raise ValueError(
            "recent_walkins cannot be negative."
        )

    if not 0 <= values["complex_request_ratio"] <= 1:
        raise ValueError(
            "complex_request_ratio must be between 0 and 1."
        )

    return values


# ---------------------------------------------------------
# PRESSURE CLASSIFICATION
# ---------------------------------------------------------

def _classify_pressure(predicted_wait):
    """
    Convert predicted waiting time into operational pressure.
    """

    if predicted_wait < 10:
        return "LOW"

    if predicted_wait < 20:
        return "MODERATE"

    if predicted_wait < 35:
        return "HIGH"

    return "CRITICAL"


# ---------------------------------------------------------
# PREDICT WAIT + PRESSURE
# ---------------------------------------------------------

def predict_pressure(model, feature_dict):
    """
    Predict branch waiting time and classify pressure.

    Returns exactly:
    {
        "predicted_wait": float,
        "pressure": str
    }
    """

    if model is None:
        raise ValueError("A trained model must be provided.")

    if not hasattr(model, "predict"):
        raise TypeError(
            "model must be a trained regression model."
        )

    features = _validate_feature_dict(feature_dict)

    # DataFrame preserves the exact training feature names/order
    input_df = pd.DataFrame(
        [[features[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS
    )

    predicted_wait = float(model.predict(input_df)[0])

    # Safety guard.
    # Waiting time cannot logically be negative.
    predicted_wait = max(0.0, predicted_wait)

    pressure = _classify_pressure(predicted_wait)

    return {
        "predicted_wait": predicted_wait,
        "pressure": pressure
    }