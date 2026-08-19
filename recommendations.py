# recommendations.py


REQUIRED_FEATURES = [
    "queue_length",
    "staff_available",
    "avg_service_time",
    "appointments_next_hour",
    "recent_walkins",
    "complex_request_ratio"
]


def _validate_features(features):
    """
    Basic validation for recommendation inputs.
    """

    if not isinstance(features, dict):
        raise TypeError("features must be a dictionary.")

    missing = [
        feature
        for feature in REQUIRED_FEATURES
        if feature not in features
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )


def identify_bottlenecks(features):
    """
    Identify visible operational bottlenecks using
    transparent business rules.

    Parameters
    ----------
    features : dict

    Returns
    -------
    list[str]
    """

    _validate_features(features)

    queue_length = float(features["queue_length"])
    staff_available = float(features["staff_available"])
    avg_service_time = float(features["avg_service_time"])
    appointments = float(
        features["appointments_next_hour"]
    )
    walkins = float(features["recent_walkins"])
    complexity = float(
        features["complex_request_ratio"]
    )

    bottlenecks = []

    # -----------------------------------------------------
    # Queue compared with available staff
    # -----------------------------------------------------

    if staff_available > 0:
        queue_per_staff = queue_length / staff_available

        if queue_per_staff >= 6:
            bottlenecks.append(
                "Large queue relative to available staff"
            )

    # -----------------------------------------------------
    # Staffing
    # -----------------------------------------------------

    if staff_available <= 2:
        bottlenecks.append(
            "Very limited staff availability"
        )

    # -----------------------------------------------------
    # Service duration
    # -----------------------------------------------------

    if avg_service_time >= 10:
        bottlenecks.append(
            "High average service duration"
        )

    # -----------------------------------------------------
    # Appointment surge
    # -----------------------------------------------------

    if appointments >= 10:
        bottlenecks.append(
            "High number of upcoming appointments"
        )

    # -----------------------------------------------------
    # Walk-in surge
    # -----------------------------------------------------

    if walkins >= 15:
        bottlenecks.append(
            "Recent walk-in customer surge"
        )

    # -----------------------------------------------------
    # Complex services
    # -----------------------------------------------------

    if complexity >= 0.45:
        bottlenecks.append(
            "High proportion of complex service requests"
        )

    return bottlenecks


def generate_recommendations(
    features,
    prediction,
    alternatives=None
):
    """
    Generate explainable operational suggestions.

    Parameters
    ----------
    features : dict
        Current branch operating conditions.

    prediction : dict
        Expected format:
        {
            "predicted_wait": float,
            "pressure": str
        }

    alternatives : list[dict] | None
        Optional alternative branch predictions.

    Returns
    -------
    list[str]
    """

    _validate_features(features)

    if not isinstance(prediction, dict):
        raise TypeError(
            "prediction must be a dictionary."
        )

    if (
        "predicted_wait" not in prediction
        or "pressure" not in prediction
    ):
        raise ValueError(
            "prediction must contain predicted_wait and pressure."
        )

    queue_length = float(features["queue_length"])
    staff_available = float(features["staff_available"])
    avg_service_time = float(features["avg_service_time"])
    appointments = float(
        features["appointments_next_hour"]
    )
    walkins = float(features["recent_walkins"])
    complexity = float(
        features["complex_request_ratio"]
    )

    predicted_wait = float(
        prediction["predicted_wait"]
    )

    pressure = str(
        prediction["pressure"]
    ).upper()

    recommendations = []

    # -----------------------------------------------------
    # Queue / staffing imbalance
    # -----------------------------------------------------

    if (
        staff_available > 0
        and queue_length / staff_available >= 6
    ):
        recommendations.append(
            "Consider assigning an additional employee "
            "or opening another service counter."
        )

    # -----------------------------------------------------
    # Very low staffing
    # -----------------------------------------------------

    if staff_available <= 2:
        recommendations.append(
            "The manager may review whether additional "
            "staff can temporarily support customer service."
        )

    # -----------------------------------------------------
    # Long service duration
    # -----------------------------------------------------

    if avg_service_time >= 10:
        recommendations.append(
            "Consider separating simple and complex "
            "service requests where operationally feasible."
        )

    # -----------------------------------------------------
    # Appointment pressure
    # -----------------------------------------------------

    if appointments >= 10:
        recommendations.append(
            "Consider prioritizing customers with existing "
            "appointments to reduce appointment congestion."
        )

    # -----------------------------------------------------
    # Walk-in surge
    # -----------------------------------------------------

    if walkins >= 15:
        recommendations.append(
            "Consider triaging recent walk-ins and directing "
            "eligible simple requests to suitable digital channels."
        )

    # -----------------------------------------------------
    # Complex requests
    # -----------------------------------------------------

    if complexity >= 0.45:
        recommendations.append(
            "Consider directing complex requests to experienced "
            "staff or a dedicated service counter."
        )

    # -----------------------------------------------------
    # General high-pressure warning
    # -----------------------------------------------------

    if pressure in {"HIGH", "CRITICAL"}:
        recommendations.append(
            "Suggested action: monitor the queue closely and "
            "review short-term staff allocation."
        )

    # -----------------------------------------------------
    # Alternative branch recommendation
    # -----------------------------------------------------

    if alternatives and pressure in {"HIGH", "CRITICAL"}:
        valid_alternatives = []

        for alternative in alternatives:

            if not isinstance(alternative, dict):
                continue

            branch_name = alternative.get("branch_name")
            alternative_wait = alternative.get(
                "predicted_wait"
            )

            if (
                branch_name is None
                or alternative_wait is None
            ):
                continue

            try:
                alternative_wait = float(
                    alternative_wait
                )
            except (TypeError, ValueError):
                continue

            if alternative_wait < predicted_wait:
                valid_alternatives.append({
                    "branch_name": str(branch_name),
                    "predicted_wait": alternative_wait
                })

        if valid_alternatives:

            best_branch = min(
                valid_alternatives,
                key=lambda item: item["predicted_wait"]
            )

            wait_difference = (
                predicted_wait
                - best_branch["predicted_wait"]
            )

            # Only recommend redirection when the difference
            # is operationally meaningful.
            if wait_difference >= 5:
                recommendations.append(
                    f"Eligible customers could be redirected to "
                    f"{best_branch['branch_name']}, where the "
                    f"estimated wait is approximately "
                    f"{best_branch['predicted_wait']:.1f} minutes."
                )

    # -----------------------------------------------------
    # No major issue detected
    # -----------------------------------------------------

    if not recommendations:
        recommendations.append(
            "Current operating conditions appear manageable. "
            "Continue monitoring branch demand and staffing."
        )

    return recommendations