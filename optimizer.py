def find_min_staff_for_target(
    model,
    features,
    predict_fn,
    target_wait=20.0,
    max_extra_staff=6
):
    """
    Test staffing scenarios and find the minimum tested
    staff level whose predicted waiting time reaches the
    desired target.

    This is scenario analysis, not a causal guarantee.
    """

    if target_wait <= 0:
        raise ValueError(
            "target_wait must be greater than 0."
        )

    if max_extra_staff < 0:
        raise ValueError(
            "max_extra_staff cannot be negative."
        )

    if "staff_available" not in features:
        raise ValueError(
            "features must contain staff_available."
        )

    current_staff = int(
        features["staff_available"]
    )

    if current_staff <= 0:
        raise ValueError(
            "staff_available must be greater than 0."
        )

    scenarios = []
    best_scenario = None

    for staff in range(
        current_staff,
        current_staff + max_extra_staff + 1
    ):

        simulated_features = dict(features)

        simulated_features[
            "staff_available"
        ] = staff

        prediction = predict_fn(
            model,
            simulated_features
        )

        scenario = {
            "staff": staff,
            "predicted_wait": round(
                float(
                    prediction["predicted_wait"]
                ),
                2
            ),
            "pressure": prediction["pressure"]
        }

        scenarios.append(scenario)

        if (
            scenario["predicted_wait"]
            <= target_wait
        ):
            best_scenario = scenario
            break

    current = scenarios[0]

    if best_scenario is None:
        chosen = scenarios[-1]
        target_met = False
    else:
        chosen = best_scenario
        target_met = True

    return {
        "current_staff": current_staff,
        "current_wait":
            current["predicted_wait"],

        "suggested_staff":
            chosen["staff"],

        "suggested_wait":
            chosen["predicted_wait"],

        "additional_staff":
            chosen["staff"] - current_staff,

        "target_wait":
            float(target_wait),

        "target_met":
            target_met,

        "scenarios":
            scenarios
    }