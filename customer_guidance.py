# customer_guidance.py

import pandas as pd

from ingestion import (
    build_branch_features,
    get_branch_names,
)

from model import predict_pressure


# =========================================================
# SERVICE OPTIONS
# =========================================================

def get_service_options(service_requests_df):
    """
    Return all unique service types represented
    in the sample service-request dataset.
    """

    if (
        service_requests_df is None
        or service_requests_df.empty
    ):
        return []

    if "service_type" not in service_requests_df.columns:
        return []

    services = (
        service_requests_df["service_type"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    services = services[
        services != ""
    ]

    return sorted(
        services.unique().tolist()
    )


# =========================================================
# DIGITAL ELIGIBILITY
# =========================================================

def get_digital_eligibility(
    service_requests_df,
    service_type
):
    """
    Determine whether a service is marked as potentially
    digital eligible in the sample dataset.

    Returns:
        True  -> consistently marked digital eligible
        False -> consistently marked branch/non-digital
        None  -> unavailable or inconsistent metadata
    """

    if (
        service_requests_df is None
        or service_requests_df.empty
    ):
        return None

    required = {
        "service_type",
        "digital_eligible",
    }

    if not required.issubset(
        service_requests_df.columns
    ):
        return None

    matching = service_requests_df[
        service_requests_df["service_type"]
        .astype(str)
        == str(service_type)
    ].copy()

    if matching.empty:
        return None

    values = pd.to_numeric(
        matching["digital_eligible"],
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    values = values.clip(0, 1)

    unique_values = set(
        values.astype(int).tolist()
    )

    # Do not make a strong recommendation if
    # the sample metadata contradicts itself.
    if len(unique_values) != 1:
        return None

    return bool(
        next(iter(unique_values))
    )


# =========================================================
# SERVICE REPRESENTATION BY BRANCH
# =========================================================

def get_branches_with_service(
    service_requests_df,
    service_type
):
    """
    Return branches where the selected service is represented
    in the sample service-request data.

    NOTE:
    This is NOT proof of official service availability.
    """

    if (
        service_requests_df is None
        or service_requests_df.empty
    ):
        return []

    required = {
        "branch_name",
        "service_type",
    }

    if not required.issubset(
        service_requests_df.columns
    ):
        return []

    matching = service_requests_df[
        service_requests_df["service_type"]
        .astype(str)
        == str(service_type)
    ]

    return sorted(
        matching["branch_name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


# =========================================================
# RANK BRANCHES
# =========================================================

def rank_branches_for_service(
    model,
    operational_data,
    service_type
):
    """
    Rank sample branches by model-estimated waiting time,
    considering only branches where the selected service
    is represented in the sample service-request data.
    """

    if operational_data is None:
        return []

    service_requests = operational_data.get(
        "service_requests",
        pd.DataFrame(),
    )

    compatible_branches = (
        get_branches_with_service(
            service_requests,
            service_type,
        )
    )

    # Fallback:
    # If sample service records do not give us enough
    # information, evaluate all available sample branches.
    if not compatible_branches:
        compatible_branches = get_branch_names(
            operational_data
        )

    results = []

    for branch_name in compatible_branches:

        try:
            features = build_branch_features(
                operational_data,
                branch_name,
            )

            prediction = predict_pressure(
                model,
                features,
            )

            results.append({
                "branch_name":
                    branch_name,

                "predicted_wait":
                    round(
                        float(
                            prediction[
                                "predicted_wait"
                            ]
                        ),
                        2,
                    ),

                "pressure":
                    prediction["pressure"],
            })

        except Exception:
            # One malformed branch should not
            # break customer guidance.
            continue

    results.sort(
        key=lambda item:
            item["predicted_wait"]
    )

    return results


# =========================================================
# CUSTOMER GUIDANCE
# =========================================================

def build_customer_guidance(
    model,
    operational_data,
    service_type,
    preferred_branch=None,
):
    """
    Build customer-facing guidance.

    This function does NOT make banking decisions.
    It provides service-channel and branch-pressure guidance
    from sample metadata + model estimates.
    """

    service_requests = operational_data.get(
        "service_requests",
        pd.DataFrame(),
    )

    digital_eligible = (
        get_digital_eligibility(
            service_requests,
            service_type,
        )
    )

    ranked_branches = (
        rank_branches_for_service(
            model,
            operational_data,
            service_type,
        )
    )

    best_branch = (
        ranked_branches[0]
        if ranked_branches
        else None
    )

    preferred_result = None

    if preferred_branch:

        preferred_result = next(
            (
                branch
                for branch in ranked_branches
                if (
                    branch["branch_name"]
                    == preferred_branch
                )
            ),
            None,
        )

    # -----------------------------------------------------
    # PRIMARY CHANNEL GUIDANCE
    # -----------------------------------------------------

    if digital_eligible is True:

        primary_channel = (
            "Digital / Self-Service"
        )

        primary_message = (
            f"{service_type} is marked as potentially "
            "digital-eligible in this prototype. "
            "A branch visit may not be necessary for "
            "eligible customers."
        )

    elif digital_eligible is False:

        primary_channel = (
            "Branch Service"
        )

        primary_message = (
            f"{service_type} is not marked as "
            "digital-eligible in the current sample data. "
            "Branch-based service guidance is shown below."
        )

    else:

        primary_channel = (
            "Eligibility Uncertain"
        )

        primary_message = (
            "The prototype does not have sufficiently "
            "consistent metadata to determine digital "
            "eligibility for this service."
        )

    # -----------------------------------------------------
    # BRANCH GUIDANCE
    # -----------------------------------------------------

    branch_message = (
        "No suitable sample branch prediction "
        "could be generated."
    )

    if best_branch:

        branch_message = (
            f"Among the sample branches evaluated, "
            f"{best_branch['branch_name']} currently has "
            f"the lowest model-estimated wait at "
            f"approximately "
            f"{best_branch['predicted_wait']:.1f} minutes."
        )

    # -----------------------------------------------------
    # CURRENT/PREFERRED BRANCH COMPARISON
    # -----------------------------------------------------

    comparison_message = None

    if (
        preferred_result is not None
        and best_branch is not None
    ):

        difference = (
            preferred_result[
                "predicted_wait"
            ]
            - best_branch[
                "predicted_wait"
            ]
        )

        if (
            preferred_result[
                "branch_name"
            ]
            == best_branch[
                "branch_name"
            ]
        ):

            comparison_message = (
                f"{preferred_branch} already has the "
                "lowest estimated wait among the "
                "sample-compatible branches."
            )

        elif difference >= 5:

            comparison_message = (
                f"{best_branch['branch_name']} has a "
                f"model-estimated wait approximately "
                f"{difference:.1f} minutes lower than "
                f"{preferred_branch}."
            )

        else:

            comparison_message = (
                "The model-estimated waiting-time "
                "difference between the preferred branch "
                "and the lowest-wait sample option is small."
            )

    return {
        "service_type":
            service_type,

        "digital_eligible":
            digital_eligible,

        "primary_channel":
            primary_channel,

        "primary_message":
            primary_message,

        "ranked_branches":
            ranked_branches,

        "best_branch":
            best_branch,

        "preferred_branch":
            preferred_branch,

        "preferred_result":
            preferred_result,

        "branch_message":
            branch_message,

        "comparison_message":
            comparison_message,
    }