import pandas as pd


def analyze_service_mix(
    service_requests_df,
    branch_name
):
    """
    Analyze service-request composition for one branch.
    """

    empty_result = {
        "total_requests": 0,
        "complex_requests": 0,
        "simple_requests": 0,
        "digital_eligible": 0,
        "digital_eligible_ratio": 0.0,
        "top_service_types": []
    }

    if (
        service_requests_df is None
        or service_requests_df.empty
    ):
        return empty_result

    required_columns = {
        "branch_name",
        "service_type",
        "is_complex",
        "digital_eligible"
    }

    missing = (
        required_columns
        - set(service_requests_df.columns)
    )

    if missing:
        raise ValueError(
            "service_requests is missing columns: "
            + ", ".join(sorted(missing))
        )

    df = service_requests_df[
        service_requests_df["branch_name"]
        .astype(str)
        == str(branch_name)
    ].copy()

    if df.empty:
        return empty_result

    complex_flags = pd.to_numeric(
        df["is_complex"],
        errors="coerce"
    ).fillna(0).clip(0, 1)

    digital_flags = pd.to_numeric(
        df["digital_eligible"],
        errors="coerce"
    ).fillna(0).clip(0, 1)

    total_requests = len(df)

    complex_requests = int(
        complex_flags.sum()
    )

    digital_eligible = int(
        digital_flags.sum()
    )

    top_service_types = (
        df["service_type"]
        .astype(str)
        .value_counts()
        .head(3)
        .index
        .tolist()
    )

    return {
        "total_requests":
            total_requests,

        "complex_requests":
            complex_requests,

        "simple_requests":
            total_requests - complex_requests,

        "digital_eligible":
            digital_eligible,

        "digital_eligible_ratio":
            round(
                digital_eligible
                / total_requests,
                3
            ),

        "top_service_types":
            top_service_types
    }