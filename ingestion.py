import os
import pandas as pd


REQUIRED_FILES = {
    "token_queue": "token_queue.csv",
    "branch_visits": "branch_visits.csv",
    "staff_roster": "staff_roster.csv",
    "appointments": "appointments.csv",
    "service_requests": "service_requests.csv",
    "customer_feedback": "customer_feedback.csv",
}


def load_operational_data(data_dir="data"):
    """
    Load all raw operational datasets.
    """

    data = {}
    missing_files = []

    for key, filename in REQUIRED_FILES.items():
        path = os.path.join(data_dir, filename)

        if not os.path.exists(path):
            missing_files.append(filename)
        else:
            data[key] = pd.read_csv(path)

    if missing_files:
        raise FileNotFoundError(
            "Missing operational data files: "
            + ", ".join(missing_files)
        )

    return data


def _rows_for_branch(df, branch_name):
    """
    Filter a raw dataset to one branch.
    """

    if df.empty or "branch_name" not in df.columns:
        return pd.DataFrame()

    return df[
        df["branch_name"].astype(str) == str(branch_name)
    ].copy()


def get_branch_names(data):
    """
    Discover available branches from loaded operational data.
    """

    names = set()

    for df in data.values():

        if not df.empty and "branch_name" in df.columns:
            names.update(
                df["branch_name"]
                .dropna()
                .astype(str)
                .tolist()
            )

    return sorted(names)


def build_branch_features(data, branch_name):
    """
    Convert raw branch operational records into the exact
    six features required by model.py.
    """

    queue_df = _rows_for_branch(
        data["token_queue"],
        branch_name
    )

    visits_df = _rows_for_branch(
        data["branch_visits"],
        branch_name
    )

    staff_df = _rows_for_branch(
        data["staff_roster"],
        branch_name
    )

    appointments_df = _rows_for_branch(
        data["appointments"],
        branch_name
    )

    requests_df = _rows_for_branch(
        data["service_requests"],
        branch_name
    )

    # -------------------------------------------------
    # Queue length
    # -------------------------------------------------

    queue_length = 0

    if "status" in queue_df.columns:
        queue_length = int(
            queue_df["status"]
            .astype(str)
            .str.lower()
            .eq("waiting")
            .sum()
        )

    # -------------------------------------------------
    # Available staff
    # -------------------------------------------------

    staff_available = 0

    if "status" in staff_df.columns:
        staff_available = int(
            staff_df["status"]
            .astype(str)
            .str.lower()
            .eq("available")
            .sum()
        )

    if staff_available <= 0:
        raise ValueError(
            f"No available staff recorded for {branch_name}."
        )

    # -------------------------------------------------
    # Average service duration
    # -------------------------------------------------

    avg_service_time = 0.0

    if (
        "service_time_minutes" in visits_df.columns
        and not visits_df.empty
    ):

        values = pd.to_numeric(
            visits_df["service_time_minutes"],
            errors="coerce"
        ).dropna()

        if not values.empty:
            avg_service_time = float(values.mean())

    # -------------------------------------------------
    # Appointments expected next hour
    # -------------------------------------------------

    appointments_next_hour = 0

    if "within_next_hour" in appointments_df.columns:

        appointments_next_hour = int(
            pd.to_numeric(
                appointments_df["within_next_hour"],
                errors="coerce"
            )
            .fillna(0)
            .clip(0, 1)
            .sum()
        )

    # -------------------------------------------------
    # Recent walk-ins
    # -------------------------------------------------

    recent_walkins = 0

    if "recent_walkin" in visits_df.columns:

        recent_walkins = int(
            pd.to_numeric(
                visits_df["recent_walkin"],
                errors="coerce"
            )
            .fillna(0)
            .clip(0, 1)
            .sum()
        )

    # -------------------------------------------------
    # Complex-request ratio
    # -------------------------------------------------

    complex_request_ratio = 0.0

    if (
        "is_complex" in requests_df.columns
        and not requests_df.empty
    ):

        complex_flags = pd.to_numeric(
            requests_df["is_complex"],
            errors="coerce"
        ).dropna().clip(0, 1)

        if not complex_flags.empty:
            complex_request_ratio = float(
                complex_flags.mean()
            )

    return {
        "queue_length": queue_length,
        "staff_available": staff_available,
        "avg_service_time": round(
            max(0.0, avg_service_time),
            2
        ),
        "appointments_next_hour":
            appointments_next_hour,
        "recent_walkins":
            recent_walkins,
        "complex_request_ratio": round(
            min(
                1.0,
                max(0.0, complex_request_ratio)
            ),
            3
        )
    }


def get_feedback_for_branch(data, branch_name):
    """
    Return anonymized textual feedback for one branch.
    """

    feedback_df = _rows_for_branch(
        data["customer_feedback"],
        branch_name
    )

    if (
        feedback_df.empty
        or "comment" not in feedback_df.columns
    ):
        return []

    return [
        str(comment).strip()
        for comment
        in feedback_df["comment"].dropna().tolist()
        if str(comment).strip()
    ]