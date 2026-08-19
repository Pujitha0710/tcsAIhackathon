# data_utils.py

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "queue_length",
    "staff_available",
    "avg_service_time",
    "appointments_next_hour",
    "recent_walkins",
    "complex_request_ratio"
]

TARGET_COLUMN = "wait_time"


def generate_synthetic_data(n=1200, seed=42):
    """
    Generate logically plausible synthetic branch-operational data.

    Waiting time generally:
    - increases with queue length
    - increases with service time
    - increases with appointments
    - increases with walk-ins
    - increases with request complexity
    - decreases as available staff increases

    Parameters
    ----------
    n : int
        Number of rows to generate.

    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
    """

    if not isinstance(n, int):
        raise TypeError("n must be an integer.")

    if n <= 0:
        raise ValueError("n must be greater than 0.")

    if not isinstance(seed, int):
        raise TypeError("seed must be an integer.")

    rng = np.random.default_rng(seed)

    # -----------------------------------------------------
    # Generate operational features
    # -----------------------------------------------------

    queue_length = rng.integers(
        low=0,
        high=46,
        size=n
    )

    staff_available = rng.integers(
        low=1,
        high=9,
        size=n
    )

    avg_service_time = rng.uniform(
        low=3.0,
        high=15.0,
        size=n
    )

    appointments_next_hour = rng.integers(
        low=0,
        high=21,
        size=n
    )

    recent_walkins = rng.integers(
        low=0,
        high=26,
        size=n
    )

    complex_request_ratio = rng.uniform(
        low=0.0,
        high=1.0,
        size=n
    )

    # -----------------------------------------------------
    # Create synthetic wait-time relationship
    # -----------------------------------------------------

    # Customers currently waiting are the strongest signal.
    #
    # Upcoming appointments and recent walk-ins also create
    # operational pressure, although with smaller weights.
    effective_demand = (
        queue_length
        + 0.40 * appointments_next_hour
        + 0.30 * recent_walkins
    )

    # More staff reduces the load per available employee.
    workload_per_staff = (
        effective_demand / staff_available
    )

    # Longer average service time increases waiting.
    base_wait = (
        workload_per_staff
        * avg_service_time
        * 0.55
    )

    # Complex requests typically consume additional service capacity.
    complexity_penalty = (
        complex_request_ratio * 8.0
    )

    # Additional demand pressure.
    appointment_penalty = (
        appointments_next_hour * 0.12
    )

    walkin_penalty = (
        recent_walkins * 0.10
    )

    # Random noise prevents perfectly deterministic data.
    noise = rng.normal(
        loc=0.0,
        scale=2.5,
        size=n
    )

    wait_time = (
        1.5
        + base_wait
        + complexity_penalty
        + appointment_penalty
        + walkin_penalty
        + noise
    )

    # Waiting time can never logically be negative.
    wait_time = np.maximum(wait_time, 0.0)

    # -----------------------------------------------------
    # Create final DataFrame
    # -----------------------------------------------------

    df = pd.DataFrame({
        "queue_length": queue_length,
        "staff_available": staff_available,
        "avg_service_time": np.round(avg_service_time, 2),
        "appointments_next_hour": appointments_next_hour,
        "recent_walkins": recent_walkins,
        "complex_request_ratio": np.round(
            complex_request_ratio,
            3
        ),
        "wait_time": np.round(wait_time, 2)
    })

    return df


def get_sample_branches():
    """
    Return sample branch conditions for demonstration
    and alternative-branch comparison.

    Returns
    -------
    list[dict]
    """

    return [
        {
            "branch_name": "Central Branch",
            "queue_length": 28,
            "staff_available": 3,
            "avg_service_time": 9.0,
            "appointments_next_hour": 12,
            "recent_walkins": 17,
            "complex_request_ratio": 0.42
        },

        {
            "branch_name": "Lake View Branch",
            "queue_length": 9,
            "staff_available": 5,
            "avg_service_time": 6.0,
            "appointments_next_hour": 5,
            "recent_walkins": 6,
            "complex_request_ratio": 0.20
        },

        {
            "branch_name": "Tech Park Branch",
            "queue_length": 18,
            "staff_available": 4,
            "avg_service_time": 7.5,
            "appointments_next_hour": 9,
            "recent_walkins": 11,
            "complex_request_ratio": 0.32
        }
    ]