from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")

RNG = np.random.default_rng(42)


BRANCHES = {

    "Central Branch": {
        "queue": 28,
        "staff_available": 3,
        "avg_service_time": 9.0,
        "appointments_next_hour": 12,
        "recent_walkins": 17,
        "complex_ratio": 0.42
    },

    "Lake View Branch": {
        "queue": 9,
        "staff_available": 5,
        "avg_service_time": 6.0,
        "appointments_next_hour": 5,
        "recent_walkins": 6,
        "complex_ratio": 0.20
    },

    "Tech Park Branch": {
        "queue": 18,
        "staff_available": 4,
        "avg_service_time": 7.5,
        "appointments_next_hour": 9,
        "recent_walkins": 11,
        "complex_ratio": 0.32
    }
}


SERVICE_TYPES = [

    (
        "Account Statement",
        0,
        1
    ),

    (
        "Balance Enquiry",
        0,
        1
    ),

    (
        "Cash Deposit",
        0,
        0
    ),

    (
        "KYC Update",
        1,
        0
    ),

    (
        "Loan Consultation",
        1,
        0
    ),

    (
        "Cheque Status",
        0,
        1
    )
]


FEEDBACK = {

    "Central Branch": [
        (
            2,
            "The queue was very long."
        ),
        (
            3,
            "Staff were helpful, but I waited too long."
        ),
        (
            2,
            "Only a few counters seemed available."
        )
    ],

    "Lake View Branch": [
        (
            5,
            "Fast service and helpful staff."
        ),
        (
            4,
            "The queue moved quickly."
        )
    ],

    "Tech Park Branch": [
        (
            3,
            "Service was fine but the wait was noticeable."
        ),
        (
            4,
            "Helpful staff and a manageable queue."
        )
    ]
}


def main():

    DATA_DIR.mkdir(
        exist_ok=True
    )

    queue_rows = []
    staff_rows = []
    visit_rows = []
    appointment_rows = []
    request_rows = []
    feedback_rows = []

    token_id = 1
    staff_id = 1
    visit_id = 1
    appointment_id = 1
    request_id = 1
    feedback_id = 1

    for branch, profile in BRANCHES.items():

        # ---------------------------------------------
        # TOKEN QUEUE
        # ---------------------------------------------

        for _ in range(
            profile["queue"]
        ):

            queue_rows.append({
                "token_id":
                    f"T{token_id:04d}",

                "branch_name":
                    branch,

                "status":
                    "waiting"
            })

            token_id += 1

        # ---------------------------------------------
        # STAFF ROSTER
        # ---------------------------------------------

        for _ in range(
            profile["staff_available"]
        ):

            staff_rows.append({
                "employee_id":
                    f"E{staff_id:04d}",

                "branch_name":
                    branch,

                "status":
                    "available"
            })

            staff_id += 1

        # Some employees are busy
        for _ in range(2):

            staff_rows.append({
                "employee_id":
                    f"E{staff_id:04d}",

                "branch_name":
                    branch,

                "status":
                    "busy"
            })

            staff_id += 1

        # ---------------------------------------------
        # BRANCH VISITS
        # ---------------------------------------------

        n_visits = max(
            20,
            profile["recent_walkins"]
        )

        walkin_flags = (
            [1]
            * profile["recent_walkins"]
            +
            [0]
            * (
                n_visits
                - profile["recent_walkins"]
            )
        )

        RNG.shuffle(
            walkin_flags
        )

        service_times = np.maximum(
            1.0,

            RNG.normal(
                profile["avg_service_time"],
                1.2,
                n_visits
            )
        )

        for walkin, service_time in zip(
            walkin_flags,
            service_times
        ):

            visit_rows.append({
                "visit_id":
                    f"V{visit_id:04d}",

                "branch_name":
                    branch,

                "service_time_minutes":
                    round(
                        float(service_time),
                        2
                    ),

                "recent_walkin":
                    int(walkin)
            })

            visit_id += 1

        # ---------------------------------------------
        # APPOINTMENTS
        # ---------------------------------------------

        for _ in range(
            profile[
                "appointments_next_hour"
            ]
        ):

            appointment_rows.append({
                "appointment_id":
                    f"A{appointment_id:04d}",

                "branch_name":
                    branch,

                "within_next_hour":
                    1,

                "status":
                    "confirmed"
            })

            appointment_id += 1

        # Additional later appointments
        for _ in range(3):

            appointment_rows.append({
                "appointment_id":
                    f"A{appointment_id:04d}",

                "branch_name":
                    branch,

                "within_next_hour":
                    0,

                "status":
                    "confirmed"
            })

            appointment_id += 1

        # ---------------------------------------------
        # SERVICE REQUESTS
        # ---------------------------------------------

        total_requests = 25

        complex_count = round(
            total_requests
            * profile["complex_ratio"]
        )

        complex_services = [
            item
            for item in SERVICE_TYPES
            if item[1] == 1
        ]

        simple_services = [
            item
            for item in SERVICE_TYPES
            if item[1] == 0
        ]

        selected_services = []

        for _ in range(
            complex_count
        ):

            selected_services.append(
                complex_services[
                    RNG.integers(
                        0,
                        len(complex_services)
                    )
                ]
            )

        for _ in range(
            total_requests
            - complex_count
        ):

            selected_services.append(
                simple_services[
                    RNG.integers(
                        0,
                        len(simple_services)
                    )
                ]
            )

        RNG.shuffle(
            selected_services
        )

        for (
            service_type,
            is_complex,
            digital_eligible
        ) in selected_services:

            request_rows.append({
                "request_id":
                    f"R{request_id:04d}",

                "branch_name":
                    branch,

                "service_type":
                    service_type,

                "is_complex":
                    is_complex,

                "digital_eligible":
                    digital_eligible
            })

            request_id += 1

        # ---------------------------------------------
        # CUSTOMER FEEDBACK
        # ---------------------------------------------

        for rating, comment in FEEDBACK[branch]:

            feedback_rows.append({
                "feedback_id":
                    f"F{feedback_id:04d}",

                "branch_name":
                    branch,

                "rating":
                    rating,

                "comment":
                    comment
            })

            feedback_id += 1

    # -------------------------------------------------
    # SAVE EVERYTHING
    # -------------------------------------------------

    pd.DataFrame(
        queue_rows
    ).to_csv(
        DATA_DIR / "token_queue.csv",
        index=False
    )

    pd.DataFrame(
        staff_rows
    ).to_csv(
        DATA_DIR / "staff_roster.csv",
        index=False
    )

    pd.DataFrame(
        visit_rows
    ).to_csv(
        DATA_DIR / "branch_visits.csv",
        index=False
    )

    pd.DataFrame(
        appointment_rows
    ).to_csv(
        DATA_DIR / "appointments.csv",
        index=False
    )

    pd.DataFrame(
        request_rows
    ).to_csv(
        DATA_DIR / "service_requests.csv",
        index=False
    )

    pd.DataFrame(
        feedback_rows
    ).to_csv(
        DATA_DIR / "customer_feedback.csv",
        index=False
    )

    print(
        "Sample operational datasets generated successfully."
    )


if __name__ == "__main__":
    main()