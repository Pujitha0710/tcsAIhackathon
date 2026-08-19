# pages/2_Customer_Guidance.py

import pandas as pd
import streamlit as st

from data_utils import (
    generate_synthetic_data,
)

from model import (
    train_model,
)

from ingestion import (
    load_operational_data,
    get_branch_names,
)

from customer_guidance import (
    get_service_options,
    build_customer_guidance,
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Customer Guidance | BranchPulse AI",
    page_icon="👤",
    layout="wide",
)


st.title(
    "👤 BranchPulse Customer Guidance"
)

st.subheader(
    "Choose the appropriate service channel "
    "and compare branch pressure."
)

st.caption(
    "Customer-facing decision guidance powered by "
    "the same BranchPulse operational intelligence."
)

st.divider()


# =========================================================
# MODEL
# =========================================================

@st.cache_data
def load_training_data():

    return generate_synthetic_data(
        n=1200,
        seed=42,
    )


@st.cache_resource
def load_model():

    return train_model(
        load_training_data()
    )


# =========================================================
# OPERATIONAL DATA
# =========================================================

@st.cache_data
def load_data():

    return load_operational_data(
        data_dir="data"
    )


try:

    model = load_model()

    operational_data = load_data()

except Exception as exc:

    st.error(
        f"Unable to initialize customer guidance: {exc}"
    )

    st.stop()


# =========================================================
# SERVICE DATA
# =========================================================

service_requests = operational_data.get(
    "service_requests",
    pd.DataFrame(),
)


service_options = get_service_options(
    service_requests
)


if not service_options:

    st.error(
        "No service types are available "
        "in the current sample data."
    )

    st.stop()


branch_names = get_branch_names(
    operational_data
)


# =========================================================
# CUSTOMER INPUT
# =========================================================

st.markdown(
    "## What service do you need?"
)


service_type = st.selectbox(
    "Select Service",
    service_options,
)


branch_choice_options = [
    "No branch preference"
] + branch_names


preferred_branch_option = (
    st.selectbox(
        "Preferred / Current Branch",
        branch_choice_options,
    )
)


preferred_branch = (
    None
    if (
        preferred_branch_option
        == "No branch preference"
    )
    else preferred_branch_option
)


st.caption(
    "BranchPulse compares sample operational "
    "conditions. Geographic distance and travel "
    "time are not included in this prototype."
)


# =========================================================
# GUIDANCE
# =========================================================

if st.button(
    "Find Best Service Option",
    type="primary",
    width="stretch",
):

    try:

        guidance = (
            build_customer_guidance(
                model=model,
                operational_data=
                    operational_data,
                service_type=
                    service_type,
                preferred_branch=
                    preferred_branch,
            )
        )


        # =================================================
        # CHANNEL GUIDANCE
        # =================================================

        st.divider()

        st.markdown(
            "## Recommended Service Channel"
        )


        digital_status = guidance[
            "digital_eligible"
        ]


        if digital_status is True:

            st.success(
                "💻 Digital / Self-Service "
                "may be suitable"
            )

        elif digital_status is False:

            st.info(
                "🏦 Branch-based service "
                "guidance recommended"
            )

        else:

            st.warning(
                "⚠️ Digital eligibility "
                "could not be confirmed"
            )


        st.write(
            guidance[
                "primary_message"
            ]
        )


        # =================================================
        # BRANCH OPTIONS
        # =================================================

        st.markdown(
            "## Branch Options"
        )


        ranked_branches = guidance[
            "ranked_branches"
        ]


        if ranked_branches:

            branch_rows = []

            for index, branch in enumerate(
                ranked_branches,
                start=1,
            ):

                branch_rows.append({
                    "Rank":
                        index,

                    "Branch":
                        branch[
                            "branch_name"
                        ],

                    "Estimated Wait (min)":
                        round(
                            branch[
                                "predicted_wait"
                            ],
                            1,
                        ),

                    "Pressure":
                        branch[
                            "pressure"
                        ],
                })


            branch_df = pd.DataFrame(
                branch_rows
            )


            st.dataframe(
                branch_df,
                width="stretch",
                hide_index=True,
            )


            st.success(
                guidance[
                    "branch_message"
                ]
            )


            if (
                guidance[
                    "comparison_message"
                ]
            ):

                st.info(
                    guidance[
                        "comparison_message"
                    ]
                )


        else:

            st.warning(
                "No branch comparison could "
                "be generated for this service."
            )


        # =================================================
        # CUSTOMER DECISION SUMMARY
        # =================================================

        st.markdown(
            "## Your Options"
        )


        if digital_status is True:

            st.write(
                "### Option 1 — Digital / Self-Service"
            )

            st.write(
                "Consider checking the bank's "
                "authorized digital or self-service "
                "channel before travelling to a branch."
            )


            if guidance["best_branch"]:

                st.write(
                    "### Option 2 — Branch Service"
                )

                st.write(
                    guidance[
                        "branch_message"
                    ]
                )


        else:

            st.write(
                "### Branch Service"
            )

            st.write(
                guidance[
                    "branch_message"
                ]
            )


        # =================================================
        # IMPORTANT LIMITATION
        # =================================================

        st.warning(
            "Prototype guidance only: branch distance, "
            "travel time, opening hours, real-time capacity "
            "and official service availability are not "
            "included. A production banking system would "
            "verify all of these before redirecting customers."
        )


    except Exception as exc:

        st.error(
            f"Unable to generate customer guidance: {exc}"
        )


# =========================================================
# EXPLANATION
# =========================================================

st.divider()


with st.expander(
    "How is customer guidance generated?"
):

    st.markdown(
        """
BranchPulse Customer Guidance uses three pieces of information:

1. **Service metadata**  
   The sample service-request data indicates whether a service
   may be digitally eligible.

2. **Service-aware branch filtering**  
   Branches where the selected service appears in the sample
   operational records are considered.

3. **ML branch-pressure prediction**  
   The same Random Forest model used by the branch-manager
   dashboard estimates waiting time for each candidate branch.

The branches are then ranked by model-estimated waiting time.

This is a **decision-support prototype**.

It does not guarantee official service availability, geographic
convenience, or actual travel-time savings.
"""
    )