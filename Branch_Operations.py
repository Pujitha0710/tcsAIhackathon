# app.py

import os
import pandas as pd
import streamlit as st

from data_utils import (
    generate_synthetic_data,
    get_sample_branches,
)

from model import (
    train_model,
    predict_pressure,
)

from recommendations import (
    identify_bottlenecks,
    generate_recommendations,
)

from ingestion import (
    load_operational_data,
    build_branch_features,
    get_branch_names,
    get_feedback_for_branch,
)

from optimizer import (
    find_min_staff_for_target,
)

from service_analysis import (
    analyze_service_mix,
)

from genai_layer import (
    analyze_customer_feedback,
    generate_operations_brief,
)


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="BranchPulse AI",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 BranchPulse AI")

st.subheader(
    "Intelligent Branch Service Load and "
    "Customer Experience Optimizer"
)

st.caption(
    "Predict service pressure • Identify bottlenecks • "
    "Test staffing scenarios • Understand customer experience"
)

st.divider()


# =========================================================
# LOAD TRAINING DATA + MODEL
# =========================================================

@st.cache_data
def load_training_data():
    """
    Generate reproducible synthetic ML training data.

    This dataset is used only to demonstrate the ML
    prediction architecture during the hackathon.
    """
    return generate_synthetic_data(
        n=1200,
        seed=42,
    )


@st.cache_resource
def load_trained_model():
    """
    Train and cache RandomForestRegressor.
    """
    training_df = load_training_data()

    return train_model(training_df)


try:
    training_df = load_training_data()
    model = load_trained_model()

except Exception as exc:
    st.error(
        f"Unable to initialize the ML model: {exc}"
    )
    st.stop()


# =========================================================
# LOAD RAW OPERATIONAL DATA
# =========================================================

@st.cache_data
def load_raw_data():
    """
    Load sample branch operational records from /data.
    """
    return load_operational_data(
        data_dir="data"
    )


operational_data = None
operational_data_error = None

try:
    operational_data = load_raw_data()

except Exception as exc:
    operational_data_error = str(exc)


# =========================================================
# DATA SOURCE
# =========================================================

st.markdown("## 1. Branch Data")

available_modes = []

if operational_data is not None:
    available_modes.append(
        "Operational Data"
    )

available_modes.append(
    "Manual Simulation"
)


data_mode = st.radio(
    "Choose input mode",
    options=available_modes,
    horizontal=True,
    help=(
        "Operational Data derives model inputs from sample "
        "queue, staff, appointment, visit and service records. "
        "Manual Simulation lets the manager test hypothetical conditions."
    ),
)


if operational_data_error:

    st.warning(
        "Raw operational data could not be loaded, so "
        "Manual Simulation remains available.\n\n"
        f"Reason: {operational_data_error}"
    )


# =========================================================
# HELPERS
# =========================================================

def filter_branch_rows(
    dataframe,
    branch_name
):
    """
    Safely filter any operational DataFrame
    to the selected branch.
    """

    if (
        dataframe is None
        or dataframe.empty
        or "branch_name" not in dataframe.columns
    ):
        return pd.DataFrame()

    return dataframe[
        dataframe["branch_name"].astype(str)
        == str(branch_name)
    ].copy()


def build_operational_alternatives(
    selected_branch_name
):
    """
    Predict wait/pressure for every other branch
    represented in the operational dataset.
    """

    alternatives = []

    branch_names = get_branch_names(
        operational_data
    )

    for branch_name in branch_names:

        if branch_name == selected_branch_name:
            continue

        try:
            alternative_features = (
                build_branch_features(
                    operational_data,
                    branch_name,
                )
            )

            alternative_prediction = (
                predict_pressure(
                    model,
                    alternative_features,
                )
            )

            alternatives.append({
                "branch_name":
                    branch_name,

                "predicted_wait":
                    alternative_prediction[
                        "predicted_wait"
                    ],

                "pressure":
                    alternative_prediction[
                        "pressure"
                    ],
            })

        except Exception:
            # One malformed sample branch should not
            # break the complete application.
            continue

    return alternatives


def build_manual_alternatives(
    selected_branch_name
):
    """
    Predict wait/pressure for the other
    predefined demonstration branches.
    """

    alternatives = []

    for branch in get_sample_branches():

        if (
            branch["branch_name"]
            == selected_branch_name
        ):
            continue

        alternative_features = {
            "queue_length":
                branch["queue_length"],

            "staff_available":
                branch["staff_available"],

            "avg_service_time":
                branch["avg_service_time"],

            "appointments_next_hour":
                branch["appointments_next_hour"],

            "recent_walkins":
                branch["recent_walkins"],

            "complex_request_ratio":
                branch["complex_request_ratio"],
        }

        alternative_prediction = (
            predict_pressure(
                model,
                alternative_features,
            )
        )

        alternatives.append({
            "branch_name":
                branch["branch_name"],

            "predicted_wait":
                alternative_prediction[
                    "predicted_wait"
                ],

            "pressure":
                alternative_prediction[
                    "pressure"
                ],
        })

    return alternatives


# =========================================================
# OPERATIONAL DATA MODE
# =========================================================

features = None
selected_branch_name = None


if data_mode == "Operational Data":

    branch_names = get_branch_names(
        operational_data
    )

    if not branch_names:

        st.error(
            "No branch names were found "
            "inside the operational datasets."
        )

        st.stop()

    selected_branch_name = st.selectbox(
        "Select Branch",
        branch_names,
    )

    try:
        features = build_branch_features(
            operational_data,
            selected_branch_name,
        )

    except Exception as exc:

        st.error(
            "Unable to derive operational features "
            f"for this branch: {exc}"
        )

        st.stop()


    st.markdown(
        "### Automatically Derived Operational State"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Queue Length",
            int(features["queue_length"]),
        )

        st.metric(
            "Available Staff",
            int(features["staff_available"]),
        )

    with c2:

        st.metric(
            "Average Service Time",
            f"{features['avg_service_time']:.1f} min",
        )

        st.metric(
            "Appointments Next Hour",
            int(
                features[
                    "appointments_next_hour"
                ]
            ),
        )

    with c3:

        st.metric(
            "Recent Walk-ins",
            int(features["recent_walkins"]),
        )

        st.metric(
            "Complex Request Ratio",
            (
                f"{features['complex_request_ratio'] * 100:.0f}%"
            ),
        )


    # -----------------------------------------------------
    # SHOW WHERE THE FEATURES CAME FROM
    # -----------------------------------------------------

    with st.expander(
        "View operational data sources used"
    ):

        source_counts = []

        source_labels = {
            "token_queue":
                "Token Queue Records",

            "staff_roster":
                "Staff Roster Records",

            "appointments":
                "Appointment Records",

            "branch_visits":
                "Branch Visit Records",

            "service_requests":
                "Service Request Records",

            "customer_feedback":
                "Customer Feedback Records",
        }

        for key, label in source_labels.items():

            dataframe = operational_data.get(
                key,
                pd.DataFrame(),
            )

            branch_df = filter_branch_rows(
                dataframe,
                selected_branch_name,
            )

            source_counts.append({
                "Source": label,
                "Records Used": len(branch_df),
            })

        st.dataframe(
            pd.DataFrame(source_counts),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "In production these records would normally "
            "arrive automatically from branch operational systems."
        )


# =========================================================
# MANUAL SIMULATION MODE
# =========================================================

else:

    sample_branches = get_sample_branches()

    branch_names = [
        branch["branch_name"]
        for branch in sample_branches
    ]

    selected_branch_name = st.selectbox(
        "Select Starting Branch Scenario",
        branch_names,
    )

    selected_branch = next(
        branch
        for branch in sample_branches
        if (
            branch["branch_name"]
            == selected_branch_name
        )
    )


    st.markdown(
        "### Manual Scenario Inputs"
    )

    st.caption(
        "Change these values to simulate hypothetical "
        "branch operating conditions."
    )


    col1, col2 = st.columns(2)


    with col1:

        queue_length = st.number_input(
            "Queue Length",
            min_value=0,
            value=int(
                selected_branch[
                    "queue_length"
                ]
            ),
            step=1,
        )

        staff_available = st.number_input(
            "Available Staff",
            min_value=1,
            value=int(
                selected_branch[
                    "staff_available"
                ]
            ),
            step=1,
        )

        avg_service_time = st.number_input(
            "Average Service Time (minutes)",
            min_value=0.0,
            value=float(
                selected_branch[
                    "avg_service_time"
                ]
            ),
            step=0.5,
        )


    with col2:

        appointments_next_hour = (
            st.number_input(
                "Appointments Next Hour",
                min_value=0,
                value=int(
                    selected_branch[
                        "appointments_next_hour"
                    ]
                ),
                step=1,
            )
        )

        recent_walkins = st.number_input(
            "Recent Walk-ins",
            min_value=0,
            value=int(
                selected_branch[
                    "recent_walkins"
                ]
            ),
            step=1,
        )

        complex_request_ratio = st.slider(
            "Complex Request Ratio",
            min_value=0.0,
            max_value=1.0,
            value=float(
                selected_branch[
                    "complex_request_ratio"
                ]
            ),
            step=0.01,
        )


    features = {
        "queue_length":
            queue_length,

        "staff_available":
            staff_available,

        "avg_service_time":
            avg_service_time,

        "appointments_next_hour":
            appointments_next_hour,

        "recent_walkins":
            recent_walkins,

        "complex_request_ratio":
            complex_request_ratio,
    }


# =========================================================
# OPTIMIZER CONFIGURATION
# =========================================================

st.markdown("### Decision Target")

target_wait = st.number_input(
    "Desired Maximum Wait Time (minutes)",
    min_value=5.0,
    max_value=60.0,
    value=20.0,
    step=1.0,
    help=(
        "BranchPulse will test staffing scenarios "
        "to find the minimum tested staffing level "
        "that reaches this model-estimated target."
    ),
)


# =========================================================
# INPUT SIGNATURE
# =========================================================

current_signature = (
    data_mode,
    selected_branch_name,
    tuple(
        sorted(
            (
                key,
                float(value)
            )
            for key, value
            in features.items()
        )
    ),
    float(target_wait),
)


# =========================================================
# ANALYZE BUTTON
# =========================================================

st.divider()


if st.button(
    "Analyze Branch",
    type="primary",
    use_container_width=True,
):

    try:

        # -------------------------------------------------
        # 1. ML WAIT-TIME PREDICTION
        # -------------------------------------------------

        prediction = predict_pressure(
            model,
            features,
        )


        # -------------------------------------------------
        # 2. ALTERNATIVE BRANCHES
        # -------------------------------------------------

        if (
            data_mode == "Operational Data"
            and operational_data is not None
        ):

            alternatives = (
                build_operational_alternatives(
                    selected_branch_name
                )
            )

        else:

            alternatives = (
                build_manual_alternatives(
                    selected_branch_name
                )
            )


        # -------------------------------------------------
        # 3. BOTTLENECK ANALYSIS
        # -------------------------------------------------

        bottlenecks = identify_bottlenecks(
            features
        )


        # -------------------------------------------------
        # 4. RULE-BASED RECOMMENDATIONS
        # -------------------------------------------------

        recommendations = (
            generate_recommendations(
                features=features,
                prediction=prediction,
                alternatives=alternatives,
            )
        )


        # -------------------------------------------------
        # 5. STAFFING SCENARIO OPTIMIZER
        # -------------------------------------------------

        current_staff = int(
            features["staff_available"]
        )

        # Synthetic training data was generated with
        # staff_available between 1 and 8.
        #
        # Avoid intentionally simulating far outside
        # the range the model saw during training.
        max_training_staff = 8

        max_extra_staff = max(
            0,
            max_training_staff - current_staff,
        )

        optimization = (
            find_min_staff_for_target(
                model=model,
                features=features,
                predict_fn=predict_pressure,
                target_wait=target_wait,
                max_extra_staff=max_extra_staff,
            )
        )


        # -------------------------------------------------
        # 6. SERVICE-MIX ANALYSIS
        # -------------------------------------------------

        service_result = {
            "total_requests": 0,
            "complex_requests": 0,
            "simple_requests": 0,
            "digital_eligible": 0,
            "digital_eligible_ratio": 0.0,
            "top_service_types": [],
        }

        feedback = []

        if operational_data is not None:

            try:

                service_result = (
                    analyze_service_mix(
                        operational_data[
                            "service_requests"
                        ],
                        selected_branch_name,
                    )
                )

            except Exception:
                pass


            try:

                feedback = (
                    get_feedback_for_branch(
                        operational_data,
                        selected_branch_name,
                    )
                )

            except Exception:
                feedback = []


        # -------------------------------------------------
        # STORE ANALYSIS
        # -------------------------------------------------

        st.session_state[
            "branchpulse_analysis"
        ] = {

            "signature":
                current_signature,

            "data_mode":
                data_mode,

            "branch_name":
                selected_branch_name,

            "features":
                dict(features),

            "prediction":
                prediction,

            "alternatives":
                alternatives,

            "bottlenecks":
                bottlenecks,

            "recommendations":
                recommendations,

            "optimization":
                optimization,

            "service_analysis":
                service_result,

            "feedback":
                feedback,
        }


        # Clear old GenAI results whenever
        # branch conditions are re-analyzed.
        st.session_state.pop(
            "branchpulse_genai",
            None,
        )


    except Exception as exc:

        st.error(
            f"Unable to analyze branch: {exc}"
        )


# =========================================================
# DISPLAY ANALYSIS
# =========================================================

analysis = st.session_state.get(
    "branchpulse_analysis"
)


if analysis:

    if (
        analysis["signature"]
        != current_signature
    ):

        st.info(
            "The branch inputs have changed. "
            "Click **Analyze Branch** again "
            "to refresh the results."
        )

    else:

        prediction = analysis["prediction"]

        alternatives = analysis[
            "alternatives"
        ]

        bottlenecks = analysis[
            "bottlenecks"
        ]

        recommendations = analysis[
            "recommendations"
        ]

        optimization = analysis[
            "optimization"
        ]

        service_result = analysis[
            "service_analysis"
        ]

        feedback = analysis[
            "feedback"
        ]


        # =================================================
        # CORE ANALYSIS
        # =================================================

        st.divider()

        st.markdown(
            "## 2. Branch Analysis"
        )


        r1, r2, r3 = st.columns(3)


        with r1:

            st.metric(
                "Predicted Wait Time",
                (
                    f"{prediction['predicted_wait']:.1f} min"
                ),
            )


        with r2:

            st.metric(
                "Pressure Level",
                prediction["pressure"],
            )


        with r3:

            st.metric(
                "Available Staff",
                int(
                    analysis[
                        "features"
                    ][
                        "staff_available"
                    ]
                ),
            )


        pressure = prediction[
            "pressure"
        ]


        if pressure == "LOW":

            st.success(
                "Current model-estimated service "
                "pressure is LOW."
            )

        elif pressure == "MODERATE":

            st.info(
                "Current model-estimated service "
                "pressure is MODERATE."
            )

        elif pressure == "HIGH":

            st.warning(
                "Current model-estimated service "
                "pressure is HIGH."
            )

        else:

            st.error(
                "Current model-estimated service "
                "pressure is CRITICAL."
            )


        st.caption(
            "The prediction is produced by a "
            "Random Forest regression model trained "
            "on synthetic hackathon data. It is not "
            "a validated real-bank service-time forecast."
        )


        # =================================================
        # BOTTLENECKS
        # =================================================

        st.markdown(
            "### Operational Bottlenecks"
        )


        if bottlenecks:

            for bottleneck in bottlenecks:

                st.warning(
                    f"• {bottleneck}"
                )

        else:

            st.success(
                "No major operational bottlenecks "
                "were detected by the current rules."
            )


        # =================================================
        # RECOMMENDATIONS
        # =================================================

        st.markdown(
            "### Recommended Actions"
        )


        for index, recommendation in enumerate(
            recommendations,
            start=1,
        ):

            st.write(
                f"**{index}.** {recommendation}"
            )


        # =================================================
        # STAFFING OPTIMIZER
        # =================================================

        st.markdown(
            "## 3. Staffing Scenario Optimizer"
        )

        st.caption(
            "This tests hypothetical staffing inputs "
            "through the same trained model. "
            "These are model-estimated scenarios, "
            "not guaranteed causal outcomes."
        )


        o1, o2, o3, o4 = st.columns(4)


        with o1:

            st.metric(
                "Current Staff",
                optimization[
                    "current_staff"
                ],
            )


        with o2:

            st.metric(
                "Current Estimated Wait",
                (
                    f"{optimization['current_wait']:.1f} min"
                ),
            )


        with o3:

            st.metric(
                "Suggested Tested Staff",
                optimization[
                    "suggested_staff"
                ],
            )


        with o4:

            wait_difference = (
                optimization[
                    "suggested_wait"
                ]
                - optimization[
                    "current_wait"
                ]
            )

            st.metric(
                "Scenario Estimated Wait",
                (
                    f"{optimization['suggested_wait']:.1f} min"
                ),
                delta=(
                    f"{wait_difference:.1f} min"
                ),
            )


        if optimization["target_met"]:

            st.success(
                "The configured target was reached "
                "within the tested staffing range. "
                f"The minimum tested level was "
                f"{optimization['suggested_staff']} staff."
            )

        else:

            st.warning(
                "The target wait was not reached "
                "within the tested staffing range."
            )


        scenario_df = pd.DataFrame(
            optimization["scenarios"]
        )

        if not scenario_df.empty:

            scenario_df = scenario_df.rename(
                columns={
                    "staff":
                        "Staff",

                    "predicted_wait":
                        "Estimated Wait (min)",

                    "pressure":
                        "Pressure",
                }
            )

            st.dataframe(
                scenario_df,
                use_container_width=True,
                hide_index=True,
            )


        # =================================================
        # SERVICE DEMAND INTELLIGENCE
        # =================================================

        st.markdown(
            "## 4. Service Demand Intelligence"
        )


        if (
            service_result[
                "total_requests"
            ] > 0
        ):

            s1, s2, s3, s4 = st.columns(4)


            with s1:

                st.metric(
                    "Service Requests",
                    service_result[
                        "total_requests"
                    ],
                )


            with s2:

                st.metric(
                    "Complex Requests",
                    service_result[
                        "complex_requests"
                    ],
                )


            with s3:

                st.metric(
                    "Potentially Digital Eligible",
                    service_result[
                        "digital_eligible"
                    ],
                )


            with s4:

                st.metric(
                    "Digital-Eligible Share",
                    (
                        f"{service_result['digital_eligible_ratio'] * 100:.0f}%"
                    ),
                )


            top_services = service_result[
                "top_service_types"
            ]

            if top_services:

                st.write(
                    "**Top Service Types:** "
                    + ", ".join(top_services)
                )


            if (
                service_result[
                    "digital_eligible"
                ] > 0
            ):

                st.info(
                    f"{service_result['digital_eligible']} "
                    "of the current sample service requests "
                    "are marked as potentially suitable for "
                    "digital/self-service handling."
                )

        else:

            st.info(
                "No service-request records are "
                "available for this branch."
            )


        # =================================================
        # BRANCH NETWORK COMPARISON
        # =================================================

        st.markdown(
            "## 5. Branch Network Comparison"
        )


        comparison_rows = [{
            "Branch":
                selected_branch_name,

            "Estimated Wait (min)":
                round(
                    prediction[
                        "predicted_wait"
                    ],
                    1,
                ),

            "Pressure":
                prediction[
                    "pressure"
                ],

            "Type":
                "Current Branch",
        }]


        for alternative in alternatives:

            comparison_rows.append({
                "Branch":
                    alternative[
                        "branch_name"
                    ],

                "Estimated Wait (min)":
                    round(
                        alternative[
                            "predicted_wait"
                        ],
                        1,
                    ),

                "Pressure":
                    alternative[
                        "pressure"
                    ],

                "Type":
                    "Alternative",
            })


        comparison_df = pd.DataFrame(
            comparison_rows
        )


        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
        )


        if alternatives:

            best_alternative = min(
                alternatives,
                key=lambda item:
                    item["predicted_wait"],
            )

            if (
                prediction["pressure"]
                in {"HIGH", "CRITICAL"}
                and
                best_alternative[
                    "predicted_wait"
                ]
                + 5
                <= prediction[
                    "predicted_wait"
                ]
            ):

                st.info(
                    "Among the sample branch options, "
                    f"**{best_alternative['branch_name']}** "
                    "currently has the lowest model-estimated "
                    f"wait at approximately "
                    f"**{best_alternative['predicted_wait']:.1f} minutes**. "
                    "Eligible customers could be considered "
                    "for redirection where operationally appropriate."
                )


        # =================================================
        # CUSTOMER FEEDBACK
        # =================================================

        st.markdown(
            "## 6. Customer Experience Data"
        )


        if feedback:

            st.write(
                f"**{len(feedback)} anonymized "
                "sample feedback comments available.**"
            )

            with st.expander(
                "View Sample Customer Feedback"
            ):

                for comment in feedback:

                    st.write(
                        f"• {comment}"
                    )

        else:

            st.info(
                "No textual customer feedback is "
                "available for this branch."
            )


        # =================================================
        # GENAI
        # =================================================

        st.markdown(
            "## 7. GenAI Operations Intelligence"
        )


        if not os.getenv(
            "GEMINI_API_KEY"
        ):

            st.warning(
                "Gemini GenAI features are currently "
                "disabled because GEMINI_API_KEY is "
                "not available to this Streamlit process.\n\n"
                "The ML prediction, optimization, "
                "service analysis and recommendations "
                "continue to work independently."
            )

        else:

            if st.button(
                "Generate GenAI Insights",
                use_container_width=True,
            ):

                try:

                    with st.spinner(
                        "Analyzing customer experience "
                        "and preparing manager brief..."
                    ):

                        feedback_analysis = (
                            analyze_customer_feedback(
                                feedback
                            )
                        )

                        operations_brief = (
                            generate_operations_brief(
                                branch_name=
                                    selected_branch_name,

                                prediction=
                                    prediction,

                                bottlenecks=
                                    bottlenecks,

                                recommendations=
                                    recommendations,

                                optimization=
                                    optimization,

                                service_analysis=
                                    service_result,

                                feedback_analysis=
                                    feedback_analysis,

                                alternatives=
                                    alternatives,
                            )
                        )


                    st.session_state[
                        "branchpulse_genai"
                    ] = {

                        "signature":
                            current_signature,

                        "feedback_analysis":
                            feedback_analysis,

                        "operations_brief":
                            operations_brief,
                    }


                except Exception as exc:

                    st.error(
                        "GenAI analysis failed, but all "
                        "core BranchPulse modules remain "
                        f"functional.\n\nError: {exc}"
                    )


            genai_result = (
                st.session_state.get(
                    "branchpulse_genai"
                )
            )


            if (
                genai_result
                and
                genai_result[
                    "signature"
                ]
                == current_signature
            ):

                st.markdown(
                    "### AI Customer Feedback Analysis"
                )

                st.write(
                    genai_result[
                        "feedback_analysis"
                    ]
                )


                st.markdown(
                    "### AI Operations Brief"
                )

                st.write(
                    genai_result[
                        "operations_brief"
                    ]
                )


# =========================================================
# ARCHITECTURE EXPLANATION
# =========================================================

st.divider()


with st.expander(
    "How does BranchPulse AI work?"
):

    st.markdown(
        """
### 1. Operational Data

BranchPulse can derive the model inputs from sample:

- token queue records
- branch visit logs
- appointment records
- staff rosters
- service request records
- customer feedback

The manual mode is retained as a scenario simulator.

### 2. Predictive ML

A `RandomForestRegressor` estimates customer waiting time from:

- queue length
- available staff
- average service time
- appointments
- recent walk-ins
- request complexity

The estimated waiting time is classified as:

**LOW → MODERATE → HIGH → CRITICAL**

### 3. Explainable Decision Rules

Transparent rules identify branch-specific bottlenecks and generate
operational recommendations.

The Random Forest does **not** generate the recommendations.

### 4. Staffing Scenario Optimization

BranchPulse evaluates hypothetical staffing levels using the same
trained model and finds the minimum tested staff level that reaches
the configured waiting-time target where possible.

This is scenario analysis, not a causal guarantee.

### 5. Service-Mix Intelligence

Service request records are analyzed to understand:

- complex demand
- simple demand
- potentially digital-eligible requests
- common service categories

### 6. GenAI

Gemini is used where generative AI is appropriate:

- analyzing unstructured customer feedback
- synthesizing verified outputs into a concise operations brief

Gemini does not replace the numerical prediction model or the
transparent operational rules.

### 7. Decision Support

BranchPulse is designed to support the branch manager.

It does not autonomously make staffing or customer-service decisions.
"""
    )


st.caption(
    "Hackathon prototype using synthetic/sample operational data. "
    "A production system would require validated historical banking "
    "data, secure enterprise integrations, access controls and "
    "continuous model monitoring."
)