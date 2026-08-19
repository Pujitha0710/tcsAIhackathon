
# 🏦 BranchPulse AI

### Intelligent Branch Service Load and Customer Experience Optimizer

BranchPulse AI is an AI-powered banking branch decision-support system developed for the **TCS AI Hackathon 2026**.

The platform helps branch managers understand current service pressure, estimate customer waiting time, identify operational bottlenecks, evaluate staffing scenarios, compare alternative branches, analyze service demand, and understand customer feedback.

It also includes a customer-facing guidance module that helps customers determine whether a service may be handled digitally or whether another lower-pressure branch may be a better option.

---

## 🎯 Problem

Bank branches often experience:

- Uneven customer traffic
- Long waiting times
- Staff overload
- Sudden walk-in demand
- Appointment surges
- Complex service requests
- Customers visiting branches for services that may be available digitally
- Uneven load across nearby branches

Traditional dashboards may show queue information, but they may not explain:

- Why a branch is overloaded
- What the manager should do
- Whether additional staff may help
- Whether customers can be redirected
- Whether some services can move to digital channels

BranchPulse AI addresses this through a combination of:

```text
Operational Data
      ↓
Machine Learning
      ↓
Pressure Classification
      ↓
Bottleneck Detection
      ↓
Explainable Recommendations
      ↓
Staffing Scenario Optimization
      ↓
Service / Branch Intelligence
      ↓
GenAI Insights
````

---

# 🚀 Main Features

## 1. Operational Data Ingestion

BranchPulse can derive branch conditions from sample operational CSV files instead of requiring all inputs to be manually entered.

The prototype uses:

```text
token_queue.csv
staff_roster.csv
branch_visits.csv
appointments.csv
service_requests.csv
customer_feedback.csv
```

These records are converted into the six ML features used by the prediction model.

---

## 2. Feature Engineering

For each branch, BranchPulse derives:

```text
queue_length
staff_available
avg_service_time
appointments_next_hour
recent_walkins
complex_request_ratio
```

Examples:

* `queue_length` → number of customers currently marked as waiting
* `staff_available` → number of staff currently available
* `avg_service_time` → average observed service duration
* `appointments_next_hour` → expected scheduled demand
* `recent_walkins` → recent unscheduled demand
* `complex_request_ratio` → proportion of requests marked as complex

---

## 3. Machine Learning Wait-Time Prediction

BranchPulse uses a:

```python
RandomForestRegressor
```

to estimate customer waiting time from current operational conditions.

The model is trained on synthetic hackathon data generated using branch-demand relationships involving:

* queue size
* staffing
* service duration
* appointments
* walk-ins
* request complexity

The Random Forest configuration uses multiple decision trees and combines their predictions to produce an estimated wait time.

The ML model predicts:

```text
Predicted Wait Time in Minutes
```

Example:

```text
Central Branch
Predicted Wait = 34.7 minutes
```

---

## 4. Pressure Classification

The predicted waiting time is converted into a manager-friendly service-pressure level.

```text
Predicted Wait < 10 min       → LOW

10 min to < 20 min            → MODERATE

20 min to < 35 min            → HIGH

35 min or more                → CRITICAL
```

The Random Forest predicts the waiting time.

The pressure label itself is generated using deterministic thresholds.

---

# 🔍 Operational Bottleneck Detection

BranchPulse uses transparent business rules to detect likely operational problems.

Examples include:

```text
Large queue relative to available staff
Very limited staff availability
Long average service duration
High upcoming appointment volume
Recent walk-in surge
High proportion of complex requests
```

Example:

```text
Queue Length = 30
Available Staff = 3

Queue / Staff = 10
```

This can trigger a:

```text
Large queue relative to available staff
```

bottleneck.

The bottleneck system is rule-based so that the manager can clearly understand why a condition was flagged.

---

# 💡 Explainable Recommendation Engine

Recommendations are generated using transparent operational rules.

The Random Forest does **not** directly generate recommendations.

Examples:

### Queue / Staff Imbalance

Possible recommendation:

```text
Consider assigning additional staff or opening another
service counter where operationally possible.
```

### Low Staff Availability

Possible recommendation:

```text
Review whether additional staff can temporarily support
customer-facing operations.
```

### Long Service Duration

Possible recommendation:

```text
Consider separating simple and complex requests to improve
service flow.
```

### Appointment Surge

Possible recommendation:

```text
Prepare capacity for upcoming booked appointments.
```

### Walk-In Surge

Possible recommendation:

```text
Triage walk-ins and guide eligible simple requests toward
digital channels.
```

### Complex Request Load

Possible recommendation:

```text
Consider allocating experienced staff or dedicated capacity
for complex requests.
```

This separation gives BranchPulse a hybrid architecture:

```text
Machine Learning
→ predicts waiting time

Rules
→ explain operational bottlenecks

Rules
→ generate transparent recommendations
```

---

# 👥 Staffing Scenario Optimizer

BranchPulse includes a staffing scenario optimizer.

The manager selects a desired maximum waiting time.

Example:

```text
Current Staff = 3
Target Wait = 20 minutes
```

The optimizer evaluates hypothetical staffing scenarios through the same trained ML model.

Conceptually:

```text
3 staff → estimated wait
4 staff → estimated wait
5 staff → estimated wait
6 staff → estimated wait
```

The system then identifies the minimum tested staffing level that reaches the selected target where possible.

Important:

This is a **model-based scenario simulation**, not a guarantee that adding a certain number of employees will cause the exact predicted reduction.

---

# 📊 Service Demand Intelligence

BranchPulse analyzes service-request records for each branch.

It calculates:

```text
Total service requests
Complex requests
Simple requests
Potentially digital-eligible requests
Digital-eligible ratio
Most common service types
```

This helps identify whether some physical branch demand could potentially be shifted toward digital or self-service channels.

Digital eligibility is based on predefined sample metadata and is not decided by the ML model or Gemini.

---

# 🏬 Branch Network Comparison

BranchPulse predicts waiting time for alternative sample branches using the same Random Forest model.

Example:

```text
Central Branch     38 min   CRITICAL
Tech Park Branch   24 min   HIGH
Lake View Branch   14 min   MODERATE
```

When the current branch is under high pressure, the manager can compare the operational state of other branches.

The prototype is designed to support branch-load balancing.

A production implementation could additionally include:

```text
Travel time
Distance
Opening hours
Real-time service availability
Appointment availability
```

---

# 👤 Customer Guidance

BranchPulse includes a customer-facing module.

The customer selects:

```text
Required Service
Preferred / Current Branch
```

The system checks whether the selected service is marked as potentially digital eligible.

Example:

```text
Account Statement
        ↓
Digital eligible
        ↓
Consider authorized digital / self-service channel
```

For services that require branch-based assistance:

```text
Loan Consultation
        ↓
Compare sample branches
        ↓
Predict branch wait
        ↓
Rank lower-pressure branch options
```

This helps reduce unnecessary physical branch visits while also improving the customer experience.

---

# 🤖 GenAI Customer Feedback Analysis

BranchPulse uses Gemini for tasks that involve unstructured language.

Customer feedback may contain comments such as:

```text
"The queue was very long."

"Staff were helpful but I waited too long."

"Only a few counters seemed available."
```

Gemini can analyze the anonymized comments and produce:

```text
Overall sentiment
Recurring complaints
Positive themes
Operational signals
```

Gemini is **not used to predict waiting time**.

The structured numerical prediction is handled by the Random Forest model.

---

# 🧠 GenAI Operations Brief

BranchPulse can also generate a concise operations brief for the branch manager.

Gemini receives already-generated system outputs such as:

```text
Predicted wait
Pressure level
Detected bottlenecks
Recommendations
Staffing scenario
Service analysis
Feedback analysis
Alternative branches
```

It then converts those results into a readable management summary.

The GenAI layer is intentionally downstream of the core prediction and rule-based logic.

This prevents the LLM from becoming the source of numerical predictions or arbitrary operational decisions.

---

# 🔐 API Key Handling

The Gemini API key is read using an environment variable:

```python
os.getenv("GEMINI_API_KEY")
```

The API key is not stored directly in the source code.

Example PowerShell setup:

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

Sensitive configuration files should remain excluded from Git using `.gitignore`.

---

# 🧠 BranchPulse Architecture

```text
                  OPERATIONAL DATA

 Token Queue ────────────────────────┐
 Staff Roster ───────────────────────┤
 Branch Visits ──────────────────────┤
 Appointments ───────────────────────┤
 Service Requests ───────────────────┤
 Customer Feedback ──────────────────┘
                     ↓
                Data Ingestion
                     ↓
              Feature Engineering
                     ↓
              Six ML Features
                     ↓
          Random Forest Regression
                     ↓
             Predicted Wait
                     ↓
           Pressure Classification
                     ↓
            Bottleneck Detection
                     ↓
         Explainable Recommendations
                     ↓
         Staffing Scenario Optimizer
                     ↓
        Branch / Service Intelligence
                     ↓
               Gemini GenAI
                     ↓
       Feedback + Operations Summary
                     ↓
               Branch Manager
```

Customer flow:

```text
Customer
   ↓
Select Service
   ↓
Digital Eligibility
   ↓
Digital Option / Branch Requirement
   ↓
Compatible Sample Branches
   ↓
ML Wait Prediction
   ↓
Customer Guidance
```

---

# 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Random Forest Regression**
* **Google Gemini API**
* **Synthetic operational data**

---

# 📁 Project Structure

The project structure may look similar to:

```text
tcsAIhackathon/
│
├── 1_Branch_Operations.py
│
├── customer_guidance.py
│
├── model.py
│
├── data_utils.py
│
├── ingestion.py
│
├── recommendations.py
│
├── optimizer.py
│
├── service_analysis.py
│
├── genai_layer.py
│
├── seed_data.py
│
├── requirements.txt
├── README.md
├── .gitignore
│
├── pages/
│   └── 2_Customer_Guidance.py
│
└── data/
    ├── token_queue.csv
    ├── branch_visits.csv
    ├── staff_roster.csv
    ├── appointments.csv
    ├── service_requests.csv
    └── customer_feedback.csv
```

Adjust filenames if your final project structure differs.

---

# ⚙️ Running the Project

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/tcsAIhackathon.git
```

Move into the project directory:

```bash
cd tcsAIhackathon
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If Gemini features are required, set the API key.

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

Then start the Streamlit application:

```powershell
python -m streamlit run 1_Branch_Operations.py
```

Streamlit will provide navigation between:

```text
Branch Operations
Customer Guidance
```

---

# 🧪 Synthetic Data Disclaimer

The current ML model is trained using synthetic hackathon data.

The purpose of the synthetic dataset is to demonstrate the architecture and workflow when real confidential banking data is unavailable.

Therefore:

> The current prototype does not claim validated predictive accuracy for real banking branches.

A production version should replace the synthetic training target with observed historical waiting-time data.

---

# ⚠️ Current Limitations

BranchPulse AI is a hackathon prototype.

Current limitations include:

* Synthetic ML training data
* Sample operational CSV data instead of real-time banking feeds
* Prototype-defined bottleneck thresholds
* Limited number of ML features
* No full time-series demand forecasting
* No live travel-time integration
* No authoritative bank service catalogue
* Staffing optimization is scenario-based rather than causal
* Digital eligibility is predefined metadata
* External Gemini dependency for GenAI features
* No enterprise authentication or banking-system integration

---

# 🔮 Future Scope

A production BranchPulse platform could add:

```text
Real-time queue integrations
Historical branch data
Time-of-day demand forecasting
Day-of-week and seasonal features
Salary-day demand patterns
Real staff scheduling systems
Live branch maps
Travel-time-aware redirection
Appointment booking
Real digital queue tokens
Enterprise authentication
Model monitoring
Model retraining
SHAP-based local explanations
Regional branch-network optimization
```

---

# 💼 Business Value

BranchPulse AI is designed to help banking teams:

* Reduce customer waiting time
* Anticipate branch congestion
* Improve staff allocation
* Identify operational bottlenecks
* Support explainable operational decisions
* Shift eligible services toward digital channels
* Balance demand across branches
* Better understand customer feedback
* Improve customer experience

---

# 🏆 TCS AI Hackathon 2026

Developed for the challenge:

## Intelligent Branch Service Load and Customer Experience Optimizer

BranchPulse follows a simple design principle:

> **ML predicts. Rules explain and recommend. Scenario analysis evaluates interventions. GenAI understands and summarizes language. Humans make the final decision.**

---

## 👥 Team

Developed as part of the TCS AI Hackathon 2026.

```

This is the README for **our actual original build**:

**Random Forest + 6 features + operational CSV ingestion + bottleneck rules + recommendation engine + staffing scenario optimizer + service analysis + branch comparison + Gemini + customer guidance.**

Do **not** put the LightGBM/31-features/30–60-minute forecasting material in this repository unless that code is actually present.
```
