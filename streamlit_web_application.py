import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# Page Configuration
st.set_page_config(
    page_title="SAD Predictive System - Student Service Requests",
    page_icon="📊",
    layout="wide"
)

# Required Schema Definitions
REQUIRED_COLUMNS = [
    "Request_ID", "Service_Type", "Requester_Type", "Assigned_Office", 
    "Queue_Length", "SLA_Days", "Processing_Days", "Satisfaction_Score"
]

# Shared Utility Functions & Caching
@st.cache_data
def load_and_preprocess_data(file_source):
    """Loads and validates CSV dataset with midterm feature engineering logic."""
    try:
        if isinstance(file_source, str):
            if not os.path.exists(file_source):
                return None
            df = pd.read_csv(file_source)
        else:
            df = pd.read_csv(file_source)
    except Exception as e:
        st.error(f"Error reading dataset: {str(e)}")
        return None

    # Schema Validation
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        st.error(f"Schema Validation Error: Missing required columns: {', '.join(missing_cols)}")
        return None

    # Midterm Target Creation Logic: SLA_Breached = 1 if Processing_Days > SLA_Days else 0
    df['SLA_Breached'] = df.apply(
        lambda row: 1 if pd.notnull(row['Processing_Days']) and row['Processing_Days'] > row['SLA_Days'] else 0, 
        axis=1
    )

    # Queue Binning Logic: Low (0-5), Medium (6-15), High (>15)
    df['Queue_Category'] = pd.cut(
        df['Queue_Length'], 
        bins=[-1, 5, 15, 999], 
        labels=['Low', 'Medium', 'High']
    )

    return df

@st.cache_resource
def load_trained_pipeline():
    """Loads the pre-fitted joblib machine learning pipeline."""
    model_path = "models/trained_model.joblib"
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            return None
    return None

# Navigation & Global Data State
st.sidebar.title("📌 SAD Navigation")
menu = st.sidebar.selectbox(
    "Select Module", 
    ["Home", "1. Dataset Upload", "2. Data Analysis (EDA)", "3. Predictive Modeling", "4. Decision Support Dashboard", "5. SAD System Design"]
)

# Load base default data if available
base_data_path = "data/student_service_requests.csv"
if 'df_data' not in st.session_state:
    st.session_state['df_data'] = load_and_preprocess_data(base_data_path)

df = st.session_state['df_data']


# MODULE 0: HOME
if menu == "Home":
    st.title("🎓 Student Service Request Monitoring System")
    st.markdown("### Systems Analysis & Design (SAD) — Midterm Streamlit Integration")
    st.info("Course: Systems Analysis and Design (BS Information Systems) | Section B")
    
    st.markdown("""
    #### Project Context & Core Objective
    This web application operationalizes the analytical findings, preprocessing steps, and predictive model developed during the **Midterm Laboratory Examination**. It converts static data mining models into an interactive, human-in-the-loop decision-support tool.

    #### System Capabilities:
    * **Data Quality & Validation**: Ingests service transaction logs and flags schema anomalies.
    * **Exploratory Bottleneck Analysis**: Identifies delay-prone offices and high-backlog queues.
    * **Submission-Time SLA Prediction**: Evaluates SLA breach risk without future data leakage.
    * **Decision Support Queue**: Empowers university staff to reassign resources dynamically.
    * **SAD Integration**: Connects system features to empirical requirements and UML use cases.

    ---
    *Developed by: Kristheal Sofia Rugas (AY 2026-2027)*
    """)


# MODULE 1: DATASET UPLOAD & VALIDATION
elif menu == "1. Dataset Upload":
    st.header("1. Dataset Upload & Schema Validation")
    st.markdown("Upload a student service request dataset (`.csv`) to validate its schema against midterm standard fields.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        processed_df = load_and_preprocess_data(uploaded_file)
        if processed_df is not None:
            st.session_state['df_data'] = processed_df
            df = processed_df
            st.success("✅ Dataset uploaded and schema validated successfully!")

    if df is not None:
        st.subheader("Data Overview Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records ($N$)", len(df))
        col2.metric("Completed Requests", len(df.dropna(subset=['Processing_Days'])))
        col3.metric("Open / Pending Requests", df['Processing_Days'].isnull().sum())
        col4.metric("Total Null Cells", df.isnull().sum().sum())

        st.subheader("Data Preview (First 20 Records)")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Data Quality & Null Summary")
        null_df = pd.DataFrame({
            "Field Name": df.columns,
            "Data Type": [str(df[col].dtype) for col in df.columns],
            "Null Count": df.isnull().sum().values,
            "Null Percentage (%)": (df.isnull().sum().values / len(df) * 100).round(2)
        })
        st.dataframe(null_df, use_container_width=True)
    else:
        st.warning("No valid dataset loaded. Please upload a matching `student_service_requests.csv` file.")


# MODULE 2: EXPLORATORY DATA ANALYSIS (EDA)
elif menu == "2. Data Analysis (EDA)":
    st.header("2. Exploratory Data Analysis & Bottleneck Identification")

    if df is None:
        st.warning("⚠️ No dataset loaded. Please upload a dataset in Module 1.")
    else:
        completed_df = df.dropna(subset=['Processing_Days']).copy()

        st.subheader("Interactive Bottleneck Metrics")
        
        # Chart 1: Volume by Service Type
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Chart 1: Total Volume by Service Type**")
            vol_df = df['Service_Type'].value_counts().reset_index()
            vol_df.columns = ['Service_Type', 'Count']
            fig1 = px.bar(
                vol_df, x='Service_Type', y='Count', 
                color='Count', color_continuous_scale='Blues',
                labels={'Service_Type': 'Service Type', 'Count': 'Request Volume'}
            )
            st.plotly_chart(fig1, use_container_width=True)

        # Chart 2: SLA Breach Rate by Assigned Office
        with col2:
            st.markdown("**Chart 2: SLA Breach Rate by Assigned Office (%)**")
            office_breach = completed_df.groupby('Assigned_Office')['SLA_Breached'].mean().reset_index()
            office_breach['Breach_Rate'] = (office_breach['SLA_Breached'] * 100).round(1)
            fig2 = px.bar(
                office_breach, x='Assigned_Office', y='Breach_Rate', text='Breach_Rate',
                color='Breach_Rate', color_continuous_scale='Reds',
                labels={'Assigned_Office': 'Assigned Office', 'Breach_Rate': 'Breach Rate (%)'}
            )
            fig2.update_traces(texttemplate='%{text}%', textposition='outside')
            st.plotly_chart(fig2, use_container_width=True)

        # Chart 3: Queue Category Risk Distribution
        st.markdown("**Chart 3: SLA Breach Risk by Backlog Queue Category**")
        queue_breach = completed_df.groupby('Queue_Category')['SLA_Breached'].mean().reset_index()
        queue_breach['Breach_Rate'] = (queue_breach['SLA_Breached'] * 100).round(1)
        fig3 = px.line(
            queue_breach, x='Queue_Category', y='Breach_Rate', markers=True,
            labels={'Queue_Category': 'Queue Category (Backlog)', 'Breach_Rate': 'Breach Risk (%)'}
        )
        st.plotly_chart(fig3, use_container_width=True)


# MODULE 3: PREDICTIVE MODELING
elif menu == "3. Predictive Modeling":
    st.header("3. Predictive Inference & Model Evaluation")
    st.markdown("Evaluate SLA breach risk for incoming requests without outcome leakage.")

    # Model Performance Summary Panel
    st.subheader("Midterm Model Performance Baseline (Decision Tree Pipeline)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", "68.2%")
    m2.metric("Precision (Breach=1)", "65.1%")
    m3.metric("Recall (Breach=1)", "82.4%")
    m4.metric("F1-Score", "0.727")

    st.markdown("---")
    st.subheader("Run Interactive Risk Prediction")

    pipeline = load_trained_pipeline()

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            service_type = st.selectbox("Service Type", ["Records", "Scholarship Inquiry", "Certification", "Clearance", "ID Renewal", "Transcripts"])
            requester_type = st.selectbox("Requester Type", ["Student", "Faculty", "Staff"])
            assigned_office = st.selectbox("Assigned Office", ["Scholarship Office", "Finance", "Registrar", "Deans Office", "IT Services"])
        with col2:
            queue_length = st.number_input("Queue Length (Open Backlog)", min_value=0, max_value=100, value=12)
            sla_days = st.number_input("SLA Target (Days Allowed)", min_value=1, max_value=30, value=5)
            docs_complete = st.selectbox("Required Documents Complete?", [1, 0], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")

        submit_btn = st.form_submit_button("Predict SLA Risk Flag")

    if submit_btn:
        # Construct input vector matching exact training feature names
        input_data = pd.DataFrame([{
            "Service_Type": service_type,
            "Requester_Type": requester_type,
            "Assigned_Office": assigned_office,
            "Queue_Length": queue_length,
            "SLA_Days": sla_days,
            "Required_Documents_Complete": docs_complete
        }])

        st.subheader("Prediction Inference Result")

        if pipeline is not None:
            pred_class = pipeline.predict(input_data)[0]
            pred_proba = pipeline.predict_proba(input_data)[0] if hasattr(pipeline, "predict_proba") else [0.5, 0.5]
            
            if pred_class == 1:
                st.error(f"⚠️ HIGH RISK OF SLA BREACH (Predicted Class: 1 | Risk Score: {pred_proba[1]*100:.1f}%)")
                st.warning("Recommended Action: Mark for fast-track routing or reassign administrative workload.")
            else:
                st.success(f"✅ LOW RISK / ON TIME (Predicted Class: 0 | Confidence: {pred_proba[0]*100:.1f}%)")
                st.info("Recommended Action: Process under standard operational workflow.")
        else:
            # Demonstration Fallback Logic aligned with midterm decision tree importances
            st.info("ℹ️ Running Rule-Engine Fallback (Joblib Pipeline model file not found in `/models`).")
            is_high_risk = 1 if (queue_length > 15 or (assigned_office == "Scholarship Office" and sla_days <= 3) or docs_complete == 0) else 0
            if is_high_risk == 1:
                st.error("⚠️ HIGH RISK OF SLA BREACH (Predicted Class: 1 | Rule Engine)")
                st.warning("Recommended Action: Priority fast-track review needed.")
            else:
                st.success("✅ LOW RISK / ON TIME (Predicted Class: 0 | Rule Engine)")
                st.info("Recommended Action: Standard operational processing.")


# MODULE 4: DECISION SUPPORT DASHBOARD
elif menu == "4. Decision Support Dashboard":
    st.header("4. Administrative Decision-Support Dashboard")
    st.markdown("Real-time operational queue for university staff review.")

    if df is None:
        st.warning("⚠️ No dataset loaded. Upload data in Module 1.")
    else:
        # High-level KPIs matching wireframe
        total_vol = len(df)
        completed_df = df.dropna(subset=['Processing_Days'])
        completed_cnt = len(completed_df)
        overall_breach_rate = (completed_df['SLA_Breached'].mean() * 100) if completed_cnt > 0 else 0.0
        pending_flagged = len(df[df['Processing_Days'].isnull() & (df['Queue_Length'] > 10)])

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("400 TOTAL VOLUME", total_vol)
        k2.metric("327 COMPLETED", completed_cnt)
        k3.metric("51.1% OVERALL BREACH", f"{overall_breach_rate:.1f}%")
        k4.metric("PENDING FLAGGED", pending_flagged)

        st.markdown("---")
        st.subheader("Priority Service Request Review Queue")

        # Operational Review Queue Construction
        open_requests = df[df['Processing_Days'].isnull()].copy()

        if not open_requests.empty:
            open_requests['PREDICTED BREACH FLAG'] = open_requests.apply(
                lambda r: "HIGH RISK (1)" if (r['Queue_Length'] > 10 or r['Assigned_Office'] == "Scholarship Office") else "LOW RISK (0)", 
                axis=1
            )
            open_requests['ACTION REQUIRED'] = open_requests['PREDICTED BREACH FLAG'].apply(
                lambda f: "Reassign / Fast-track" if "HIGH" in f else "Standard Queue"
            )

            display_cols = [
                'Request_ID', 'Service_Type', 'Assigned_Office', 
                'Queue_Length', 'SLA_Days', 'PREDICTED BREACH FLAG', 'ACTION REQUIRED'
            ]
            st.dataframe(open_requests[display_cols].head(15), use_container_width=True)
        else:
            st.info("All requests currently processed.")

        st.warning("""
        **Human-in-the-Loop Safeguard Policy**: Predictions are advisory decision-support signals only. 
        High-risk flags trigger staff review and fast-tracking; they do **not** automatically reject or cancel student service requests.
        """)


# MODULE 5: SAD SYSTEM DESIGN
elif menu == "5. SAD System Design":
    st.header("5. Systems Analysis & Design (SAD) Architecture")

    st.subheader("Functional Requirements (Evidence-Based)")
    st.markdown("""
    * **FR-1**: The system shall display SLA breach performance grouped by assigned office (*Derived from 78.9% breach rate at Scholarship Office*).
    * **FR-2**: The system shall categorize queue status and highlight high-queue requests (*Derived from elevated breach risk in high backlogs*).
    * **FR-3**: The system shall generate an advisory delay-risk flag using submission-time classification (*Derived from Decision Tree pipeline*).
    * **FR-4**: The system shall strictly isolate incomplete post-service records from machine learning model training to avoid data leakage.
    * **FR-5**: The system shall present an administrative decision queue for authorized staff to record manual fast-tracking actions.
    """)

    st.subheader("Non-Functional Requirements")
    st.markdown("""
    * **NFR-1 (Performance)**: Prediction inference response time must be under 500ms per request.
    * **NFR-2 (Security)**: Student transactional records must be processed without transmitting PII externally.
    * **NFR-3 (Usability)**: Visual indicators (color-coded flags) must clearly distinguish between High-Risk (1) and Low-Risk (0) items.
    """)

    st.subheader("UML 2.5 Use Case Architecture")
    if os.path.exists("assets/use_case_diagram.png"):
        st.image("assets/use_case_diagram.png", caption="UML 2.5 Use Case Diagram - Student Service Request Monitoring System")
    else:
        st.info("Place your UML diagram image at `assets/use_case_diagram.png` to preview here.")

    st.markdown("""
    **Actor Responsibilities**:
    * **Administrator**: Reviews SLA overall performance dashboards and monitors systemic queue backlogs.
    * **Service Officer**: Reviews flagged high-risk requests, inspects predictor feature explanations, and authorizes reassignment.
    """)