from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from civiclens.analytics import build_overview
from civiclens.data import load_demo_data, validate_dataset
from civiclens.modeling import train_models
from civiclens.summarizer import summarize_text

st.set_page_config(page_title="CivicLens AI", layout="wide")


@st.cache_data
def get_demo_data() -> pd.DataFrame:
    return load_demo_data()


@st.cache_resource
def get_demo_artifacts():
    return train_models(get_demo_data())


def render_metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div style="padding:1rem;border-radius:16px;background:#f4f1ea;border:1px solid #d5cfc4;">
            <div style="font-size:0.9rem;color:#5f5a52;">{label}</div>
            <div style="font-size:1.8rem;font-weight:700;color:#1f2a2e;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_prediction_frame(dataset: pd.DataFrame, artifacts) -> pd.DataFrame:
    predictions = artifacts.predict(dataset["complaint_text"].tolist())
    enriched = dataset.copy()
    enriched["predicted_department"] = predictions["predicted_department"]
    enriched["predicted_urgency"] = predictions["predicted_urgency"]
    return enriched


st.title("CivicLens AI")
st.caption("A deployable ML + NLP platform for complaint triage, risk detection, and issue intelligence.")

with st.sidebar:
    st.header("Dataset")
    uploaded_file = st.file_uploader("Upload complaint CSV", type=["csv"])
    st.markdown(
        "Required columns: `complaint_text`, `department`, `urgency`. "
        "If you do not upload a file, the app uses the built-in demo dataset."
    )

if uploaded_file is not None:
    uploaded_df = pd.read_csv(io.BytesIO(uploaded_file.read()))
    dataset = validate_dataset(uploaded_df)
    artifacts = train_models(dataset)
    dataset_name = "Uploaded dataset"
else:
    dataset = get_demo_data()
    artifacts = get_demo_artifacts()
    dataset_name = "Demo dataset"

prediction_frame = make_prediction_frame(dataset, artifacts)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Predict", "Data Explorer", "Operations View", "Model Quality"]
)

with tab1:
    overview = build_overview(prediction_frame)

    st.subheader(dataset_name)
    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("Total Cases", str(overview["total_cases"]))
    with c2:
        render_metric_card("High Risk Cases", str(overview["high_risk_cases"]))
    with c3:
        render_metric_card("Average Risk Score", str(overview["average_risk_score"]))

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("### Department Load")
        st.dataframe(pd.DataFrame(overview["department_distribution"]), use_container_width=True)

        st.markdown("### Top Risk Complaints")
        st.dataframe(pd.DataFrame(overview["top_risks"]), use_container_width=True)
    with right:
        st.markdown("### Executive Summary")
        st.write(overview["executive_summary"])

        st.markdown("### Frequent Keywords")
        st.dataframe(
            pd.DataFrame(overview["keywords"], columns=["keyword", "count"]),
            use_container_width=True,
        )

with tab2:
    st.subheader("Analyze a New Complaint")
    text = st.text_area(
        "Complaint text",
        value="Street lights near the school have been out for five days and residents feel unsafe walking home after sunset.",
        height=180,
    )
    if st.button("Run NLP Analysis"):
        prediction = artifacts.predict([text]).iloc[0]
        st.write(
            {
                "predicted_department": prediction["predicted_department"],
                "predicted_urgency": prediction["predicted_urgency"],
                "summary": summarize_text(text, max_sentences=2),
            }
        )

with tab3:
    st.subheader("Dataset Explorer")
    explorer_col1, explorer_col2 = st.columns(2)
    with explorer_col1:
        st.markdown("### Sample Records")
        st.dataframe(dataset.head(15), use_container_width=True)
    with explorer_col2:
        st.markdown("### Dataset Profile")
        st.write(
            {
                "rows": int(len(dataset)),
                "departments": int(dataset["department"].nunique()),
                "urgency_levels": int(dataset["urgency"].nunique()),
                "avg_text_length": round(dataset["complaint_text"].str.len().mean(), 1),
            }
        )

        dept_counts = dataset["department"].value_counts().rename_axis("department").reset_index(name="count")
        urgency_counts = dataset["urgency"].value_counts().rename_axis("urgency").reset_index(name="count")
        st.markdown("### Label Distribution")
        dist_col1, dist_col2 = st.columns(2)
        with dist_col1:
            st.bar_chart(dept_counts.set_index("department"))
        with dist_col2:
            st.bar_chart(urgency_counts.set_index("urgency"))

with tab4:
    st.subheader("Operations View")
    focus_department = st.selectbox(
        "Filter by predicted department",
        options=["all"] + sorted(prediction_frame["predicted_department"].unique().tolist()),
    )
    filtered = prediction_frame.copy()
    if focus_department != "all":
        filtered = filtered[filtered["predicted_department"] == focus_department]

    urgency_filter = st.multiselect(
        "Urgency filter",
        options=sorted(prediction_frame["predicted_urgency"].unique().tolist()),
        default=sorted(prediction_frame["predicted_urgency"].unique().tolist()),
    )
    filtered = filtered[filtered["predicted_urgency"].isin(urgency_filter)]

    ops_col1, ops_col2 = st.columns([1, 1.4])
    with ops_col1:
        st.markdown("### Workload Snapshot")
        st.write(
            {
                "visible_cases": int(len(filtered)),
                "high_or_critical": int(
                    len(filtered[filtered["predicted_urgency"].isin(["high", "critical"])])
                ),
            }
        )
        st.markdown("### Department Queue")
        st.bar_chart(
            filtered["predicted_department"].value_counts().rename_axis("department").to_frame("cases")
        )
    with ops_col2:
        st.markdown("### Batch Analysis Table")
        st.dataframe(
            filtered[
                [
                    "complaint_text",
                    "department",
                    "urgency",
                    "predicted_department",
                    "predicted_urgency",
                ]
            ],
            use_container_width=True,
        )

        if not filtered.empty:
            st.markdown("### Focus Summary")
            st.write(summarize_text(" ".join(filtered["complaint_text"].head(10).tolist()), max_sentences=3))

with tab5:
    st.subheader("Validation Metrics")
    metrics = artifacts.metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Department Accuracy", metrics["department_accuracy"])
        department_report = pd.DataFrame(metrics["department_report"]).transpose()
        st.dataframe(department_report, use_container_width=True)
    with col2:
        st.metric("Urgency Accuracy", metrics["urgency_accuracy"])
        urgency_report = pd.DataFrame(metrics["urgency_report"]).transpose()
        st.dataframe(urgency_report, use_container_width=True)
