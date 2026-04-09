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

from civiclens.analytics import (
    build_executive_report,
    build_overview,
    enrich_predictions,
    format_department,
    top_keywords,
)
from civiclens.data import load_demo_data, validate_dataset
from civiclens.modeling import train_models
from civiclens.summarizer import summarize_text

st.set_page_config(page_title="CivicLens AI", page_icon="C", layout="wide")


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Manrope:wght@400;500;700&display=swap');

        :root {
            --ink: #112033;
            --muted: #5c6c7f;
            --panel: rgba(255, 255, 255, 0.84);
            --panel-strong: rgba(255, 255, 255, 0.94);
            --line: rgba(148, 163, 184, 0.28);
            --accent: #0f8b8d;
            --accent-soft: #dff5f2;
            --amber: #f4a261;
            --rose: #f28482;
        }

        html, body, [class*="css"] {
            font-family: 'Manrope', sans-serif;
            color: var(--ink);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, #ddf2ef 0%, rgba(221, 242, 239, 0.55) 22%, transparent 48%),
                radial-gradient(circle at top right, #fde7d1 0%, rgba(253, 231, 209, 0.55) 20%, transparent 42%),
                linear-gradient(180deg, #f7fafc 0%, #eef4f9 100%);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #102033 0%, #162845 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] * {
            color: #edf5ff;
        }

        h1, h2, h3, h4 {
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: -0.04em;
            color: var(--ink);
        }

        .hero-card {
            background: linear-gradient(135deg, rgba(17, 32, 51, 0.98) 0%, rgba(15, 139, 141, 0.9) 58%, rgba(244, 162, 97, 0.82) 100%);
            padding: 2.4rem;
            border-radius: 30px;
            color: #ffffff;
            box-shadow: 0 26px 60px rgba(17, 32, 51, 0.22);
            margin-bottom: 1.2rem;
        }

        .hero-card h1 {
            color: #ffffff;
            margin: 0.35rem 0 0.6rem 0;
            font-size: 3.4rem;
        }

        .hero-kicker {
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.78rem;
            opacity: 0.82;
        }

        .hero-copy {
            max-width: 900px;
            font-size: 1.08rem;
            line-height: 1.7;
            color: rgba(255, 255, 255, 0.9);
        }

        .hero-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.7rem;
            margin-top: 1.1rem;
        }

        .hero-pill {
            border: 1px solid rgba(255, 255, 255, 0.18);
            background: rgba(255, 255, 255, 0.12);
            color: #ffffff;
            padding: 0.45rem 0.75rem;
            border-radius: 999px;
            font-size: 0.92rem;
        }

        .metric-card {
            background: var(--panel);
            border: 1px solid rgba(255, 255, 255, 0.7);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            box-shadow: 0 16px 40px rgba(148, 163, 184, 0.16);
            min-height: 132px;
        }

        .metric-label {
            font-size: 0.86rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            margin-bottom: 0.55rem;
        }

        .metric-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2rem;
            line-height: 1.05;
            color: var(--ink);
            margin-bottom: 0.35rem;
        }

        .metric-detail {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.5;
        }

        .surface-card {
            background: var(--panel-strong);
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1.2rem;
            box-shadow: 0 16px 32px rgba(148, 163, 184, 0.12);
            margin-bottom: 1rem;
        }

        .panel-kicker {
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.78rem;
            margin-bottom: 0.4rem;
        }

        .panel-copy {
            color: var(--muted);
            line-height: 1.7;
            font-size: 0.98rem;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 0.85rem;
        }

        .signal-chip {
            background: var(--accent-soft);
            border: 1px solid rgba(15, 139, 141, 0.12);
            color: var(--ink);
            padding: 0.42rem 0.72rem;
            border-radius: 999px;
            font-size: 0.9rem;
        }

        .sidebar-card {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 18px;
            padding: 1rem;
            margin-bottom: 1rem;
        }

        .sidebar-title {
            font-family: 'Space Grotesk', sans-serif;
            color: #ffffff;
            font-size: 1.2rem;
            margin-bottom: 0.35rem;
        }

        div[data-testid="stTabs"] button[data-baseweb="tab"] {
            font-size: 1rem;
            font-weight: 600;
            color: #526070;
        }

        div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
            color: var(--ink);
        }

        div[data-testid="stMetric"] {
            background: var(--panel-strong);
            border: 1px solid var(--line);
            padding: 0.8rem;
            border-radius: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def get_demo_data() -> pd.DataFrame:
    return load_demo_data()


@st.cache_resource
def get_demo_artifacts():
    return train_models(get_demo_data())


def render_metric_card(label: str, value: str, detail: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_surface_card(kicker: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="surface-card">
            <div class="panel-kicker">{kicker}</div>
            <h3>{title}</h3>
            <div class="panel-copy">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_row(items: list[str]) -> None:
    chips = "".join(f'<span class="signal-chip">{item}</span>' for item in items)
    st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)


def load_active_dataset(uploaded_file) -> tuple[pd.DataFrame, object, str]:
    if uploaded_file is None:
        return get_demo_data(), get_demo_artifacts(), "Demo dataset"

    try:
        uploaded_df = pd.read_csv(io.BytesIO(uploaded_file.read()))
        dataset = validate_dataset(uploaded_df)
        return dataset, train_models(dataset), "Uploaded dataset"
    except Exception as exc:
        st.sidebar.error(f"Could not process the uploaded CSV: {exc}")
        st.stop()


def make_prediction_frame(dataset: pd.DataFrame, artifacts) -> pd.DataFrame:
    predictions = artifacts.predict(dataset["complaint_text"].tolist())
    enriched = dataset.reset_index(drop=True).copy()
    for column in predictions.columns:
        if column != "complaint_text":
            enriched[column] = predictions[column]
    return enrich_predictions(enriched)


def build_dashboard_brief(dataset_name: str, overview: dict[str, object]) -> str:
    return (
        f"{dataset_name} currently contains {overview['total_cases']} complaints. "
        f"{overview['high_risk_cases']} items are marked high risk and "
        f"{overview['critical_cases']} are critical. "
        f"The heaviest projected load is on {overview['most_loaded_department']}, while the "
        f"average model confidence across routing and urgency is {overview['average_confidence']}%."
    )


def format_prediction_table(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    rename_map = {
        "complaint_text": "Complaint",
        "department": "Actual Department",
        "urgency": "Actual Urgency",
        "predicted_department": "Predicted Department",
        "predicted_urgency": "Predicted Urgency",
        "department_confidence": "Department Confidence",
        "urgency_confidence": "Urgency Confidence",
        "combined_confidence": "Combined Confidence",
        "impact_area": "Impact Area",
        "target_sla": "Target SLA",
        "priority_label": "Priority",
        "risk_score": "Risk Score",
        "recommended_action": "Recommended Action",
        "confidence_band": "Confidence Band",
    }
    for column in ["department", "predicted_department"]:
        if column in display.columns:
            display[column] = display[column].astype(str).map(format_department)
    for column in ["urgency", "predicted_urgency"]:
        if column in display.columns:
            display[column] = display[column].astype(str).str.title()
    for column in ["department_confidence", "urgency_confidence", "combined_confidence"]:
        if column in display.columns:
            display[column] = (display[column] * 100).round(1).astype(str) + "%"
    return display.rename(columns=rename_map)


def queue_download(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


inject_theme()

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-title">CivicLens AI</div>
            <div class="panel-copy">
                A polished ML and NLP workspace for routing complaints, flagging urgency, and turning raw text into an operational queue.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Upload complaint CSV", type=["csv"])
    st.caption("Required columns: complaint_text, department, urgency")
    st.caption("Use the built-in demo data if you want to explore the dashboard instantly.")

dataset, artifacts, dataset_name = load_active_dataset(uploaded_file)
prediction_frame = make_prediction_frame(dataset, artifacts)
overview = build_overview(prediction_frame)
executive_report = build_executive_report(prediction_frame)

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-title">Included Tools</div>
            <div class="panel-copy">
                Single-case triage, batch queue review, executive brief export, confidence scoring, SLA guidance, and searchable dataset inspection.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.download_button(
        "Download Executive Brief",
        data=executive_report,
        file_name="civiclens_executive_brief.txt",
        mime="text/plain",
        use_container_width=True,
    )
    st.download_button(
        "Download Full Queue CSV",
        data=queue_download(prediction_frame.sort_values("risk_score", ascending=False)),
        file_name="civiclens_queue.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-kicker">Operational ML and NLP workspace</div>
        <h1>CivicLens AI</h1>
        <div class="hero-copy">
            Turn unstructured complaints into routing decisions, urgency signals, analyst-ready queues, and leadership-ready reporting from a single dashboard.
        </div>
        <div class="hero-meta">
            <span class="hero-pill">Source: {dataset_name}</span>
            <span class="hero-pill">Most loaded: {overview['most_loaded_department']}</span>
            <span class="hero-pill">Average confidence: {overview['average_confidence']}%</span>
            <span class="hero-pill">High-risk cases: {overview['high_risk_cases']}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(build_dashboard_brief(dataset_name, overview))

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Command Center", "Case Studio", "Data Explorer", "Ops Console", "Model Quality"]
)

with tab1:
    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card("Incoming Cases", str(overview["total_cases"]), f"Live view from {dataset_name.lower()}.")
    with metric_cols[1]:
        render_metric_card(
            "Urgent Share",
            f"{overview['urgent_rate']}%",
            f"{overview['high_risk_cases']} complaints need accelerated response.",
        )
    with metric_cols[2]:
        render_metric_card(
            "Critical Cases",
            str(overview["critical_cases"]),
            "Immediate escalation candidates based on predicted urgency.",
        )
    with metric_cols[3]:
        render_metric_card(
            "Avg Confidence",
            f"{overview['average_confidence']}%",
            "Combined routing and urgency certainty from the classifier.",
        )

    left, right = st.columns([1.15, 0.85])
    department_df = pd.DataFrame(overview["department_distribution"])
    urgency_df = pd.DataFrame(overview["urgency_distribution"])
    action_df = pd.DataFrame(overview["action_board"])
    queue_df = pd.DataFrame(overview["top_risks"])

    with left:
        st.markdown("### Department Pressure Board")
        if not department_df.empty:
            pressure_chart = department_df.set_index("predicted_department")[["cases", "urgent_cases"]]
            pressure_chart.index = pressure_chart.index.map(format_department)
            st.bar_chart(pressure_chart, use_container_width=True)
            dept_display = department_df.copy()
            dept_display["predicted_department"] = dept_display["predicted_department"].map(format_department)
            dept_display = dept_display.rename(
                columns={
                    "predicted_department": "Department",
                    "cases": "Cases",
                    "urgent_cases": "Urgent Cases",
                    "avg_risk_score": "Avg Risk Score",
                }
            )
            st.dataframe(dept_display, use_container_width=True, hide_index=True)

        st.markdown("### Priority Queue")
        if not queue_df.empty:
            queue_display = format_prediction_table(queue_df)
            st.dataframe(queue_display, use_container_width=True, hide_index=True)
            st.download_button(
                "Download priority queue",
                data=queue_download(queue_df),
                file_name="civiclens_priority_queue.csv",
                mime="text/csv",
            )

    with right:
        render_surface_card(
            "Executive Summary",
            "Leadership Brief",
            executive_report.replace("\n", "<br>"),
        )

        st.markdown("### Recurring Signals")
        recurring_signals = [term.replace("_", " ") for term, _ in overview["keywords"][:8]]
        if recurring_signals:
            render_signal_row(recurring_signals)

        st.markdown("### Urgency Mix")
        if not urgency_df.empty:
            urgency_chart = urgency_df.set_index("predicted_urgency")
            urgency_chart.index = urgency_chart.index.str.title()
            st.bar_chart(urgency_chart, use_container_width=True)

        st.markdown("### Action Board")
        if not action_df.empty:
            action_display = action_df.copy()
            action_display["predicted_department"] = action_display["predicted_department"].map(
                format_department
            )
            action_display = action_display.rename(
                columns={
                    "predicted_department": "Department",
                    "urgent_cases": "Urgent Cases",
                    "avg_risk_score": "Avg Risk Score",
                    "avg_sla_hours": "Avg SLA (hours)",
                }
            )
            st.dataframe(action_display, use_container_width=True, hide_index=True)

with tab2:
    st.markdown("### Case Studio")
    st.caption("Analyze a fresh complaint and get routing, urgency, confidence, and response guidance in one view.")

    case_templates = {
        "Street safety incident": "Street lights near the girls' hostel have been out for a week and students feel unsafe returning after evening classes.",
        "Healthcare delay": "My father was admitted in pain but the registration system crashed and we waited hours without a bed assignment.",
        "Water disruption": "Water supply has been contaminated for two days and several families in our block reported stomach illness after using it.",
        "Custom input": "The city bus tracking app keeps showing wrong arrival times and commuters are missing work connections every morning.",
    }
    selected_case = st.selectbox("Quick scenario", list(case_templates.keys()))
    with st.form("case_form"):
        case_text = st.text_area(
            "Complaint narrative",
            value=case_templates[selected_case],
            height=220,
        )
        run_case = st.form_submit_button("Run Case Intelligence")

    if run_case:
        case_frame = enrich_predictions(artifacts.predict([case_text]))
        case = case_frame.iloc[0]
        case_summary = summarize_text(case_text, max_sentences=2)
        issue_signals = [term.replace("_", " ") for term, _ in top_keywords([case_text], limit=6)]

        result_cols = st.columns(4)
        with result_cols[0]:
            render_metric_card("Department", format_department(case["predicted_department"]), "Primary owner for the complaint.")
        with result_cols[1]:
            render_metric_card("Urgency", str(case["predicted_urgency"]).title(), case["priority_label"])
        with result_cols[2]:
            render_metric_card(
                "Dept Confidence",
                f"{case['department_confidence'] * 100:.1f}%",
                case["confidence_band"],
            )
        with result_cols[3]:
            render_metric_card("Target SLA", case["target_sla"], "Response window suggested by the urgency model.")

        detail_left, detail_right = st.columns([1.05, 0.95])
        with detail_left:
            render_surface_card("Summary", "Case Summary", case_summary)
            render_surface_card("Routing", "Recommended Action", str(case["recommended_action"]))
            st.markdown("### Detected Signals")
            if issue_signals:
                render_signal_row(issue_signals)

        with detail_right:
            render_surface_card(
                "Impact",
                "Service Impact Area",
                f"This complaint most closely maps to <strong>{str(case['impact_area']).title()}</strong> and should be tracked under the <strong>{format_department(case['predicted_department'])}</strong> queue.",
            )
            st.metric("Urgency Confidence", f"{case['urgency_confidence'] * 100:.1f}%")
            st.progress(float(case["urgency_confidence"]))
            st.metric("Combined Confidence", f"{case['combined_confidence'] * 100:.1f}%")
            st.progress(float(case["combined_confidence"]))

with tab3:
    st.markdown("### Data Explorer")
    explorer = prediction_frame.copy()

    filter_cols = st.columns(3)
    with filter_cols[0]:
        search_text = st.text_input("Search complaint text", placeholder="water, billing, safety...")
    with filter_cols[1]:
        department_filter = st.multiselect(
            "Actual department filter",
            options=sorted(dataset["department"].unique().tolist()),
            default=sorted(dataset["department"].unique().tolist()),
        )
    with filter_cols[2]:
        urgency_filter = st.multiselect(
            "Actual urgency filter",
            options=sorted(dataset["urgency"].unique().tolist()),
            default=sorted(dataset["urgency"].unique().tolist()),
        )

    if search_text:
        explorer = explorer[explorer["complaint_text"].str.contains(search_text, case=False, na=False)]
    explorer = explorer[explorer["department"].isin(department_filter)]
    explorer = explorer[explorer["urgency"].isin(urgency_filter)]

    department_agreement = (prediction_frame["department"] == prediction_frame["predicted_department"]).mean() * 100
    urgency_agreement = (prediction_frame["urgency"] == prediction_frame["predicted_urgency"]).mean() * 100
    avg_length = prediction_frame["complaint_text"].str.len().mean()

    profile_cols = st.columns(4)
    with profile_cols[0]:
        render_metric_card("Visible Rows", str(len(explorer)), "Rows after search and label filters.")
    with profile_cols[1]:
        render_metric_card("Avg Text Length", f"{avg_length:.0f}", "Average characters per complaint.")
    with profile_cols[2]:
        render_metric_card("Dept Agreement", f"{department_agreement:.1f}%", "Actual label vs prediction on the active dataset.")
    with profile_cols[3]:
        render_metric_card("Urgency Agreement", f"{urgency_agreement:.1f}%", "Useful for spotting where more training data is needed.")

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("### Actual Department Distribution")
        actual_department_counts = dataset["department"].value_counts().rename_axis("department").to_frame("cases")
        actual_department_counts.index = actual_department_counts.index.map(format_department)
        st.bar_chart(actual_department_counts, use_container_width=True)
    with chart_right:
        st.markdown("### Predicted Department Distribution")
        predicted_department_counts = (
            prediction_frame["predicted_department"].value_counts().rename_axis("department").to_frame("cases")
        )
        predicted_department_counts.index = predicted_department_counts.index.map(format_department)
        st.bar_chart(predicted_department_counts, use_container_width=True)

    st.markdown("### Filtered Records")
    explorer_display = format_prediction_table(
        explorer[
            [
                "complaint_text",
                "department",
                "urgency",
                "predicted_department",
                "predicted_urgency",
                "combined_confidence",
                "impact_area",
                "target_sla",
            ]
        ]
    )
    st.dataframe(explorer_display, use_container_width=True, hide_index=True)

with tab4:
    st.markdown("### Ops Console")
    ops = prediction_frame.copy()

    ops_cols = st.columns(4)
    with ops_cols[0]:
        predicted_department_filter = st.selectbox(
            "Predicted department",
            options=["all"] + sorted(prediction_frame["predicted_department"].unique().tolist()),
        )
    with ops_cols[1]:
        predicted_urgency_filter = st.multiselect(
            "Predicted urgency",
            options=sorted(prediction_frame["predicted_urgency"].unique().tolist()),
            default=sorted(prediction_frame["predicted_urgency"].unique().tolist()),
        )
    with ops_cols[2]:
        confidence_threshold = st.slider("Min confidence", 0.0, 1.0, 0.45, 0.05)
    with ops_cols[3]:
        max_rows = st.slider("Rows to show", 5, max(5, len(prediction_frame)), min(12, len(prediction_frame)))

    if predicted_department_filter != "all":
        ops = ops[ops["predicted_department"] == predicted_department_filter]
    ops = ops[ops["predicted_urgency"].isin(predicted_urgency_filter)]
    ops = ops[ops["combined_confidence"] >= confidence_threshold]

    if ops.empty:
        st.warning("No complaints match the current filters. Lower the confidence threshold or widen the urgency selection.")
    else:
        summary_cols = st.columns(4)
        with summary_cols[0]:
            render_metric_card("Visible Queue", str(len(ops)), "Filtered complaint count for review.")
        with summary_cols[1]:
            render_metric_card("Avg Risk", f"{ops['risk_score'].mean():.1f}", "Risk score blends urgency, length, and uncertainty.")
        with summary_cols[2]:
            render_metric_card(
                "Fast SLA Cases",
                str(int((ops["target_sla_hours"] <= 8).sum())),
                "Cases expected to move within the same working window.",
            )
        with summary_cols[3]:
            render_metric_card(
                "Needs Review",
                str(int((ops["combined_confidence"] < 0.6).sum())),
                "Low-confidence items that may need manual analyst validation.",
            )

        ops_left, ops_right = st.columns([0.95, 1.05])
        with ops_left:
            st.markdown("### Department vs Urgency Heatmap")
            heatmap = ops.pivot_table(
                index="predicted_department",
                columns="predicted_urgency",
                values="complaint_text",
                aggfunc="count",
                fill_value=0,
            )
            heatmap.index = heatmap.index.map(format_department)
            st.dataframe(heatmap, use_container_width=True)

            st.markdown("### Queue Brief")
            st.write(build_executive_report(ops))

        with ops_right:
            st.markdown("### Operational Queue")
            queue_columns = [
                "complaint_text",
                "predicted_department",
                "predicted_urgency",
                "combined_confidence",
                "impact_area",
                "target_sla",
                "priority_label",
                "risk_score",
                "recommended_action",
            ]
            queue_view = ops.sort_values(["risk_score", "combined_confidence"], ascending=[False, True]).head(max_rows)
            queue_display = format_prediction_table(queue_view[queue_columns])
            st.dataframe(queue_display, use_container_width=True, hide_index=True)

            download_col1, download_col2 = st.columns(2)
            with download_col1:
                st.download_button(
                    "Download filtered queue",
                    data=queue_download(queue_view[queue_columns]),
                    file_name="civiclens_filtered_queue.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with download_col2:
                st.download_button(
                    "Download filtered brief",
                    data=build_executive_report(ops),
                    file_name="civiclens_filtered_brief.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

with tab5:
    st.markdown("### Model Quality")
    st.info(
        "These metrics come from a holdout validation split on the active dataset. On the demo data, treat the scores as directional portfolio metrics rather than production-grade benchmarks."
    )

    metrics = artifacts.metrics
    metric_cols = st.columns(2)
    with metric_cols[0]:
        st.metric("Department Accuracy", metrics["department_accuracy"])
        department_report = pd.DataFrame(metrics["department_report"]).transpose().round(3)
        st.dataframe(department_report, use_container_width=True)
    with metric_cols[1]:
        st.metric("Urgency Accuracy", metrics["urgency_accuracy"])
        urgency_report = pd.DataFrame(metrics["urgency_report"]).transpose().round(3)
        st.dataframe(urgency_report, use_container_width=True)

    st.markdown("### Review Queue for Model Improvement")
    low_confidence_cases = prediction_frame.sort_values("combined_confidence").head(8)
    low_confidence_display = format_prediction_table(
        low_confidence_cases[
            [
                "complaint_text",
                "department",
                "urgency",
                "predicted_department",
                "predicted_urgency",
                "combined_confidence",
                "impact_area",
            ]
        ]
    )
    st.dataframe(low_confidence_display, use_container_width=True, hide_index=True)

    render_surface_card(
        "Roadmap",
        "High-Value Next Improvements",
        "Expand the training dataset, add richer complaint metadata such as location and timestamp, persist trained models, and track drift between uploaded data and the original training mix.",
    )
