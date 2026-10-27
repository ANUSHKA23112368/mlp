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
    build_geo_and_channel_overview,
    build_overview,
    enrich_predictions,
    format_department,
    top_keywords,
)
from civiclens.data import load_demo_data, validate_dataset
from civiclens.deduplication import (
    calculate_deduplication_metrics,
    find_incident_clusters,
    find_similar_complaints,
)
from civiclens.explainability import explain_prediction
from civiclens.modeling import (
    SUPPORTED_ALGORITHMS,
    benchmark_models,
    train_models,
)
from civiclens.summarizer import summarize_text
from civiclens.topics import discover_topics

# ── CivicShield Extensions ────────────────────────────────────────
from civicshield.security import (
    generate_demo_events, RuleEngine, AnomalyDetector,
    correlate_events, SecurityEvent,
)
from civicshield.ml import check_data_quality, hash_dataset

st.set_page_config(page_title="CivicShield AI", page_icon="🏛️", layout="wide")


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Manrope:wght@400;500;700&display=swap');

        :root {
            --ink: #112033;
            --muted: #5c6c7f;
            --panel: rgba(255, 255, 255, 0.88);
            --panel-strong: rgba(255, 255, 255, 0.96);
            --line: rgba(148, 163, 184, 0.28);
            --accent: #0f8b8d;
            --accent-soft: #dff5f2;
            --amber: #f4a261;
            --rose: #f28482;
            --purple: #8338ec;
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

        [data-testid="stFileUploaderDropzone"] {
            background: linear-gradient(180deg, rgba(61, 94, 140, 0.26) 0%, rgba(30, 47, 78, 0.48) 100%) !important;
            border: 1px dashed rgba(255, 255, 255, 0.22) !important;
            border-radius: 20px !important;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
        }

        [data-testid="stFileUploaderDropzone"] * {
            color: #f5fbff !important;
        }

        [data-testid="stFileUploaderDropzone"] button {
            background: rgba(10, 21, 38, 0.82) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.24) !important;
            border-radius: 14px !important;
        }

        [data-testid="stSidebar"] .stDownloadButton button {
            width: 100%;
            background: linear-gradient(135deg, #0f8b8d 0%, #2aa9a5 100%) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            border-radius: 14px !important;
            box-shadow: 0 10px 24px rgba(4, 12, 24, 0.22);
        }

        h1, h2, h3, h4 {
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: -0.04em;
            color: var(--ink);
        }

        .hero-card {
            background: linear-gradient(135deg, rgba(17, 32, 51, 0.98) 0%, rgba(15, 139, 141, 0.9) 58%, rgba(244, 162, 97, 0.82) 100%);
            padding: 2.2rem;
            border-radius: 28px;
            color: #ffffff;
            box-shadow: 0 24px 50px rgba(17, 32, 51, 0.20);
            margin-bottom: 1.2rem;
        }

        .hero-card h1 {
            color: #ffffff;
            margin: 0.35rem 0 0.5rem 0;
            font-size: 3.2rem;
        }

        .hero-kicker {
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.78rem;
            opacity: 0.84;
        }

        .hero-copy {
            max-width: 950px;
            font-size: 1.05rem;
            line-height: 1.65;
            color: rgba(255, 255, 255, 0.92);
        }

        .hero-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-top: 1rem;
        }

        .hero-pill {
            border: 1px solid rgba(255, 255, 255, 0.22);
            background: rgba(255, 255, 255, 0.14);
            color: #ffffff;
            padding: 0.42rem 0.75rem;
            border-radius: 999px;
            font-size: 0.90rem;
            font-weight: 500;
        }

        .metric-card {
            background: var(--panel);
            border: 1px solid rgba(255, 255, 255, 0.7);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            box-shadow: 0 14px 35px rgba(148, 163, 184, 0.15);
            min-height: 132px;
        }

        .metric-label {
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            margin-bottom: 0.45rem;
        }

        .metric-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2rem;
            line-height: 1.05;
            color: var(--ink);
            margin-bottom: 0.30rem;
        }

        .metric-detail {
            color: var(--muted);
            font-size: 0.90rem;
            line-height: 1.45;
        }

        .surface-card {
            background: var(--panel-strong);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1.2rem;
            box-shadow: 0 14px 30px rgba(148, 163, 184, 0.12);
            margin-bottom: 1rem;
        }

        .panel-kicker {
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.78rem;
            margin-bottom: 0.35rem;
            font-weight: 700;
        }

        .panel-copy {
            color: var(--muted);
            line-height: 1.65;
            font-size: 0.96rem;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 0.75rem;
        }

        .signal-chip {
            background: var(--accent-soft);
            border: 1px solid rgba(15, 139, 141, 0.18);
            color: var(--ink);
            padding: 0.38rem 0.70rem;
            border-radius: 999px;
            font-size: 0.88rem;
            font-weight: 500;
        }

        .cluster-box {
            background: #ffffff;
            border: 1px solid rgba(15, 139, 141, 0.25);
            border-left: 6px solid #0f8b8d;
            border-radius: 14px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.04);
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
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }

        .highlight-container {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 1.1rem;
            line-height: 1.8;
            font-size: 1.02rem;
            margin-top: 0.5rem;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def get_demo_data() -> pd.DataFrame:
    return load_demo_data()


@st.cache_resource
def get_cached_artifacts(algorithm: str = "logistic_regression"):
    return train_models(get_demo_data(), algorithm=algorithm)


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
            <h3 style="margin: 0.2rem 0 0.5rem 0;">{title}</h3>
            <div class="panel-copy">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_row(items: list[str]) -> None:
    chips = "".join(f'<span class="signal-chip">{item}</span>' for item in items)
    st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)


def load_active_dataset(uploaded_file, algorithm: str = "logistic_regression") -> tuple[pd.DataFrame, object, str]:
    if uploaded_file is None:
        return get_demo_data(), get_cached_artifacts(algorithm), "Demo dataset (150 complaints)"

    try:
        uploaded_df = pd.read_csv(io.BytesIO(uploaded_file.read()))
        dataset = validate_dataset(uploaded_df)
        return dataset, train_models(dataset, algorithm=algorithm), "Uploaded dataset"
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


def format_prediction_table(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    rename_map = {
        "complaint_text": "Complaint",
        "department": "Actual Department",
        "urgency": "Actual Urgency",
        "predicted_department": "Predicted Department",
        "predicted_urgency": "Predicted Urgency",
        "department_confidence": "Dept Confidence",
        "urgency_confidence": "Urg Confidence",
        "combined_confidence": "Combined Confidence",
        "impact_area": "Impact Area",
        "target_sla": "Target SLA",
        "priority_label": "Priority",
        "risk_score": "Risk Score",
        "frustration_score": "Frustration (0-100)",
        "escalation_risk": "Escalation Risk",
        "ward": "Ward / District",
        "channel": "Intake Channel",
        "timestamp": "Timestamp",
        "status": "Status",
        "recommended_action": "Recommended Action",
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

# ---------------------------------------------------------
# Sidebar Controls & Global Setup
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-title">🛡️ CivicShield AI v3.0</div>
            <div class="panel-copy">
                Intelligent Civic Operations & Cybersecurity Platform — Complaint Triage, Security Monitoring, XAI, and Real-Time Analytics.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    selected_algo = st.selectbox(
        "Active ML Algorithm",
        options=SUPPORTED_ALGORITHMS,
        format_func=lambda a: a.replace("_", " ").title(),
        index=0,
    )

    uploaded_file = st.file_uploader("Upload custom CSV", type=["csv"])
    st.caption("Supports: complaint_text, department, urgency, ward, channel, status")

dataset, artifacts, dataset_name = load_active_dataset(uploaded_file, algorithm=selected_algo)
prediction_frame = make_prediction_frame(dataset, artifacts)
overview = build_overview(prediction_frame)
executive_report = build_executive_report(prediction_frame)

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-title">Quick Exports</div>
            <div class="panel-copy">
                Export intelligence brief or full filtered operational queue in standard formats.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.download_button(
        "📄 Download Executive Brief",
        data=executive_report,
        file_name="civiclens_executive_brief.txt",
        mime="text/plain",
        use_container_width=True,
    )
    st.download_button(
        "📥 Download Full Queue CSV",
        data=queue_download(prediction_frame.sort_values("risk_score", ascending=False)),
        file_name="civiclens_queue.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
dedup_metrics = overview.get("deduplication_metrics", {})
st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-kicker">Civic Intelligence & Security Operations</div>
        <h1>CivicShield AI</h1>
        <div class="hero-copy">
            Unified civic intelligence and cybersecurity platform — classify complaints, detect security threats, explain ML predictions, and monitor operations in real time.
        </div>
        <div class="hero-meta">
            <span class="hero-pill">Dataset: {dataset_name}</span>
            <span class="hero-pill">Model: {selected_algo.replace('_', ' ').title()}</span>
            <span class="hero-pill">Dept Pressure: {overview['most_loaded_department']}</span>
            <span class="hero-pill">Confidence: {overview['average_confidence']}%</span>
            <span class="hero-pill">Dedup Savings: {dedup_metrics.get('dispatches_saved', 0)} dispatches avoided</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Main Tabs Navigation
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "Command Center",
        "Case Studio & Explainability",
        "Incident Clusters & Deduplication",
        "Emerging Topics (NMF)",
        "Data Explorer",
        "Ops Console",
        "Model Quality & Benchmark",
    ]
)

# ---------------------------------------------------------
# TAB 1: Command Center
# ---------------------------------------------------------
with tab1:
    metric_cols = st.columns(5)
    with metric_cols[0]:
        render_metric_card("Total Inflow", str(overview["total_cases"]), f"Cases from {dataset_name.lower()}.")
    with metric_cols[1]:
        render_metric_card(
            "Urgent Share",
            f"{overview['urgent_rate']}%",
            f"{overview['high_risk_cases']} items require expedited action.",
        )
    with metric_cols[2]:
        render_metric_card(
            "Critical Cases",
            str(overview["critical_cases"]),
            "Immediate emergency triage escalation.",
        )
    with metric_cols[3]:
        render_metric_card(
            "Citizen Frustration",
            f"{overview.get('average_frustration', 0)}/100",
            "Calculated emotional distress index.",
        )
    with metric_cols[4]:
        render_metric_card(
            "Dedup Savings",
            f"{dedup_metrics.get('dispatches_saved', 0)} cases",
            f"{dedup_metrics.get('deduplication_rate_pct', 0)}% redundant dispatches saved.",
        )

    left, right = st.columns([1.15, 0.85])
    department_df = pd.DataFrame(overview["department_distribution"])
    urgency_df = pd.DataFrame(overview["urgency_distribution"])
    action_df = pd.DataFrame(overview["action_board"])
    queue_df = pd.DataFrame(overview["top_risks"])
    geo_data = overview.get("geo_analytics", {})

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
                    "cases": "Total Cases",
                    "urgent_cases": "Urgent Cases",
                    "avg_risk_score": "Avg Risk Score",
                }
            )
            st.dataframe(dept_display, use_container_width=True, hide_index=True)

        st.markdown("### Priority Escalation Queue")
        if not queue_df.empty:
            queue_display = format_prediction_table(queue_df)
            st.dataframe(queue_display, use_container_width=True, hide_index=True)
            st.download_button(
                "Download priority queue CSV",
                data=queue_download(queue_df),
                file_name="civiclens_priority_queue.csv",
                mime="text/csv",
            )

    with right:
        render_surface_card(
            "Executive Brief",
            "Leadership Summary",
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

        if geo_data.get("ward_distribution"):
            st.markdown("### Ward / Geographic Pressure")
            ward_df = pd.DataFrame(geo_data["ward_distribution"]).head(5)
            if not ward_df.empty:
                w_chart = ward_df.set_index("ward")[["cases", "urgent_cases"]]
                st.bar_chart(w_chart, use_container_width=True)

        if geo_data.get("channel_distribution"):
            st.markdown("### Intake Channels")
            ch_df = pd.DataFrame(geo_data["channel_distribution"])
            st.dataframe(ch_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# TAB 2: Case Studio & Explainability
# ---------------------------------------------------------
with tab2:
    st.markdown("### Case Studio & Explainable AI (XAI)")
    st.caption("Evaluate a single complaint, inspect token-level feature attribution, check citizen frustration, and detect duplicate tickets.")

    case_templates = {
        "Water Main Rupture": "Major main water pipeline burst on 5th Avenue near the central library, gushing thousands of gallons and flooding nearby shops.",
        "Transformer Explosion": "Electric transformer exploded with loud bang near West Industrial gate 3, thick smoke billowing and power cut.",
        "Healthcare Emergency Delay": "The hospital emergency triage nurse refused treatment to accident victim demanding prior police clearance.",
        "Broken Traffic Signal": "Traffic signals at Central Crossway junction are dead, creating massive gridlock and near-head-on collisions.",
        "Street Safety Hazard": "Street lights near the girls' hostel have been out for a week and students feel unsafe returning after evening classes.",
        "Custom Input": "The city grievance website does not send complaint acknowledgements after submission and error 500 occurs.",
    }
    selected_scenario = st.selectbox("Preloaded Scenarios", list(case_templates.keys()))
    
    with st.form("case_form"):
        case_text = st.text_area(
            "Complaint Narrative",
            value=case_templates[selected_scenario],
            height=140,
        )
        run_case = st.form_submit_button("Run Deep Case Intelligence")

    if run_case or case_text:
        case_frame = enrich_predictions(artifacts.predict([case_text]))
        case = case_frame.iloc[0]
        case_summary = summarize_text(case_text, max_sentences=2)
        issue_signals = [term.replace("_", " ") for term, _ in top_keywords([case_text], limit=6)]

        # Run Explainability
        dept_explanation = explain_prediction(artifacts.department_model, case_text, top_k=6)
        urg_explanation = explain_prediction(artifacts.urgency_model, case_text, top_k=6)

        # Check duplicate similarity against database
        similar_in_queue = find_similar_complaints(
            case_text, dataset["complaint_text"].tolist(), top_k=2, threshold=0.22
        )

        st.markdown("---")
        result_cols = st.columns(5)
        with result_cols[0]:
            render_metric_card("Department", format_department(case["predicted_department"]), "Primary routing destination.")
        with result_cols[1]:
            render_metric_card("Urgency", str(case["predicted_urgency"]).title(), case["priority_label"])
        with result_cols[2]:
            render_metric_card("Dept Confidence", f"{case['department_confidence'] * 100:.1f}%", case["confidence_band"])
        with result_cols[3]:
            render_metric_card("Target SLA", case["target_sla"], f"{case['target_sla_hours']}h turnaround window.")
        with result_cols[4]:
            render_metric_card(
                "Frustration Index",
                f"{case['frustration_score']}/100",
                case["escalation_risk"],
            )

        # In-Queue Duplicate Warning
        if similar_in_queue:
            st.warning(
                f"⚠️ **Potential Duplicate Alert:** Detected {len(similar_in_queue)} similar incident report(s) already in the active queue! (Top Similarity: {similar_in_queue[0]['similarity_score'] * 100:.1f}%)"
            )
            with st.expander("View Matching Reports in Queue"):
                for m in similar_in_queue:
                    st.write(f"- **Sim: {m['similarity_score'] * 100:.1f}%**: {m['complaint_text']}")

        detail_left, detail_right = st.columns([1.1, 0.9])
        with detail_left:
            st.markdown("#### Explainable AI: Token Attribution")
            st.caption("Words shaded in teal contributed most strongly to the predicted department:")
            st.markdown(
                f'<div class="highlight-container">{dept_explanation["highlighted_html"]}</div>',
                unsafe_allow_html=True,
            )

            # Bar chart of token importance
            if dept_explanation["positive_features"]:
                tok_df = pd.DataFrame(dept_explanation["positive_features"]).set_index("token")[["score"]]
                tok_df = tok_df.rename(columns={"score": "Importance Score"})
                st.bar_chart(tok_df, use_container_width=True)

            render_surface_card("Summary", "Extractive Summary", case_summary)
            render_surface_card("Routing Guidance", "Recommended Action", str(case["recommended_action"]))

        with detail_right:
            render_surface_card(
                "Liaison Guidance",
                f"Citizen Escalation Level: {case['escalation_risk']}",
                f"{case['liaison_guidance']}<br><br><strong>Frustration Signals:</strong> {', '.join(case['frustration_signals']) if case['frustration_signals'] else 'None detected'}",
            )

            render_surface_card(
                "Impact Mapping",
                "Service Impact Area",
                f"Maps to <strong>{str(case['impact_area']).title()}</strong> under the <strong>{format_department(case['predicted_department'])}</strong> operational domain.",
            )

            st.markdown("#### Detected Issue Keywords")
            if issue_signals:
                render_signal_row(issue_signals)

            st.metric("Urgency Confidence", f"{case['urgency_confidence'] * 100:.1f}%")
            st.progress(float(case["urgency_confidence"]))
            st.metric("Combined Confidence", f"{case['combined_confidence'] * 100:.1f}%")
            st.progress(float(case["combined_confidence"]))

# ---------------------------------------------------------
# TAB 3: Incident Clusters & Deduplication Hub
# ---------------------------------------------------------
with tab3:
    st.markdown("### Incident Clusters & Deduplication Hub")
    st.caption("When emergencies occur, citizens report the same issue multiple times. This engine groups near-duplicate reports to prevent redundant field dispatches.")

    cluster_threshold = st.slider("Cosine Similarity Threshold", 0.15, 0.45, 0.22, 0.01)
    clusters = find_incident_clusters(prediction_frame, similarity_threshold=cluster_threshold)
    dedup = calculate_deduplication_metrics(clusters, len(prediction_frame))

    c_metric_cols = st.columns(4)
    with c_metric_cols[0]:
        render_metric_card("Incident Clusters", str(dedup["incident_clusters_count"]), "Distinct physical events detected.")
    with c_metric_cols[1]:
        render_metric_card("Duplicate Reports", str(dedup["duplicate_cases_found"]), "Corroborating follow-up complaints.")
    with c_metric_cols[2]:
        render_metric_card("Deduplication Rate", f"{dedup['deduplication_rate_pct']}%", "Share of duplicate tickets.")
    with c_metric_cols[3]:
        render_metric_card("Dispatches Saved", f"{dedup['dispatches_saved']} crews", "Avoided redundant dispatches.")

    st.markdown("---")
    st.markdown("#### Active Incident Clusters")

    if not clusters:
        st.info("No incident clusters detected at this threshold. Try lowering the similarity threshold slightly.")
    else:
        for c in clusters:
            with st.container():
                st.markdown(
                    f"""
                    <div class="cluster-box">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h4 style="margin: 0; color: #0f8b8d;">Cluster #{c['cluster_id']}: {c['incident_topic']}</h4>
                            <span class="hero-pill" style="background: #0f8b8d; font-size: 0.8rem;">{c['total_reports']} Reports</span>
                        </div>
                        <p style="margin: 0.4rem 0 0.2rem 0; font-size: 0.85rem; color: #5c6c7f;">
                            <strong>Department:</strong> {format_department(c.get('department', 'General'))} &nbsp;|&nbsp;
                            <strong>Urgency:</strong> {str(c.get('urgency', 'medium')).title()} &nbsp;|&nbsp;
                            <strong>Ward:</strong> {c.get('ward', 'General')}
                        </p>
                        <p style="margin: 0.3rem 0; font-size: 0.95rem; color: #112033;">
                            <strong>Primary Ticket:</strong> "{c['primary_complaint']}"
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.expander(f"View {c['corroborating_count']} Corroborating Ticket(s) in Cluster #{c['cluster_id']}"):
                    for rep in c["corroborating_reports"]:
                        st.markdown(f"- *\"{rep}\"*")

# ---------------------------------------------------------
# TAB 4: Emerging Topics (NMF Unsupervised NLP)
# ---------------------------------------------------------
with tab4:
    st.markdown("### Emerging Topics (Unsupervised NLP Mining)")
    st.caption("Automatically discover underlying issue themes and emerging civic trends from raw text without requiring pre-assigned category labels.")

    num_topics = st.slider("Number of Topics to Mine", min_value=3, max_value=8, value=5)
    discovered = discover_topics(prediction_frame["complaint_text"].tolist(), n_topics=num_topics)

    t_cols = st.columns(min(len(discovered), 4))
    for i, t in enumerate(discovered[:4]):
        with t_cols[i]:
            render_metric_card(
                f"Topic #{t['topic_id']}",
                f"{t['prevalence_pct']}%",
                f"{t['sample_count']} complaints",
            )

    st.markdown("---")
    for t in discovered:
        with st.container():
            st.markdown(f"#### Topic #{t['topic_id']}: {t['topic_name']} ({t['prevalence_pct']}% of queue)")
            render_signal_row(t["keywords"])
            with st.expander(f"Representative Complaints for Topic #{t['topic_id']}"):
                for rep in t["representative_complaints"]:
                    st.write(f"- *\"{rep}\"*")

# ---------------------------------------------------------
# TAB 5: Data Explorer
# ---------------------------------------------------------
with tab5:
    st.markdown("### Data Explorer")
    explorer = prediction_frame.copy()

    filter_cols = st.columns(4)
    with filter_cols[0]:
        search_text = st.text_input("Search text", placeholder="pipeline, hospital, dark...")
    with filter_cols[1]:
        dept_options = sorted(dataset["department"].unique().tolist())
        dept_filter = st.multiselect("Department", options=dept_options, default=dept_options)
    with filter_cols[2]:
        urg_options = sorted(dataset["urgency"].unique().tolist())
        urg_filter = st.multiselect("Urgency", options=urg_options, default=urg_options)
    with filter_cols[3]:
        ward_options = sorted(dataset["ward"].unique().tolist()) if "ward" in dataset.columns else []
        ward_filter = st.multiselect("Ward", options=ward_options, default=ward_options)

    if search_text:
        explorer = explorer[explorer["complaint_text"].str.contains(search_text, case=False, na=False)]
    if dept_filter:
        explorer = explorer[explorer["department"].isin(dept_filter)]
    if urg_filter:
        explorer = explorer[explorer["urgency"].isin(urg_filter)]
    if ward_filter and "ward" in explorer.columns:
        explorer = explorer[explorer["ward"].isin(ward_filter)]

    dept_agree = (prediction_frame["department"] == prediction_frame["predicted_department"]).mean() * 100
    urg_agree = (prediction_frame["urgency"] == prediction_frame["predicted_urgency"]).mean() * 100
    avg_len = prediction_frame["complaint_text"].str.len().mean()

    p_cols = st.columns(4)
    with p_cols[0]:
        render_metric_card("Visible Records", str(len(explorer)), "Filtered count.")
    with p_cols[1]:
        render_metric_card("Avg Text Length", f"{avg_len:.0f} chars", "Average length.")
    with p_cols[2]:
        render_metric_card("Dept Agreement", f"{dept_agree:.1f}%", "Ground truth match.")
    with p_cols[3]:
        render_metric_card("Urgency Agreement", f"{urg_agree:.1f}%", "Ground truth match.")

    st.markdown("#### Records")
    display_cols = [
        "complaint_text",
        "department",
        "predicted_department",
        "urgency",
        "predicted_urgency",
        "combined_confidence",
        "frustration_score",
        "escalation_risk",
        "ward",
        "channel",
        "target_sla",
    ]
    avail_cols = [c for c in display_cols if c in explorer.columns]
    explorer_display = format_prediction_table(explorer[avail_cols])
    st.dataframe(explorer_display, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# TAB 6: Ops Console
# ---------------------------------------------------------
with tab6:
    st.markdown("### Ops Console")
    ops = prediction_frame.copy()

    ops_cols = st.columns(4)
    with ops_cols[0]:
        pred_dept = st.selectbox(
            "Filter Department",
            options=["all"] + sorted(prediction_frame["predicted_department"].unique().tolist()),
            format_func=lambda d: "All Departments" if d == "all" else format_department(d),
        )
    with ops_cols[1]:
        pred_urg = st.multiselect(
            "Filter Urgency",
            options=sorted(prediction_frame["predicted_urgency"].unique().tolist()),
            default=sorted(prediction_frame["predicted_urgency"].unique().tolist()),
        )
    with ops_cols[2]:
        min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.40, 0.05)
    with ops_cols[3]:
        max_rows = st.slider("Display Limit", 5, len(prediction_frame), min(15, len(prediction_frame)))

    if pred_dept != "all":
        ops = ops[ops["predicted_department"] == pred_dept]
    ops = ops[ops["predicted_urgency"].isin(pred_urg)]
    ops = ops[ops["combined_confidence"] >= min_conf]

    if ops.empty:
        st.warning("No complaints match the current filters.")
    else:
        o_cols = st.columns(4)
        with o_cols[0]:
            render_metric_card("Active Queue", str(len(ops)), "Filtered complaints.")
        with o_cols[1]:
            render_metric_card("Avg Risk Score", f"{ops['risk_score'].mean():.1f}", "Blended operational severity.")
        with o_cols[2]:
            render_metric_card(
                "Expedited SLA (≤8h)",
                str(int((ops["target_sla_hours"] <= 8).sum())),
                "High & critical cases.",
            )
        with o_cols[3]:
            render_metric_card(
                "Needs Analyst Review",
                str(int((ops["combined_confidence"] < 0.6).sum())),
                "Low confidence items.",
            )

        st.markdown("#### Operational Dispatch Queue")
        q_cols = [
            "complaint_text",
            "predicted_department",
            "predicted_urgency",
            "combined_confidence",
            "frustration_score",
            "escalation_risk",
            "target_sla",
            "risk_score",
            "recommended_action",
        ]
        q_view = ops.sort_values(["risk_score", "frustration_score"], ascending=[False, False]).head(max_rows)
        st.dataframe(format_prediction_table(q_view[q_cols]), use_container_width=True, hide_index=True)

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "📥 Export Filtered Queue (CSV)",
                data=queue_download(q_view[q_cols]),
                file_name="civiclens_filtered_queue.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "📄 Export Queue Brief (TXT)",
                data=build_executive_report(ops),
                file_name="civiclens_queue_brief.txt",
                mime="text/plain",
                use_container_width=True,
            )

# ---------------------------------------------------------
# TAB 7: Model Quality & Benchmark
# ---------------------------------------------------------
with tab7:
    st.markdown("### Model Quality & Multi-Model Benchmark")
    st.caption("Compare multiple machine learning architectures on cross-validated holdout metrics to pick the best model for production routing.")

    st.markdown("#### Algorithm Leaderboard (3-Fold Stratified CV)")
    with st.spinner("Benchmarking models across cross-validation folds..."):
        bench_results = benchmark_models(dataset, cv_splits=3)

    bench_df = pd.DataFrame(bench_results["leaderboard"])
    bench_display = bench_df.rename(
        columns={
            "display_name": "Algorithm",
            "department_accuracy": "Dept Accuracy",
            "department_weighted_f1": "Dept Weighted F1",
            "urgency_accuracy": "Urgency Accuracy",
            "urgency_weighted_f1": "Urgency Weighted F1",
            "training_latency_ms": "Latency (ms)",
        }
    ).drop(columns=["algorithm"])
    st.dataframe(bench_display, use_container_width=True, hide_index=True)

    st.success(f"🏆 **Recommended Champion Model:** {bench_results['best_algorithm'].replace('_', ' ').title()}")

    st.markdown("---")
    st.markdown(f"#### Active Model Performance ({selected_algo.replace('_', ' ').title()})")
    metrics = artifacts.metrics

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.metric("Department Accuracy", f"{metrics['department_accuracy'] * 100:.1f}%")
        st.caption("Classification Report: Department")
        dept_rep = pd.DataFrame(metrics["department_report"]).transpose().round(3)
        st.dataframe(dept_rep, use_container_width=True)

    with m_col2:
        st.metric("Urgency Accuracy", f"{metrics['urgency_accuracy'] * 100:.1f}%")
        st.caption("Classification Report: Urgency")
        urg_rep = pd.DataFrame(metrics["urgency_report"]).transpose().round(3)
        st.dataframe(urg_rep, use_container_width=True)

    # Confusion Matrices
    if "confusion_matrices" in metrics:
        st.markdown("#### Interactive Confusion Matrices")
        cm_c1, cm_c2 = st.columns(2)
        with cm_c1:
            st.markdown("**Department Confusion Matrix**")
            d_cm = metrics["confusion_matrices"]["department"]
            d_cm_df = pd.DataFrame(d_cm["matrix"], index=d_cm["labels"], columns=d_cm["labels"])
            st.dataframe(d_cm_df, use_container_width=True)
        with cm_c2:
            st.markdown("**Urgency Confusion Matrix**")
            u_cm = metrics["confusion_matrices"]["urgency"]
            u_cm_df = pd.DataFrame(u_cm["matrix"], index=u_cm["labels"], columns=u_cm["labels"])
            st.dataframe(u_cm_df, use_container_width=True)

    st.markdown("#### Active Learning: Lowest Confidence Queue")
    st.caption("These items have the lowest model certainty and represent the highest-value candidates for manual analyst labeling:")
    low_conf = prediction_frame.sort_values("combined_confidence").head(6)
    lc_cols = ["complaint_text", "predicted_department", "predicted_urgency", "combined_confidence", "impact_area"]
    st.dataframe(format_prediction_table(low_conf[lc_cols]), use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# TAB 8: Security Operations Dashboard
# ---------------------------------------------------------
with tab8:
    st.markdown("### \U0001f6e1 Security Operations Center")
    st.caption("Real-time security event monitoring, threat detection, and incident correlation. All data labelled as **demo environment**.")

    sec_col1, sec_col2 = st.columns([0.3, 0.7])
    with sec_col1:
        event_count = st.slider("Demo Events to Generate", 20, 200, 75, 5)
        hours_back = st.slider("Time Window (hours)", 1, 72, 24)

    demo_events = generate_demo_events(count=event_count, hours_back=hours_back, seed=42)

    # Run rule engine on all events
    rule_engine = RuleEngine()
    all_alerts = []
    for i, evt in enumerate(demo_events):
        history = demo_events[max(0, i-50):i]
        alerts = rule_engine.evaluate(evt, history)
        all_alerts.extend(alerts)

    # Run anomaly detector
    detector = AnomalyDetector(contamination=0.08)
    detector.fit(demo_events)
    anomaly_alerts = []
    for evt in demo_events:
        ml_alert = detector.detect(evt)
        if ml_alert:
            anomaly_alerts.append(ml_alert)

    all_alerts.extend(anomaly_alerts)

    # Correlate events
    threads = correlate_events(demo_events, time_window_minutes=15)

    # KPI Cards
    s_cols = st.columns(5)
    with s_cols[0]:
        render_metric_card("Total Events", str(len(demo_events)), f"Last {hours_back}h (demo)")
    with s_cols[1]:
        critical_count = sum(1 for e in demo_events if e.severity.value == "critical")
        render_metric_card("Critical Events", str(critical_count), "Require immediate review")
    with s_cols[2]:
        render_metric_card("Rule Alerts", str(len(all_alerts) - len(anomaly_alerts)), "Deterministic detections")
    with s_cols[3]:
        render_metric_card("ML Anomalies", str(len(anomaly_alerts)), "Isolation Forest detections")
    with s_cols[4]:
        render_metric_card("Correlated Threads", str(len(threads)), "Related event groups")

    st.markdown("---")

    sec_left, sec_right = st.columns([1.1, 0.9])
    with sec_left:
        st.markdown("#### Recent Security Events")
        event_records = []
        for e in demo_events[:30]:
            event_records.append({
                "Event ID": e.event_id,
                "Time": e.timestamp.strftime("%H:%M:%S"),
                "Source IP": e.source_ip,
                "Service": e.source_service,
                "Category": e.category.value.replace("_", " ").title(),
                "Severity": e.severity.value.upper(),
                "Description": e.description[:80],
            })
        st.dataframe(pd.DataFrame(event_records), use_container_width=True, hide_index=True)

    with sec_right:
        st.markdown("#### Active Alerts")
        if all_alerts:
            for alert in all_alerts[:10]:
                severity_colors = {"critical": "\U0001f534", "high": "\U0001f7e0", "medium": "\U0001f7e1", "low": "\U0001f7e2"}
                icon = severity_colors.get(alert.severity.value, "\u26aa")
                st.markdown(f"{icon} **{alert.title}** ({alert.severity.value.upper()})")
                st.caption(f"{alert.description[:120]}")
                st.markdown("---")
        else:
            st.success("No active alerts detected.")

        st.markdown("#### Correlated Threat Threads")
        if threads:
            for t in threads[:5]:
                st.markdown(
                    f"**Thread #{t['thread_id']}** — IP: `{t['source_ip']}` | "
                    f"{t['event_count']} events | Max Severity: {t['max_severity'].upper()} | "
                    f"Span: {t['time_span_minutes']}min"
                )
                with st.expander(f"Details for Thread #{t['thread_id']}"):
                    for desc in t['descriptions'][:3]:
                        st.write(f"- {desc}")
        else:
            st.info("No correlated threat threads detected.")

    # Category distribution chart
    st.markdown("#### Event Category Distribution")
    cat_counts = pd.Series([e.category.value for e in demo_events]).value_counts()
    cat_df = pd.DataFrame({"Category": cat_counts.index.str.replace("_", " ").str.title(), "Count": cat_counts.values})
    st.bar_chart(cat_df.set_index("Category"), use_container_width=True)

    st.info("\u2139\ufe0f All security events shown above are **synthetic demo data** generated for demonstration purposes.")


# ---------------------------------------------------------
# TAB 9: ML Lab — Enhanced Model Training
# ---------------------------------------------------------
with tab9:
    st.markdown("### \U0001f9ea ML Lab — Enhanced Training Pipeline")
    st.caption("Train models with character n-gram features, compare word-only vs combined features, and track model versions.")

    lab_col1, lab_col2 = st.columns(2)
    with lab_col1:
        use_char = st.checkbox("Enable Character N-gram Features", value=True)
        st.caption("Adds character-level TF-IDF (3-5 grams) alongside word features for better handling of typos and morphological patterns.")
    with lab_col2:
        st.markdown("**Feature Combination Strategy**")
        if use_char:
            st.success("\u2705 Word (1,2)-grams + Char (3,5)-grams via FeatureUnion")
        else:
            st.info("Word (1,2)-grams only (baseline)")

    if st.button("\U0001f680 Train Enhanced Model", use_container_width=True):
        with st.spinner("Training enhanced pipeline..."):
            from civicshield.ml import train_enhanced_model, compute_enhanced_metrics
            from sklearn.model_selection import train_test_split

            texts = dataset["complaint_text"].tolist()
            dept_labels = dataset["department"].tolist()
            urg_labels = dataset["urgency"].tolist()

            # Split
            from civiclens.modeling import safe_train_test_split
            train_df, test_df = safe_train_test_split(dataset, test_size=0.25, random_state=42)

            # Train department model
            dept_model = train_enhanced_model(
                train_df["complaint_text"].tolist(),
                train_df["department"].tolist(),
                use_char_features=use_char,
            )
            dept_pred = dept_model.predict(test_df["complaint_text"].tolist())
            dept_metrics = compute_enhanced_metrics(
                test_df["department"].tolist(), dept_pred.tolist(), label="department"
            )

            # Train urgency model
            urg_model = train_enhanced_model(
                train_df["complaint_text"].tolist(),
                train_df["urgency"].tolist(),
                use_char_features=use_char,
            )
            urg_pred = urg_model.predict(test_df["complaint_text"].tolist())
            urg_metrics = compute_enhanced_metrics(
                test_df["urgency"].tolist(), urg_pred.tolist(), label="urgency"
            )

        st.markdown("#### Enhanced Model Results")
        r_cols = st.columns(4)
        with r_cols[0]:
            render_metric_card("Dept Accuracy", f"{dept_metrics['department_accuracy']*100:.1f}%", "Enhanced pipeline")
        with r_cols[1]:
            render_metric_card("Dept Macro F1", f"{dept_metrics['department_macro_f1']*100:.1f}%", "Across all classes")
        with r_cols[2]:
            render_metric_card("Urg Accuracy", f"{urg_metrics['urgency_accuracy']*100:.1f}%", "Enhanced pipeline")
        with r_cols[3]:
            render_metric_card("Urg Macro F1", f"{urg_metrics['urgency_macro_f1']*100:.1f}%", "Across all classes")

        # Comparison with baseline
        st.markdown("#### Baseline vs Enhanced Comparison")
        baseline_metrics = artifacts.metrics
        comparison = pd.DataFrame({
            "Metric": ["Department Accuracy", "Urgency Accuracy"],
            "Baseline (word only)": [
                f"{baseline_metrics['department_accuracy']*100:.1f}%",
                f"{baseline_metrics['urgency_accuracy']*100:.1f}%",
            ],
            "Enhanced": [
                f"{dept_metrics['department_accuracy']*100:.1f}%",
                f"{urg_metrics['urgency_accuracy']*100:.1f}%",
            ],
            "Features": [
                "Word+Char" if use_char else "Word only",
                "Word+Char" if use_char else "Word only",
            ],
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        # Model version
        from civicshield.ml import generate_model_version, hash_dataset
        ds_hash = hash_dataset(dataset)
        version = generate_model_version("enhanced_lr", ds_hash)
        st.info(f"\U0001f4e6 Model Version: `{version}` | Dataset Hash: `{ds_hash[:16]}...`")


# ---------------------------------------------------------
# TAB 10: Data Quality Dashboard
# ---------------------------------------------------------
with tab10:
    st.markdown("### \U0001f4cb Data Quality Dashboard")
    st.caption("Automated data quality checks, class balance analysis, and dataset health monitoring.")

    quality_report = check_data_quality(dataset)

    # Health banner
    if quality_report.is_healthy:
        st.success("\u2705 Dataset is healthy — no critical quality issues detected.")
    else:
        st.warning(f"\u26a0\ufe0f {len(quality_report.warnings)} quality warning(s) detected.")

    # KPI cards
    dq_cols = st.columns(5)
    with dq_cols[0]:
        render_metric_card("Total Rows", str(quality_report.total_rows), "Records in dataset")
    with dq_cols[1]:
        render_metric_card("Missing Text", str(quality_report.missing_text), "Empty complaint fields")
    with dq_cols[2]:
        render_metric_card("Duplicates", str(quality_report.duplicate_texts), "Exact text duplicates")
    with dq_cols[3]:
        render_metric_card("Short Texts", str(quality_report.very_short_texts), "< 10 characters")
    with dq_cols[4]:
        render_metric_card("Imbalance Ratio", f"{quality_report.imbalance_ratio:.1f}x", "Max/min class ratio")

    st.markdown("---")

    dq_left, dq_right = st.columns(2)
    with dq_left:
        st.markdown("#### Class Distribution")
        if quality_report.class_distribution:
            class_df = pd.DataFrame(
                list(quality_report.class_distribution.items()),
                columns=["Department", "Count"]
            )
            class_df["Department"] = class_df["Department"].apply(
                lambda d: d.replace("_", " ").title()
            )
            st.bar_chart(class_df.set_index("Department"), use_container_width=True)
            st.dataframe(class_df, use_container_width=True, hide_index=True)

    with dq_right:
        st.markdown("#### Quality Warnings")
        if quality_report.warnings:
            for warning in quality_report.warnings:
                st.warning(f"\u26a0\ufe0f {warning}")
        else:
            st.success("No warnings — dataset passes all quality checks.")

        st.markdown("#### Text Length Distribution")
        if "complaint_text" in dataset.columns:
            lengths = dataset["complaint_text"].str.len()
            length_stats = pd.DataFrame({
                "Statistic": ["Mean", "Median", "Min", "Max", "Std Dev"],
                "Characters": [
                    f"{lengths.mean():.0f}",
                    f"{lengths.median():.0f}",
                    f"{lengths.min():.0f}",
                    f"{lengths.max():.0f}",
                    f"{lengths.std():.0f}",
                ],
            })
            st.dataframe(length_stats, use_container_width=True, hide_index=True)

    # Dataset hash for versioning
    from civicshield.ml import hash_dataset
    ds_hash = hash_dataset(dataset)
    st.info(f"\U0001f4be Dataset Hash: `{ds_hash[:32]}...` — use this to track which dataset version trained which model.")
