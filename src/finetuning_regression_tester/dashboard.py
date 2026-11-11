"""Optional Streamlit investigation dashboard."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path


def launch(comparison_path: Path, demo: bool = False) -> None:
    if not comparison_path.is_file():
        raise RuntimeError(f"Comparison artifact does not exist: {comparison_path}")
    try:
        import streamlit  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Dashboard dependencies are missing. Install with: uv sync --extra dashboard"
        ) from exc
    environment = os.environ.copy()
    environment["FINETUNING_REGRESSION_TESTER_COMPARISON"] = str(comparison_path.resolve())
    environment["FINETUNING_REGRESSION_TESTER_DEMO"] = "1" if demo else "0"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())],
        env=environment,
        check=False,
    )


def _render_dashboard(comparison_path: Path, demo: bool = False) -> None:
    try:
        import plotly.graph_objects as go
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError(
            "Dashboard dependencies are missing. Install with: uv sync --extra dashboard"
        ) from exc
    comparison = json.loads(comparison_path.read_text())
    st.set_page_config(page_title="FineTuning Regression Tester", layout="wide")
    st.title("FineTuning Regression Tester")
    st.caption("See what your model gained. See what it forgot.")
    if demo or comparison.get("demo_data"):
        st.error("DEMO DATA — fictional fixture results, not measurements from model training.")
    manifest_path = comparison_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    st.subheader("Experiment overview")
    summary = comparison["summary"]
    training = manifest.get("training", comparison.get("training_config", {}))
    training_preset = (
        f"{training.get('method', 'unknown')}, {training.get('iterations', 'unknown')} iterations"
        if training
        else "Not recorded"
    )
    overview = st.columns(5)
    for col, label, value in zip(
        overview,
        ["Experiment", "Model", "Adapter", "Training preset", "Dataset version"],
        [
            manifest.get("experiment_id", "Not recorded"),
            comparison.get("model", "Not recorded"),
            comparison.get("adapter", "Not recorded"),
            training_preset,
            manifest.get("dataset_version", comparison.get("dataset_version", "Not recorded")),
        ],
    ):
        col.caption(label)
        col.write(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
    cols = st.columns(6)
    for col, label, key in zip(
        cols,
        [
            "Total evaluated",
            "Valid paired",
            "Base accuracy",
            "Adapted accuracy",
            "Regressions",
            "Improvements",
        ],
        [
            "total_cases",
            "paired_valid",
            "baseline_accuracy",
            "adapted_accuracy",
            "regressions",
            "improvements",
        ],
    ):
        value = summary.get(key)
        col.metric(label, _percent(value) if "accuracy" in key else str(value))
    time_cols = st.columns(4)
    timings = comparison.get("timings", {})
    time_cols[0].metric("Training duration", _seconds(timings.get("training_seconds")))
    time_cols[1].metric(
        "Evaluation duration",
        _seconds(
            (timings.get("baseline_evaluation_seconds") or 0)
            + (timings.get("adapted_evaluation_seconds") or 0)
        ),
    )
    time_cols[2].metric(
        "Valid base / adapted",
        f"{summary.get('baseline_valid', summary['paired_valid'])} / "
        f"{summary.get('adapted_valid', summary['paired_valid'])}",
    )
    time_cols[3].caption("Dataset hashes")
    time_cols[3].write(manifest.get("dataset_hashes", "Not recorded"))
    capabilities = comparison["capabilities"]
    names = list(capabilities)
    z = [[capabilities[name]["accuracy_change"] for name in names]]
    text = [[_percent(value) for value in z[0]]]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=["Accuracy change"],
            y=names,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale="RdYlGn",
            text=text,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
    st.subheader("Capability map")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(
        [
            {
                "Capability": name,
                "Base": capabilities[name]["baseline_accuracy"],
                "Adapted": capabilities[name]["adapted_accuracy"],
                "Change": capabilities[name]["accuracy_change"],
                "Regressions": capabilities[name]["regressions"],
                "Improvements": capabilities[name]["improvements"],
                "Retention": capabilities[name]["retention_rate"],
            }
            for name in names
        ],
        width="stretch",
    )
    selected_capability = st.selectbox("Inspect capability cases", names)
    selected_cases = [
        row for row in comparison["cases"] if row["capability"] == selected_capability
    ]
    st.dataframe(_display_cases(selected_cases), width="stretch")
    cases = comparison["cases"]
    filters = st.columns(6)
    transitions = filters[0].multiselect(
        "Transition",
        sorted({row["transition"] for row in cases}),
        default=["REGRESSION", "IMPROVEMENT"],
    )
    selected_capabilities = filters[1].multiselect(
        "Capability",
        sorted({row["capability"] for row in cases}),
        default=sorted({row["capability"] for row in cases}),
    )
    severity_choices = sorted({row["severity"] for row in cases if row["severity"]})
    severities = filters[2].multiselect("Severity", severity_choices, default=severity_choices)
    format_filter = filters[3].selectbox("Format validity", ["Any", "Changed", "Unchanged"])
    baseline_filter = filters[4].selectbox(
        "Baseline outcome", ["Any", "Passed", "Failed", "Inconclusive"]
    )
    adapted_filter = filters[5].selectbox(
        "Adapted outcome", ["Any", "Passed", "Failed", "Inconclusive"]
    )

    def outcome_matches(value: bool | None, selection: str) -> bool:
        return (
            selection == "Any"
            or (selection == "Passed" and value is True)
            or (selection == "Failed" and value is False)
            or (selection == "Inconclusive" and value is None)
        )

    selected = [
        row
        for row in cases
        if row["transition"] in transitions
        and row["capability"] in selected_capabilities
        and (row["severity"] is None or row["severity"] in severities)
        and (
            format_filter == "Any"
            or (row["base_format_valid"] != row["adapted_format_valid"])
            == (format_filter == "Changed")
        )
        and outcome_matches(row["base_passed"], baseline_filter)
        and outcome_matches(row["adapted_passed"], adapted_filter)
    ]
    st.subheader("Regression explorer")
    st.dataframe(_display_cases(selected), width="stretch")

    target_cases = [case for case in cases if case["suite"] == "target"]
    if target_cases:
        st.subheader("Target task")
        target = comparison.get("target_analysis", {})
        base_col, adapted_col = st.columns(2)
        for label, side, col in (("Base", "base", base_col), ("Adapted", "adapted", adapted_col)):
            metrics = target.get(side, {})
            col.metric(
                f"{label} exact-record accuracy", _percent(metrics.get("exact_record_accuracy"))
            )
            col.metric(f"{label} JSON validity", _percent(metrics.get("json_validity")))
        field_scores = target.get("adapted", {}).get("field_accuracy", {})
        if field_scores:
            chart = go.Figure(
                go.Bar(x=list(field_scores), y=[field_scores[key] for key in field_scores])
            )
            chart.update_yaxes(range=[0, 1], tickformat=".0%")
            chart.update_layout(height=280, yaxis_title="Adapted field accuracy")
            st.plotly_chart(chart, width="stretch")
        st.json(target)

    st.subheader("Training analysis")
    training_log = comparison_path.parent / "training.log"
    if training_log.exists():
        log_text = training_log.read_text(errors="replace")
        train_points, validation_points = [], []
        for line in log_text.splitlines():
            step = re.search(r"(?:Iter(?:ation)?|Step)\s*(\d+)", line, re.IGNORECASE)
            if not step:
                continue
            train_loss = re.search(r"Train loss\s*[:= ]+([0-9.eE+-]+)", line, re.IGNORECASE)
            val_loss = re.search(r"Val(?:idation)? loss\s*[:= ]+([0-9.eE+-]+)", line, re.IGNORECASE)
            if train_loss:
                train_points.append((int(step.group(1)), float(train_loss.group(1))))
            if val_loss:
                validation_points.append((int(step.group(1)), float(val_loss.group(1))))
        if train_points or validation_points:
            chart = go.Figure()
            if train_points:
                chart.add_scatter(
                    x=[point[0] for point in train_points],
                    y=[point[1] for point in train_points],
                    name="Training loss",
                )
            if validation_points:
                chart.add_scatter(
                    x=[point[0] for point in validation_points],
                    y=[point[1] for point in validation_points],
                    name="Validation loss",
                )
            st.plotly_chart(chart, width="stretch")
        else:
            st.info("Training ran, but no loss lines were parsed from the MLX-LM log.")
        with st.expander("Raw training log"):
            st.code(log_text)
        steps = [
            int(match.group(1))
            for match in re.finditer(r"(?:Iter(?:ation)?|Step)\s*(\d+)", log_text, re.IGNORECASE)
        ]
        training_col, steps_col, checkpoints_col = st.columns(3)
        training_col.metric("Elapsed training", _seconds(timings.get("training_seconds")))
        steps_col.metric("Reported steps", str(max(steps)) if steps else "Not recorded")
        adapter_path = comparison.get("adapter")
        checkpoint_files = []
        if adapter_path and Path(adapter_path).is_dir():
            checkpoint_files = [
                item.name
                for item in Path(adapter_path).iterdir()
                if item.suffix in {".safetensors", ".npz"}
            ]
        checkpoints_col.metric("Checkpoints", str(len(checkpoint_files)))
        if checkpoint_files:
            st.write("Checkpoint files:", ", ".join(sorted(checkpoint_files)))
    else:
        st.info("No training log is attached to this comparison.")

    st.subheader("Run metadata")
    st.json(
        {
            "manifest": manifest,
            "model": comparison.get("model"),
            "adapter": comparison.get("adapter"),
            "decoding": comparison.get("decoding"),
            "dataset_hashes": comparison.get("dataset_hashes"),
            "comparison_schema": comparison.get("schema_version"),
            "generated_at": comparison.get("generated_at"),
            "interpretation": comparison.get("methodology"),
        }
    )


def _percent(value: float | int | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def _seconds(value: float | int | None) -> str:
    return "N/A" if value is None else f"{value:.1f} s"


def _display_cases(rows: list[dict]) -> list[dict]:
    return [
        {
            key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
            for key, value in row.items()
        }
        for row in rows
    ]


if __name__ == "__main__" and os.environ.get("FINETUNING_REGRESSION_TESTER_COMPARISON"):
    _render_dashboard(
        Path(os.environ["FINETUNING_REGRESSION_TESTER_COMPARISON"]),
        os.environ.get("FINETUNING_REGRESSION_TESTER_DEMO") == "1",
    )
