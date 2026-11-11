"""Offline report exports from paired comparison artifacts."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any


def export_reports(comparison_path: Path, output: Path) -> list[Path]:
    comparison = json.loads(comparison_path.read_text())
    manifest_path = comparison_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    output.mkdir(parents=True, exist_ok=True)
    cases = comparison["cases"]
    paths = [
        output / "report.md",
        output / "report.html",
        output / "comparisons.csv",
        output / "capabilities.json",
    ]
    summary = comparison["summary"]
    model = comparison.get("model") or "Not recorded"
    if isinstance(model, dict):
        model = json.dumps(model, ensure_ascii=False)
    adapter = comparison.get("adapter") or "Not recorded"
    training_config = comparison.get("training_config")
    target = comparison.get("target_analysis", {})
    suites: dict[str, list[dict[str, Any]]] = {}
    for row in cases:
        suites.setdefault(row["suite"], []).append(row)
    demo_banner = (
        "DEMO DATA — fictional fixture outputs, not measurements from model training."
        if comparison.get("demo_data")
        else None
    )
    md = [
        "# FineTuning Regression Tester experiment report",
        "",
    ]
    if demo_banner:
        md.extend([f"> **{demo_banner}**", ""])
    md.extend(
        [
            "Observed behavior changes from paired evaluations.",
            "",
            "## Summary",
            "",
            f"- Model: `{model}`",
            f"- Adapter: `{adapter}`",
            f"- Paired cases: {summary['paired_valid']} / {summary['total_cases']}",
            f"- Baseline accuracy: {_pct(summary['baseline_accuracy'])} ({summary.get('baseline_valid', summary['paired_valid'])} valid)",
            f"- Adapted accuracy: {_pct(summary['adapted_accuracy'])} ({summary.get('adapted_valid', summary['paired_valid'])} valid)",
            f"- Accuracy change: {_pct(summary['accuracy_change'])}",
            f"- Regressions: {summary['regressions']}",
            f"- Improvements: {summary['improvements']}",
            f"- Retention: {_pct(summary['retention_rate'])}",
            f"- Forgetting: {_pct(summary['forgetting_rate'])}",
            f"- Experiment ID: `{manifest.get('experiment_id', comparison.get('experiment_id', 'Not recorded'))}`",
            f"- Evaluation cases hash: `{comparison.get('evaluation_cases_hash', 'Not recorded')}`",
            f"- Resolved model revision: `{comparison.get('resolved_model_revision', manifest.get('resolved_model_revision', 'Not recorded'))}`",
            f"- Dataset version / seed: `{comparison.get('dataset_version', manifest.get('dataset_version', 'Not recorded'))}` / `{comparison.get('dataset_seed', 'Not recorded')}`",
            f"- Environment: `{json.dumps(comparison.get('environment', {key: manifest.get(key) for key in ('python', 'os', 'mlx_lm_version')}), sort_keys=True)}`",
            "",
            "## Dataset and execution details",
            "",
            f"- Dataset hashes: `{json.dumps(comparison.get('dataset_hashes', manifest.get('dataset_hashes', 'Not recorded')), sort_keys=True)}`",
            f"- Baseline evaluation time: {summary_time(comparison.get('timings', {}).get('baseline_evaluation_seconds'))}",
            f"- Training time: {summary_time(comparison.get('timings', {}).get('training_seconds'))}",
            f"- Adapted evaluation time: {summary_time(comparison.get('timings', {}).get('adapted_evaluation_seconds'))}",
            "",
            "## Capability map",
            "",
            "| Capability | Base | Adapted | Change | Regressions | Improvements |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for cap, m in comparison["capabilities"].items():
        md.append(
            f"| {cap} | {_pct(m['baseline_accuracy'])} | {_pct(m['adapted_accuracy'])} | {_pct(m['accuracy_change'])} | {m['regressions']} | {m['improvements']} |"
        )
    if target:
        md.extend(
            [
                "",
                "## Target extraction diagnostics",
                "",
                "| Model stage | Exact record | JSON valid | Schema valid | Unexpected fields | Extra prose |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for side in ("base", "adapted"):
            values = target.get(side, {})
            md.append(
                f"| {side.title()} | {_pct(values.get('exact_record_accuracy'))} | "
                f"{_pct(values.get('json_validity'))} | {_pct(values.get('schema_validity'))} | "
                f"{values.get('unexpected_fields', 0)} | {values.get('extra_prose', 0)} |"
            )
            if values.get("field_accuracy"):
                md.extend(["", f"{side.title()} field accuracy:", ""])
                for field, score in values["field_accuracy"].items():
                    md.append(f"- `{field}`: {_pct(score)}")
    for suite, title in (
        ("format", "Format adherence"),
        ("unsupported", "Missing-data abstention"),
        ("unsupported_capability", "Unsupported-capability requests"),
    ):
        suite_rows = suites.get(suite, [])
        if suite_rows:
            md.extend(["", f"## {title}", "", f"Cases: {len(suite_rows)}", ""])
            for side in ("base", "adapted"):
                valid = [row for row in suite_rows if row[f"{side}_passed"] is not None]
                passed = sum(row[f"{side}_passed"] is True for row in valid)
                format_valid = [row for row in valid if row[f"{side}_format_valid"] is not None]
                format_count = sum(row[f"{side}_format_valid"] is True for row in format_valid)
                md.append(
                    f"- {side.title()}: {passed}/{len(valid)} semantically correct; "
                    f"format valid in {format_count}/{len(format_valid)} scored responses."
                )
    if training_config:
        md.extend(
            [
                "",
                "## Training configuration",
                "",
                "```json",
                json.dumps(training_config, indent=2),
                "```",
            ]
        )
    for transition, title in [("REGRESSION", "Regressions"), ("IMPROVEMENT", "Improvements")]:
        md.extend(["", f"## {title}", ""])
        matches = [r for r in cases if r["transition"] == transition]
        if not matches:
            md.append("No cases in this category.")
        for r in matches:
            md.extend(
                [
                    f"### {r['case_id']} ({r['capability']}, {r['severity']})",
                    "",
                    f"**Input:** {r['input']}",
                    "",
                    f"**Expected:** `{json.dumps(r['expected'], ensure_ascii=False)}`",
                    "",
                    f"**Base:** `{r['base_output']}`",
                    "",
                    f"**Adapted:** `{r['adapted_output']}`",
                    "",
                ]
            )
    inconclusive = [row for row in cases if row["transition"] == "INCONCLUSIVE"]
    if inconclusive:
        md.extend(["", "## Inconclusive cases", ""])
        for row in inconclusive:
            md.append(
                f"- `{row['case_id']}`: base error `{row.get('base_error') or 'none'}`; "
                f"adapted error `{row.get('adapted_error') or 'none'}`."
            )
    md.extend(
        [
            "## Interpretation limits",
            "",
            comparison["methodology"]["interpretation"],
            "",
            "Small samples are exploratory. Scores summarize these cases and do not establish general model capability.",
            "",
            "## Reproduction",
            "",
            "The raw paired outputs and frozen evaluation cases are retained beside the experiment manifest. "
            "Recompute the analysis without model inference by running:",
            "",
            "```bash",
            "finetuning-regression-tester compare --baseline PATH/baseline_results.jsonl \\",
            "  --adapted PATH/adapted_results.jsonl --output PATH/comparison.json",
            "finetuning-regression-tester report --comparison PATH/comparison.json --output PATH/report",
            "```",
            "",
            "Exact model revision, MLX-LM version, decoding settings, seeds, dataset hashes, and completed stages are recorded when available in `manifest.json`.",
        ]
    )
    paths[0].write_text("\n".join(md) + "\n")
    table = "".join(
        f"<tr><td>{html.escape(c)}</td><td>{_pct(m['baseline_accuracy'])}</td><td>{_pct(m['adapted_accuracy'])}</td><td>{_pct(m['accuracy_change'])}</td><td>{m['regressions']}</td><td>{m['improvements']}</td></tr>"
        for c, m in comparison["capabilities"].items()
    )
    target_rows = "".join(
        f"<tr><td>{html.escape(side.title())}</td><td>{_pct(values.get('exact_record_accuracy'))}</td>"
        f"<td>{_pct(values.get('json_validity'))}</td><td>{_pct(values.get('schema_validity'))}</td>"
        f"<td>{values.get('unexpected_fields', 0)}</td><td>{values.get('extra_prose', 0)}</td></tr>"
        for side in ("base", "adapted")
        for values in [target.get(side, {})]
    )
    field_rows = "".join(
        f"<tr><td>{html.escape(field)}</td><td>{_pct(score)}</td></tr>"
        for field, score in target.get("adapted", {}).get("field_accuracy", {}).items()
    )
    training_html = (
        f"<h2>Training configuration</h2><pre>{html.escape(json.dumps(training_config, indent=2))}</pre>"
        if training_config
        else ""
    )
    dataset_info = html.escape(
        json.dumps(
            {
                "version": comparison.get("dataset_version", manifest.get("dataset_version")),
                "seed": comparison.get("dataset_seed", manifest.get("seed")),
                "hashes": comparison.get(
                    "dataset_hashes", manifest.get("dataset_hashes", "Not recorded")
                ),
                "model_revision": comparison.get(
                    "resolved_model_revision", manifest.get("resolved_model_revision")
                ),
                "environment": comparison.get("environment"),
            },
            indent=2,
        )
    )
    suite_rows_html = ""
    for suite, title in (
        ("format", "Format adherence"),
        ("unsupported", "Missing-data abstention"),
        ("unsupported_capability", "Unsupported-capability requests"),
    ):
        selected = suites.get(suite, [])
        for side in ("base", "adapted"):
            valid = [row for row in selected if row[f"{side}_passed"] is not None]
            passed = sum(row[f"{side}_passed"] is True for row in valid)
            formatted = [row for row in valid if row[f"{side}_format_valid"] is not None]
            valid_format = sum(row[f"{side}_format_valid"] is True for row in formatted)
            suite_rows_html += (
                f"<tr><td>{html.escape(title)}</td><td>{side.title()}</td>"
                f"<td>{passed}/{len(valid)}</td><td>{valid_format}/{len(formatted)}</td></tr>"
            )
    rows = "".join(
        f"<tr><td>{html.escape(r['case_id'])}</td><td>{html.escape(r['transition'])}</td><td>{html.escape(r['capability'])}</td><td>{html.escape(r['input'])}</td><td>{html.escape(r['base_output'])}</td><td>{html.escape(r['adapted_output'])}</td><td>{html.escape(r.get('base_error') or '')}</td><td>{html.escape(r.get('adapted_error') or '')}</td></tr>"
        for r in cases
    )
    metrics = "".join(
        f'<span class="metric">{html.escape(label)}: {_pct(value)}</span>'
        for label, value in [
            ("baseline accuracy", summary["baseline_accuracy"]),
            ("adapted accuracy", summary["adapted_accuracy"]),
            ("accuracy change", summary["accuracy_change"]),
        ]
    )
    interpretation = html.escape(comparison["methodology"]["interpretation"])
    banner_html = (
        f'<p style="padding:1rem;background:#fff1c2"><strong>{html.escape(demo_banner)}</strong></p>'
        if demo_banner
        else ""
    )
    page = (
        """<!doctype html><html><meta charset="utf-8"><title>FineTuning Regression Tester report</title>
<style>body{font:15px system-ui;max-width:1100px;margin:3rem auto;padding:0 1rem;color:#17212b}
table{border-collapse:collapse;width:100%;margin:1rem 0}td,th{border:1px solid #ccd3d8;padding:.5rem;text-align:left}
th{background:#edf2f4}.metric{display:inline-block;padding:.8rem 1rem;background:#edf2f4;margin:.3rem}</style>
<h1>FineTuning Regression Tester experiment report</h1>"""
        + banner_html
        + """<p>Observed behavior changes from paired evaluations.</p>
<section>"""
        + metrics
        + f"""</section><p>Model: {html.escape(str(model))} · Adapter: {html.escape(str(adapter))}</p>
<h2>Dataset and execution details</h2><pre>{dataset_info}</pre>
<p>Regressions: {summary["regressions"]} · Improvements: {summary["improvements"]} · Paired: {summary["paired_valid"]}/{summary["total_cases"]} · Valid base/adapted: {summary.get("baseline_valid", summary["paired_valid"])}/{summary.get("adapted_valid", summary["paired_valid"])}</p>
<p>Training time: {summary_time(comparison.get("timings", {}).get("training_seconds"))} ·
Baseline evaluation: {summary_time(comparison.get("timings", {}).get("baseline_evaluation_seconds"))} ·
Adapted evaluation: {summary_time(comparison.get("timings", {}).get("adapted_evaluation_seconds"))}</p>
<h2>Target extraction diagnostics</h2><table><thead><tr><th>Stage</th><th>Exact record</th><th>JSON valid</th><th>Schema valid</th><th>Unexpected fields</th><th>Extra prose</th></tr></thead><tbody>{target_rows}</tbody></table>
<h3>Adapted field accuracy</h3><table><thead><tr><th>Field</th><th>Accuracy</th></tr></thead><tbody>{field_rows}</tbody></table>
<h2>Capability map</h2><table><thead><tr><th>Capability</th><th>Base</th><th>Adapted</th><th>Change</th><th>Regressions</th><th>Improvements</th></tr></thead><tbody>{table}</tbody></table>
<h2>Format and abstention results</h2><table><thead><tr><th>Suite</th><th>Model stage</th><th>Semantically correct</th><th>Format valid</th></tr></thead><tbody>{suite_rows_html}</tbody></table>
{training_html}
<h2>Paired cases</h2><table><thead><tr><th>ID</th><th>Transition</th><th>Capability</th><th>Input</th><th>Base output</th><th>Adapted output</th><th>Base error</th><th>Adapted error</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Interpretation limits</h2><p>{interpretation}</p>
<h2>Reproduction</h2><p>Use the saved raw results and evaluation cases to recompute this comparison without model inference.</p>
<pre>finetuning-regression-tester compare --baseline PATH/baseline_results.jsonl --adapted PATH/adapted_results.jsonl --output PATH/comparison.json</pre>
</html>"""
    )
    paths[1].write_text(page)
    with paths[2].open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cases[0]) if cases else ["case_id"])
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            }
            for row in cases
        )
    paths[3].write_text(json.dumps(comparison["capabilities"], indent=2) + "\n")
    return paths


def _pct(value: Any) -> str:
    return "N/A" if value is None else f"{value:+.1%}" if abs(value) < 1 else f"{value:.1%}"


def summary_time(value: Any) -> str:
    return "Not recorded" if value is None else f"{value:.1f} seconds"
