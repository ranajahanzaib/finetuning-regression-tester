import json
import sys
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from finetuning_regression_tester.cli import app
from finetuning_regression_tester.config import load_config
from finetuning_regression_tester.core import (
    Case,
    Product,
    Result,
    build_suites,
    compare,
    digest,
    eval_cases,
    generate_dataset,
    grade,
    read_jsonl,
    write_jsonl,
)
from finetuning_regression_tester.mlx import MLXRunner, resolve_model
from finetuning_regression_tester.report import export_reports


def test_product_schema_decimal_and_nullable_fields():
    value = Product.model_validate(
        {
            "product_name": "TrailPack 20L",
            "brand": "Acme",
            "price": "79.99",
            "currency": "USD",
            "color": "navy blue",
            "size": "20L",
            "quantity": None,
        }
    )
    assert value.price == "79.99"
    with pytest.raises(ValueError):
        Product.model_validate({**value.model_dump(), "price": 79.99})
    with pytest.raises(ValueError):
        Product.model_validate({**value.model_dump(), "quantity": 0})


def test_generator_is_deterministic_and_split_independent(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    a = generate_dataset(first, 30, 10, 12, seed=8)
    b = generate_dataset(second, 30, 10, 12, seed=8)
    assert a == b
    assert read_jsonl(first / "train.jsonl") == read_jsonl(second / "train.jsonl")
    assert len(read_jsonl(first / "target_cases.jsonl")) == 12
    assert len({r["case_id"] for r in read_jsonl(first / "target_cases.jsonl")}) == 12
    canonical = next(
        r
        for r in read_jsonl(first / "target_cases.jsonl")
        if r["case_id"] == "product-canonical-example"
    )
    assert canonical["expected"]["product_name"] == "TrailPack 20L"
    assert canonical["expected"]["brand"] == "Acme"
    prompts = []
    for split in ("train", "valid", "test"):
        prompts.extend(
            row["messages"][0]["content"].casefold() for row in read_jsonl(first / f"{split}.jsonl")
        )
    assert len(prompts) == len(set(prompts))
    generated_targets = read_jsonl(first / "target_cases.jsonl")
    assert any(
        case["expected"]["color"] == "black, silver" and case["expected"]["quantity"] == 2
        for case in generated_targets
    )
    assert digest(read_jsonl(first / "train.jsonl")) == a["hashes"]["train"]


def test_evaluation_suite_counts_and_distinct_format_contracts():
    cases = build_suites()
    assert sum(case.suite == "retention" for case in cases) == 80
    assert sum(case.suite == "format" for case in cases) == 12
    assert sum(case.suite == "unsupported" for case in cases) == 20
    assert sum(case.suite == "unsupported_capability" for case in cases) == 10
    assert sum(case.suite == "sentinel" for case in cases) == 10
    modes = {case.grading["mode"] for case in cases if case.suite == "format"}
    assert {"json", "csv", "boolean", "numbered", "json_subset", "plain"} <= modes


@pytest.mark.parametrize(
    ("mode", "expected", "output", "passed", "format_valid"),
    [
        ("csv", {"city": "Oslo", "country": "Norway"}, "Oslo,Norway", True, True),
        ("csv", {"city": "Oslo", "country": "Norway"}, "Oslo", False, False),
        ("boolean", "true", "true", True, True),
        ("boolean", "true", "false", False, True),
        ("boolean", "true", "YES", False, False),
        ("numbered", ["1. oak", "2. pine", "3. elm"], "1. oak\n2. pine\n3. elm", True, True),
        ("json_subset", {"name": "Mira"}, '{"name":"Mira"}', True, True),
        ("json_subset", {"name": "Mira"}, '{"name":"Jon"}', False, True),
        ("json_subset", {"name": "Mira"}, '{"name":"Mira","city":"Oslo"}', False, True),
        ("plain", "Status: ready", "**Status: ready**", False, False),
    ],
)
def test_format_graders_separate_syntax_semantics_and_adherence(
    mode, expected, output, passed, format_valid
):
    case = Case(
        case_id="format-test",
        input="test",
        expected=expected,
        suite="format",
        capability="format_test",
        grading={"mode": mode},
    )
    result = grade(case, output)
    assert result["passed"] is passed
    assert result["format_valid"] is format_valid
    assert "semantic_correct" in result["details"]
    assert "instruction_adherent" in result["details"]


def test_missing_data_and_unsupported_capability_are_distinct():
    cases = {
        case.capability: case for case in build_suites() if case.suite.startswith("unsupported")
    }
    missing = cases["missing_data_abstention"]
    unsupported = cases["unsupported_capability_refusal"]
    assert grade(missing, "The description does not specify this.")["passed"]
    assert not grade(missing, "It is probably one year.")["passed"]
    assert grade(unsupported, "I can't access live weather information.")["passed"]
    assert not grade(unsupported, "It is sunny in Oslo.")["passed"]


def test_model_resolution_is_local_only_unless_download_is_explicit(monkeypatch):
    calls = []
    snapshot = "/fake/cache/models--org--model/snapshots/abc123"

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return snapshot

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    runner = MLXRunner("org/model")
    assert runner.resolve() == snapshot
    assert runner.metadata()["revision"] == "abc123"
    assert calls[-1]["local_files_only"] is True
    assert resolve_model("org/model", download=True) == snapshot
    assert calls[-1]["local_files_only"] is False


def test_yaml_configuration_rejects_unknown_and_unbounded_values(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("training:\n  iters: 100000\n")
    with pytest.raises(ValueError):
        load_config(config_path)
    config_path.write_text("training:\n  iterations: 100000\n")
    with pytest.raises(ValueError):
        load_config(config_path)


def test_strict_product_json_detects_prose_null_and_wrong_fields():
    expected = {
        "product_name": "Thing",
        "brand": None,
        "price": None,
        "currency": None,
        "color": None,
        "size": None,
        "quantity": None,
    }
    c = Case(
        case_id="x",
        input="x",
        expected=expected,
        suite="target",
        capability="product_extraction",
        grading={"strict_json": True},
    )
    assert grade(c, json.dumps(expected))["passed"]
    assert not grade(c, "Here is JSON: " + json.dumps(expected))["format_valid"]
    result = grade(c, json.dumps({**expected, "brand": "Invented"}))
    assert not result["passed"]
    assert result["details"]["field_accuracy"]["brand"] is False


def _result(case, passed):
    return Result(
        case_id=case.case_id,
        suite=case.suite,
        capability=case.capability,
        input=case.input,
        expected=case.expected,
        raw_output="ok",
        passed=passed,
    )


def test_paired_transitions_and_zero_baseline_denominator():
    cases = build_suites()[:4]
    base = [_result(c, p) for c, p in zip(cases, [True, True, False, False])]
    adapted = [_result(c, p) for c, p in zip(cases, [True, False, True, False])]
    comp = compare(base, adapted)
    transitions = {r["transition"] for r in comp["cases"]}
    assert {"PRESERVED_SUCCESS", "REGRESSION", "IMPROVEMENT", "PRESERVED_FAILURE"} <= transitions
    assert comp["summary"]["accuracy_change"] == 0
    assert comp["summary"]["regressions"] == comp["summary"]["improvements"] == 1
    zero = compare([_result(c, False) for c in cases], [_result(c, False) for c in cases])
    assert zero["summary"]["forgetting_rate"] is None
    assert zero["summary"]["retention_rate"] is None
    errored = _result(cases[0], False).model_copy(
        update={"raw_output": "", "error": "RuntimeError: model unavailable"}
    )
    partial = compare([_result(cases[0], True)], [errored])
    assert partial["cases"][0]["transition"] == "INCONCLUSIVE"
    assert partial["summary"]["baseline_accuracy"] == 1
    assert partial["summary"]["adapted_accuracy"] is None
    assert partial["summary"]["accuracy_change"] is None
    assert partial["summary"]["retention_rate"] is None


def test_inconclusive_separated_and_config_mismatch_rejected():
    cases = build_suites()[:2]
    base = [_result(c, p) for c, p in zip(cases, [True, False])]
    adapted = [_result(c, p) for c, p in zip(cases, [None, False])]
    assert compare(base, adapted)["summary"]["inconclusive"] == 1
    with pytest.raises(ValueError, match="configurations differ"):
        compare(base, adapted, {"temperature": 0}, {"temperature": 1})
    changed_case = Result.model_validate(
        {**adapted[0].model_dump(), "input": "different paired input"}
    )
    with pytest.raises(ValueError, match="content differs"):
        compare(base, [changed_case, *adapted[1:]])


def test_fake_evaluation_resumes_and_preserves_prompt(tmp_path):
    case = Case(
        case_id="target-one",
        input="A navy mug costs $4.",
        expected={
            "product_name": "mug",
            "brand": None,
            "price": "4",
            "currency": "USD",
            "color": "navy",
            "size": None,
            "quantity": None,
        },
        suite="target",
        capability="product_extraction",
    )

    class Runner:
        def __init__(self):
            self.calls = 0
            self.unloaded = False

        def generate(self, prompt):
            self.calls += 1
            assert "Return only one JSON object" in prompt
            return json.dumps(case.expected)

        def unload(self):
            self.unloaded = True

    runner = Runner()
    path = tmp_path / "results.jsonl"
    first = eval_cases([case], runner, path)
    assert runner.calls == 1 and runner.unloaded
    assert first[0].passed is True
    assert first[0].prompt and "A navy mug" in first[0].prompt
    second_runner = Runner()
    resumed = eval_cases([case], second_runner, path)
    assert second_runner.calls == 0
    assert resumed[0].raw_output == first[0].raw_output


def test_target_metrics_and_offline_report_exports(tmp_path):
    case = Case(
        case_id="target-one",
        input="Thing",
        expected={
            "product_name": "Thing",
            "brand": None,
            "price": None,
            "currency": None,
            "color": None,
            "size": None,
            "quantity": None,
        },
        suite="target",
        capability="product_extraction",
    )
    expected = case.expected
    base = Result(
        case_id=case.case_id,
        suite=case.suite,
        capability=case.capability,
        input=case.input,
        expected=expected,
        raw_output="not json",
        passed=False,
        format_valid=False,
        grading={"extra_prose": True},
    )
    adapted = Result(
        case_id=case.case_id,
        suite=case.suite,
        capability=case.capability,
        input=case.input,
        expected=expected,
        raw_output=json.dumps(expected),
        passed=True,
        format_valid=True,
        parsed=expected,
        grading={
            "schema_valid": True,
            "field_accuracy": {key: True for key in expected},
            "unexpected_fields": [],
        },
    )
    comparison = compare([base], [adapted])
    comparison["demo_data"] = True
    assert comparison["target_analysis"]["adapted"]["exact_record_accuracy"] == 1
    assert comparison["target_analysis"]["adapted"]["field_accuracy"]["brand"] == 1
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(comparison))
    paths = export_reports(path, tmp_path / "exports")
    assert all(item.is_file() for item in paths)
    assert "Target extraction diagnostics" in paths[0].read_text()
    assert "DEMO DATA" in paths[0].read_text()
    assert "DEMO DATA" in paths[1].read_text()
    assert "PRESERVED" not in paths[1].read_text()


def test_demo_command_uses_labeled_checked_in_fixtures(tmp_path):
    output = tmp_path / "demo"
    result = CliRunner().invoke(app, ["demo", "--output", str(output)])
    assert result.exit_code == 0, result.output
    comparison = json.loads((output / "comparison.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert comparison["demo_data"] is True
    assert manifest["demo_data"] is True
    assert "DEMO DATA" in (output / "report.html").read_text()


def test_validator_detects_train_target_contamination(tmp_path):
    dataset = tmp_path / "dataset"
    manifest = generate_dataset(dataset, 12, 6, 8)
    train = read_jsonl(dataset / "train.jsonl")
    target = read_jsonl(dataset / "target_cases.jsonl")[0]
    train[0]["messages"][0]["content"] = "Extract product fields.\n\n" + target["input"]
    from finetuning_regression_tester.core import write_json

    write_jsonl(dataset / "train.jsonl", train)
    manifest["hashes"]["train"] = digest(train)
    write_json(dataset / "dataset_manifest.json", manifest)
    result = CliRunner().invoke(app, ["data", "validate", "--dataset", str(dataset)])
    assert result.exit_code != 0
    assert "contamination" in result.output
