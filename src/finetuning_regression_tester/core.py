"""Typed data, deterministic generation, grading, and paired analysis."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import time
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

FIELDS = ("product_name", "brand", "price", "currency", "color", "size", "quantity")
DEFAULT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


class Product(BaseModel):
    """Normalized product record; price is a decimal string, never a float."""

    model_config = ConfigDict(extra="forbid")
    product_name: str | None
    brand: str | None
    price: str | None = Field(pattern=r"^\d+(?:\.\d{1,2})?$")
    currency: str | None
    color: str | None
    size: str | None
    quantity: int | None = Field(default=None, gt=0)


class Case(BaseModel):
    case_id: str
    input: str
    expected: Any
    suite: str
    capability: str
    difficulty: str = "standard"
    metadata: dict[str, Any] = Field(default_factory=dict)
    grading: dict[str, Any] = Field(default_factory=dict)


class Result(BaseModel):
    case_id: str
    suite: str
    capability: str
    input: str
    expected: Any
    raw_output: str
    passed: bool | None
    format_valid: bool | None = None
    parsed: Any = None
    error: str | None = None
    duration_seconds: float = 0
    grading: dict[str, Any] = Field(default_factory=dict)
    prompt: str | None = None
    severity: str = "medium"


class ModelRunner(Protocol):
    def load(self) -> None: ...

    def generate(self, prompt: str) -> str: ...

    def unload(self) -> None: ...

    def metadata(self) -> dict[str, Any]: ...


class Transition(StrEnum):
    PRESERVED_SUCCESS = "PRESERVED_SUCCESS"
    REGRESSION = "REGRESSION"
    IMPROVEMENT = "IMPROVEMENT"
    PRESERVED_FAILURE = "PRESERVED_FAILURE"
    INCONCLUSIVE = "INCONCLUSIVE"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    temp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("".join(canonical(row) + "\n" for row in rows))
    temp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: each JSONL record must be an object")
        rows.append(value)
    return rows


def _product_record(i: int, split: str) -> tuple[str, dict[str, Any], str, str]:
    # Split-specific entities and template families prevent identity/template leakage.
    brands = {
        "train": ["Summit House", "NovaTech", "FieldWorks", "Morrow"],
        "valid": ["Northstar", "Pine & Co", "Cinder"],
        "test": ["Lumen", "Redwood", "Orbit", "Marble"],
    }[split]
    kinds = [
        ("NoiseCancel Pro", "headphones", "black", None, 129, "electronics"),
        ("TrailPack 20L", "backpack", "navy blue", "20L", 79.99, "backpacks"),
        ("SilverKey USB drive", "USB drive", "silver", "32GB", 14.5, "electronics"),
        ("Everyday Tee", "shirt", "forest green", "M", 24, "clothing"),
        ("RunFlex 3", "running shoes", "white", "10", 88, "footwear"),
        ("Stoneware Set", "dinnerware set", "cream", None, 42.5, "kitchenware"),
        ("Arc Lamp", "desk lamp", "brass", None, 56, "furniture"),
        ("Daily Balm", "skin balm", "unscented", "50ml", 12, "personal care"),
        ("CourtPro Paddle", "paddle", "blue", None, 64, "sporting goods"),
        ("Grid Notebook", "notebook", "red", "A5", 9.75, "office accessories"),
    ]
    split_offsets = {"train": 0, "valid": 1001, "test": 2003}
    entity_id = split_offsets[split] + i
    name, category, color, size, price, family = kinds[entity_id % len(kinds)]
    name = f"{name} Series {entity_id // len(kinds) + 1}"
    brand = brands[(entity_id // len(kinds)) % len(brands)]
    color_variants = {
        "electronics": ["black", "graphite", "silver", "deep blue"],
        "backpacks": ["navy blue", "olive", "charcoal", "sand"],
        "clothing": ["forest green", "black", "navy", "cream"],
        "footwear": ["white", "black", "gray", "red"],
        "kitchenware": ["cream", "blue", "sage", "charcoal"],
        "furniture": ["brass", "black", "white", "walnut"],
        "personal care": ["unscented", "lavender", "citrus", "rose"],
        "sporting goods": ["blue", "red", "black", "green"],
        "office accessories": ["red", "blue", "black", "kraft"],
    }
    color = color_variants[family][entity_id % 4]
    if size is not None:
        size_variants = {
            "backpacks": ["18L", "20L", "24L", "30L"],
            "electronics": ["16GB", "32GB", "64GB", "128GB"],
            "clothing": ["S", "M", "L", "XL"],
            "footwear": ["8", "9", "10", "11"],
            "personal care": ["30ml", "50ml", "75ml", "100ml"],
            "office accessories": ["A4", "A5", "B5", "Letter"],
        }
        size = size_variants[family][entity_id % 4]
    price_value = Decimal(str(price)) + Decimal(entity_id % 5) * Decimal("0.25")
    price_text = (
        str(price_value.quantize(Decimal("1")))
        if price_value == price_value.to_integral_value()
        else f"{price_value:.2f}"
    )
    quantity = (entity_id % 3) + 1 if entity_id % 7 == 0 else None
    templates = {
        "train": [
            "{brand} {name} {category}, {color}, ${price}.",
            "Get the {color} {name} by {brand} for ${price}.",
            "Introducing {brand}'s {name}: {category} in {color}, priced at ${price}.",
            "{name} from {brand} costs ${price}; color: {color}.",
        ],
        "valid": [
            "{brand} offers the {color} {name} ({category}) at ${price}.",
            "For ${price}, choose {brand} {name} in {color}.",
            "The {name} by {brand} is a {color} {category}, ${price}.",
        ],
        "test": [
            "Meet {name}, a {color} {category} from {brand}, available for ${price}.",
            "${price} gets you {brand}'s {name} in {color}.",
            "Available in {color}: {name} by {brand} ({category}), price ${price}.",
        ],
    }
    template = templates[split][i % len(templates[split])]
    wording_family = f"{split}_template_{i % len(templates[split])}"
    text = template.format(brand=brand, name=name, category=category, color=color, price=price_text)
    if size:
        text += f" Size {size}."
    # Explicitly generated ambiguity/missing values; still grounded in records.
    if i % 17 == 6:
        wording_family = f"{split}_multicolor_bundle"
        color = "black, silver"
        quantity = 2
        if split == "train":
            text = f"By {brand}, {name} ({category}) is stocked in black and silver at ${price_text}; bundle has 2 units."
        elif split == "valid":
            text = f"Colorways: black and silver. {category.title()}: {brand} {name}. Price ${price_text}; two units per pack."
        else:
            text = f"{brand} lists {name} as a black and silver {category}; ${price_text}; bundle of 2."
    elif i % 11 == 3:
        wording_family = f"{split}_missing_brand_price"
        name = f"{split.title()} {name}"
        if split == "train":
            text = f"Listing: {name} — {category}; color {color}."
        elif split == "valid":
            text = f"In {color}, this {category} is called {name}."
        else:
            text = f"{name} / {category} / shade={color}."
        text += f" Size {size}." if size else ""
        brand = None
        price_text = None
    elif i % 13 == 5:
        wording_family = f"{split}_product_code"
        if split == "train":
            text = f"Limited offer: {brand} {name} in {color}; now ${price_text}. Product code T{i:04d}."
        elif split == "valid":
            text = f"Code V{i:04d} — {name} by {brand}. {category.title()}, shade {color}, price ${price_text}."
        else:
            text = f"{brand} {name} ({category}), {color}; ${price_text}. Reference E{i:04d}."
    elif i % 17 == 4:
        wording_family = f"{split}_missing_brand"
        name = f"{split.title()} {name}"
        if split == "train":
            text = f"{name} — {color} {category}; price ${price_text}."
        elif split == "valid":
            text = f"Price ${price_text}. {category.title()} {name}, shade {color}."
        else:
            text = f"For ${price_text}, choose the {color} {category} named {name}."
        brand = None
    elif i % 19 == 5:
        wording_family = f"{split}_missing_price"
        if split == "train":
            text = f"{brand} {name} ({category}), {color}; the seller has not listed a price."
        elif split == "valid":
            text = f"Price unavailable. {color} {category}: {name} from {brand}."
        else:
            text = f"{name}, by {brand}; color {color}; no price information is provided."
        price_text = None
    if i % 23 == 9:
        wording_family += "_long_description"
        long_details = {
            "train": "Built for daily use, with a durable finish and carefully considered details.",
            "valid": "Its materials are selected for regular use, and the finish is easy to maintain.",
            "test": "Made with practical construction for everyday routines and repeated handling.",
        }
        text += " " + long_details[split]
    elif i % 23 == 10:
        wording_family += "_informal"
        text = f"hey—{text} sounds like a great deal!"
    expected = {
        "product_name": name,
        "brand": brand,
        "price": price_text,
        "currency": "USD" if price_text else None,
        "color": color,
        "size": size,
        "quantity": quantity,
    }
    # The sentence template does not state quantity; keep it missing unless stated.
    if quantity:
        text += f" Pack of {quantity}."
    return text, expected, wording_family, "standard" if i % 4 else "challenging"


def generate_dataset(
    output: Path, train: int = 200, valid: int = 40, test: int = 60, seed: int = 42
) -> dict[str, Any]:
    if min(train, valid, test) < 1:
        raise ValueError("train, valid, and test counts must each be at least 1")
    output.mkdir(parents=True, exist_ok=True)
    counts = {"train": train, "valid": valid, "test": test}
    eval_cases: list[dict[str, Any]] = []
    hashes = {}
    for split, count in counts.items():
        mlx_rows, cases = [], []
        indices = list(range(count))
        random.Random(seed + {"train": 0, "valid": 1, "test": 2}[split]).shuffle(indices)
        for i in indices:
            text, expected, family, difficulty = _product_record(i, split)
            mlx_rows.append(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Extract product fields as JSON. Use null when unknown; join multiple colors in source order with comma and space.\n\n"
                            + text,
                        },
                        {"role": "assistant", "content": canonical(expected)},
                    ]
                }
            )
            if split == "test":
                cases.append(
                    Case(
                        case_id=f"product-{i:04d}",
                        input=text,
                        expected=expected,
                        suite="target",
                        capability="product_extraction",
                        difficulty=difficulty,
                        metadata={"template_family": family, "entity_split": split},
                        grading={"strict_json": True},
                    ).model_dump()
                )
        write_jsonl(output / f"{split}.jsonl", mlx_rows)
        if split == "test":
            # Reserve one target evaluation slot for the plan's canonical
            # acceptance example while keeping the requested case count exact.
            eval_cases.extend(cases[:-1])
        hashes[split] = digest(mlx_rows)
    # Keep the PLAN's canonical product example as a held-out, human-readable
    # acceptance case. Acme is excluded from the training brand pool above.
    canonical_case = Case(
        case_id="product-canonical-example",
        input="Introducing the Acme TrailPack 20L backpack. Available in navy blue for $79.99.",
        expected={
            "product_name": "TrailPack 20L",
            "brand": "Acme",
            "price": "79.99",
            "currency": "USD",
            "color": "navy blue",
            "size": "20L",
            "quantity": None,
        },
        suite="target",
        capability="product_extraction",
        difficulty="standard",
        metadata={"template_family": "canonical_acceptance", "entity_split": "test"},
        grading={"strict_json": True},
    ).model_dump()
    eval_cases.append(canonical_case)
    write_jsonl(output / "target_cases.jsonl", eval_cases)
    suites = build_suites()
    write_jsonl(output / "retention_cases.jsonl", [c.model_dump() for c in suites])
    combined_cases = eval_cases + [c.model_dump() for c in suites]
    write_jsonl(output / "evaluation_cases.jsonl", combined_cases)
    hashes["target_cases"] = digest(eval_cases)
    hashes["retention_cases"] = digest([c.model_dump() for c in suites])
    hashes["evaluation_cases"] = digest(combined_cases)
    manifest = {
        "generator_version": "1",
        "seed": seed,
        "counts": counts,
        "hashes": hashes,
        "currency_normalization": "Only explicit $ denotes USD. Prices are decimal strings; no locale or exchange-rate inference.",
        "leakage_policy": "Product identities and wording template families are partitioned by split before generation.",
    }
    write_json(output / "dataset_manifest.json", manifest)
    return manifest


def build_suites() -> list[Case]:
    cases: list[Case] = []

    def add(
        suite: str, capability: str, prompt: str, expected: Any, mode: str = "exact", **meta: Any
    ) -> None:
        cid = f"{suite}-{len(cases):03d}"
        cases.append(
            Case(
                case_id=cid,
                input=prompt,
                expected=expected,
                suite=suite,
                capability=capability,
                metadata={"grader": mode, "severity": meta.pop("severity", "medium"), **meta},
                grading={"mode": mode},
            )
        )

    # 80 objective general capability cases, balanced over five capability families.
    for i in range(16):
        a, b = 17 + i * 3, 4 + i
        op = ["+", "-", "*", "%", "word"][i % 5]
        if op == "+":
            prompt, answer = f"What is {a} + {b}? Reply with the number only.", str(a + b)
        elif op == "-":
            prompt, answer = f"What is {a} - {b}? Reply with the number only.", str(a - b)
        elif op == "*":
            prompt, answer = f"What is {a} * {b}? Reply with the number only.", str(a * b)
        elif op == "%":
            prompt, answer = f"What is 10 percent of {a * 10}? Reply with the number only.", str(a)
        else:
            prompt, answer = (
                f"A box has {a} pencils. Add {b} more. How many pencils? Number only.",
                str(a + b),
            )
        add("retention", "arithmetic", prompt, answer, "number")
    for i in range(16):
        if i % 4 == 0:
            prompt, ans = f"Answer YES or NO only: Is {i} even?", "YES" if i % 2 == 0 else "NO"
        elif i % 4 == 1:
            prompt, ans = (
                f"Return exactly three words describing the color {['amber', 'blue', 'green', 'violet'][i % 4]}.",
                ["bright warm amber", "calm clear blue", "fresh vivid green", "deep rich violet"][
                    i % 4
                ],
            )
        elif i % 4 == 2:
            prompt, ans = (
                f"Return exactly two items as '1. X 2. Y': first={i}, second={i + 1}.",
                f"1. {i} 2. {i + 1}",
            )
        else:
            prompt, ans = (
                f"Use template 'status:ready; code:{i}' exactly.",
                f"status:ready; code:{i}",
            )
        add(
            "retention",
            "instruction_following",
            prompt,
            ans,
            "normalized" if i % 4 == 1 else "exact",
        )
    for i in range(16):
        words = [
            "Small foxes cross quiet rivers.",
            "Blue lamps glow beside the window.",
            "A bright kite flies above fields.",
            "Careful notes make work easier.",
        ]
        src = words[i % len(words)]
        mode = i % 4
        prompt, ans = [
            (f"Convert exactly to uppercase: {src}", src.upper()),
            (f"Normalize spaces to single spaces, preserve punctuation: {src}", src),
            (
                f"Return the first and last word separated by a comma: {src}",
                f"{src.split()[0].strip('.,')},{src.split()[-1].strip('.,')}",
            ),
            (
                f"Remove punctuation and lowercase this: {src}",
                re.sub(r"[^a-zA-Z0-9 ]", "", src).lower(),
            ),
        ][mode]
        add("retention", "text_transformation", prompt, ans, "exact")
    for i in range(16):
        if i % 2 == 0:
            day = i + 1
            prompt, ans = (
                f"Return JSON with date from: The meeting is on 2027-04-{day:02d}.",
                {"date": f"2027-04-{day:02d}"},
            )
        else:
            prompt, ans = (
                f"Extract the person and city as JSON: {['Mira Chen visited Oslo.', 'Jon Bell moved to Quito.', 'Asha Noor works in Accra.', 'Leo Park arrived in Riga.'][i % 4]}",
                {
                    "person": ["Jon Bell", "Asha Noor", "Leo Park", "Mira Chen"][i % 4],
                    "city": ["Quito", "Accra", "Riga", "Oslo"][i % 4],
                },
            )
        add("retention", "general_extraction", prompt, ans, "json")
    sentiments = [
        ("I love the thoughtful design.", "positive"),
        ("The service was awful and slow.", "negative"),
        ("The package arrived today.", "neutral"),
        ("A delightful and useful tool.", "positive"),
    ]
    for i in range(16):
        if i % 2 == 0:
            text, label = sentiments[(i // 2) % 4]
            prompt, ans = (
                f"Classify sentiment as positive, negative, or neutral. One word only: {text}",
                label,
            )
        else:
            n = i // 2
            prompt, ans = (
                f"Rule: values at least 10 are HIGH, otherwise LOW. Classify {n * 3}.",
                "HIGH" if n * 3 >= 10 else "LOW",
            )
        add("retention", "classification", prompt, ans, "normalized")
    # Format and missing-data suites plus sentinel subset are independent and additive.
    for i in range(2):
        expected = {"ok": bool(i % 2), "count": i}
        add(
            "format",
            "structured_json",
            f"Return only JSON with keys ok (boolean) and count (number): ok={str(bool(i % 2)).lower()}, count={i}.",
            expected,
            "json",
            severity="high",
        )
    for city, country in (("Oslo", "Norway"), ("Quito", "Ecuador")):
        add(
            "format",
            "csv_output",
            f"Return exactly one CSV row without a header, columns city,country: {city}; {country}.",
            {"city": city, "country": country},
            "csv",
            severity="high",
        )
    for value in (True, False):
        add(
            "format",
            "boolean_token",
            f"Return only the lowercase boolean token for {str(value).lower()}.",
            str(value).lower(),
            "boolean",
            severity="high",
        )
    for items in (("oak", "pine", "elm"), ("red", "blue", "green")):
        expected = [f"{i}. {item}" for i, item in enumerate(items, 1)]
        add(
            "format",
            "numbered_items",
            "Return exactly three numbered items, one per line: " + ", ".join(items) + ".",
            expected,
            "numbered",
            severity="high",
        )
    for name, value in (("Mira", "Oslo"), ("Jon", "Quito")):
        add(
            "format",
            "json_subset",
            f"Return JSON with exactly the key name and its value from: {name} visited {value}.",
            {"name": name},
            "json_subset",
            severity="high",
        )
    for status in ("ready", "complete"):
        expected = f"Status: {status}"
        add(
            "format",
            "no_markdown",
            f"Return exactly this plain text without Markdown: {expected}",
            expected,
            "plain",
            severity="high",
        )
    for i in range(10):
        add(
            "unsupported",
            "missing_data_abstention",
            f"What is the warranty period? Description: Product {i} is made of steel.",
            "abstain",
            "abstain",
        )
    for request in (
        "Give me the live weather in Oslo right now.",
        "Check today's exchange rate for USD to EUR.",
        "Open the current front page of example.com and summarize it.",
        "Tell me the latest score in the match happening now.",
        "Search the web for the newest release of Python.",
        "Look up the current traffic on the road to the airport.",
        "Read my private email inbox and find the latest message.",
        "Check the live stock price for ACME.",
        "Book a table at a restaurant for tonight.",
        "Inspect a file on my computer that I have not uploaded.",
    ):
        add(
            "unsupported_capability",
            "unsupported_capability_refusal",
            request,
            "unsupported",
            "unsupported_capability",
            severity="medium",
            abstention_type="unsupported_capability",
        )
    for i in range(10):
        add(
            "unsupported",
            "unsupported_requests",
            f"What is the live inventory count for item {i}? No inventory data is supplied.",
            "abstain",
            "abstain",
        )
    sentinels = [
        ("What is 19 + 23? Number only.", "42", "number", "arithmetic"),
        ("Return YES or NO only: is 81 divisible by 9?", "YES", "exact", "instruction_following"),
        ("Uppercase exactly: quiet harbor", "QUIET HARBOR", "exact", "text_transformation"),
        (
            "Return JSON only with key city: fictional city Larkspur.",
            {"city": "Larkspur"},
            "json",
            "general_extraction",
        ),
        (
            "Label this sentiment positive or negative: The repair was excellent.",
            "positive",
            "normalized",
            "classification",
        ),
        ("What is 56 - 17? Number only.", "39", "number", "arithmetic"),
        (
            "Return exactly two words: name the season after summer.",
            "early autumn",
            "exact",
            "instruction_following",
        ),
        (
            'Return JSON only: {"ready":true,"count":3}',
            {"ready": True, "count": 3},
            "json",
            "structured_json",
        ),
        (
            "Remove punctuation and lowercase: Hello, World!",
            "hello world",
            "exact",
            "text_transformation",
        ),
        (
            "Is the statement true under the rule 'numbers over 4 are large': 9? YES or NO only.",
            "YES",
            "exact",
            "classification",
        ),
    ]
    for prompt, expected, mode, capability in sentinels:
        add("sentinel", capability, prompt, expected, mode, severity="medium")
    return cases


def grade(case: Case, output: str) -> dict[str, Any]:
    mode = case.grading.get("mode", case.metadata.get("grader", "exact"))
    stripped = output.strip()
    if not stripped:
        return {"passed": False, "format_valid": False, "parsed": None, "details": {"empty": True}}
    if mode == "json":
        try:
            parsed = json.loads(stripped)
            valid = isinstance(parsed, (dict, list))
            ok = valid and parsed == case.expected
            adherent = valid and type(parsed) is type(case.expected)
            if isinstance(parsed, dict) and isinstance(case.expected, dict):
                adherent = set(parsed) == set(case.expected)
            elif isinstance(parsed, list) and isinstance(case.expected, list):
                adherent = len(parsed) == len(case.expected)
            return {
                "passed": ok,
                "format_valid": valid,
                "parsed": parsed,
                "details": {
                    "syntax_valid": valid,
                    "semantic_correct": ok,
                    "instruction_adherent": bool(adherent),
                    "semantic_match": ok,
                },
            }
        except json.JSONDecodeError:
            return {
                "passed": False,
                "format_valid": False,
                "parsed": None,
                "details": {
                    "syntax_valid": False,
                    "semantic_correct": False,
                    "instruction_adherent": False,
                    "strict_json": False,
                },
            }
    if case.suite == "target":
        try:
            parsed = json.loads(stripped)
            schema_ok = True
            try:
                Product.model_validate(parsed)
            except (ValidationError, TypeError):
                schema_ok = False
            fields = (
                {field: parsed.get(field) == case.expected.get(field) for field in FIELDS}
                if isinstance(parsed, dict)
                else {}
            )
            exact = fields and all(fields.values()) and not (set(parsed) - set(FIELDS))
            adherent = isinstance(parsed, dict) and schema_ok and not (set(parsed) - set(FIELDS))
            return {
                "passed": bool(exact and adherent),
                "format_valid": isinstance(parsed, dict),
                "parsed": parsed,
                "details": {
                    "syntax_valid": True,
                    "semantic_correct": bool(exact),
                    "instruction_adherent": bool(adherent),
                    "schema_valid": schema_ok,
                    "field_accuracy": fields,
                    "unexpected_fields": sorted(set(parsed) - set(FIELDS))
                    if isinstance(parsed, dict)
                    else [],
                    "extra_prose": False,
                },
            }
        except json.JSONDecodeError:
            extra_prose = not (stripped.startswith("{") and stripped.endswith("}"))
            return {
                "passed": False,
                "format_valid": False,
                "parsed": None,
                "details": {
                    "syntax_valid": False,
                    "schema_valid": False,
                    "extra_prose": extra_prose,
                },
            }
    if mode == "csv":
        try:
            parsed_rows = list(csv.reader(io.StringIO(stripped), strict=True))
            syntax_valid = len(parsed_rows) == 1 and len(parsed_rows[0]) == len(case.expected)
            parsed = dict(zip(case.expected, parsed_rows[0])) if syntax_valid else None
        except (csv.Error, ValueError):
            syntax_valid, parsed = False, None
        semantic = syntax_valid and parsed == case.expected
        return {
            "passed": bool(semantic),
            "format_valid": bool(syntax_valid),
            "parsed": parsed,
            "details": {
                "syntax_valid": bool(syntax_valid),
                "semantic_correct": bool(semantic),
                "instruction_adherent": bool(syntax_valid),
            },
        }
    if mode == "boolean":
        syntax_valid = stripped in {"true", "false"}
        semantic = syntax_valid and stripped == case.expected
        return {
            "passed": bool(semantic),
            "format_valid": syntax_valid,
            "parsed": stripped if syntax_valid else None,
            "details": {
                "syntax_valid": syntax_valid,
                "semantic_correct": bool(semantic),
                "instruction_adherent": syntax_valid,
            },
        }
    if mode == "numbered":
        lines = stripped.splitlines()
        syntax_valid = len(lines) == 3 and all(
            re.fullmatch(rf"{index}\. .+", line) for index, line in enumerate(lines, 1)
        )
        parsed = lines if syntax_valid else None
        semantic = syntax_valid and lines == case.expected
        return {
            "passed": bool(semantic),
            "format_valid": bool(syntax_valid),
            "parsed": parsed,
            "details": {
                "syntax_valid": bool(syntax_valid),
                "semantic_correct": bool(semantic),
                "instruction_adherent": bool(syntax_valid),
            },
        }
    if mode == "json_subset":
        try:
            parsed = json.loads(stripped)
            syntax_valid = isinstance(parsed, dict)
            semantic = syntax_valid and parsed == case.expected
        except json.JSONDecodeError:
            parsed, syntax_valid, semantic = None, False, False
        adherent = syntax_valid and set(parsed) == set(case.expected)
        return {
            "passed": bool(semantic),
            "format_valid": bool(syntax_valid),
            "parsed": parsed,
            "details": {
                "syntax_valid": bool(syntax_valid),
                "semantic_correct": bool(semantic),
                "instruction_adherent": bool(adherent),
                "unexpected_fields": sorted(set(parsed) - set(case.expected))
                if isinstance(parsed, dict)
                else [],
            },
        }
    if mode == "plain":
        markdown = bool(re.search(r"(^\s*#|```|\*\*|^\s*[-*+]\s)", stripped, re.MULTILINE))
        semantic = stripped == case.expected
        adherent = not markdown
        return {
            "passed": semantic and adherent,
            "format_valid": not markdown,
            "parsed": stripped,
            "details": {
                "syntax_valid": True,
                "semantic_correct": semantic,
                "instruction_adherent": adherent,
            },
        }
    if mode == "number":
        value = re.fullmatch(r"\s*[-+]?\d+(?:\.\d+)?\s*", output)
        parsed: Any = value.group().strip() if value else None
        passed = parsed == case.expected
    elif mode == "json":
        parsed, passed = None, False
    elif mode == "abstain":
        parsed = stripped.lower()
        passed = bool(
            re.search(
                r"\b(unknown|not (?:provided|specified|available|given|listed|mentioned|stated)|does not specify|does not state|insufficient|cannot determine|no information|not enough information|unspecified)\b",
                parsed,
            )
        )
    elif mode == "unsupported_capability":
        parsed = stripped.lower()
        passed = bool(
            re.search(
                r"\b(cannot|can't|unable|do not have|don't have|no access|not able|not available|cannot browse|can't browse|can't access)\b",
                parsed,
            )
        )
    elif mode == "normalized":
        parsed = re.sub(r"\s+", " ", stripped).casefold()
        expected = re.sub(r"\s+", " ", str(case.expected)).casefold()
        passed = parsed == expected
    else:
        parsed, passed = stripped, stripped == str(case.expected)
    return {"passed": passed, "format_valid": True, "parsed": parsed, "details": {"mode": mode}}


def eval_cases(
    cases: list[Case], runner: ModelRunner, result_path: Path | None = None
) -> list[Result]:
    existing = (
        {r["case_id"]: r for r in read_jsonl(result_path)}
        if result_path and result_path.exists()
        else {}
    )
    for case in cases:
        saved = existing.get(case.case_id)
        if saved and (saved.get("input") != case.input or saved.get("expected") != case.expected):
            raise ValueError(
                f"Cannot resume {case.case_id}: saved input or expected answer differs from current case"
            )
    results = [Result.model_validate(existing[c.case_id]) for c in cases if c.case_id in existing]
    done = set(existing)
    for case in cases:
        if case.case_id in done:
            continue
        start = time.perf_counter()
        try:
            prompt = (
                (
                    "Extract the product attributes from the user description. Return only one JSON object with exactly these keys: "
                    "product_name, brand, price, currency, color, size, quantity. Join multiple colors in source order with comma and space. "
                    "Use JSON null for every attribute not explicitly provided. "
                    "Keep price as a decimal string. Do not add commentary.\n\n" + case.input
                )
                if case.suite == "target"
                else case.input
            )
            output = runner.generate(prompt)
            duration = time.perf_counter() - start
            g = grade(case, output)
            result = Result(
                case_id=case.case_id,
                suite=case.suite,
                capability=case.capability,
                input=case.input,
                expected=case.expected,
                raw_output=output,
                passed=g["passed"],
                format_valid=g["format_valid"],
                parsed=g["parsed"],
                grading=g["details"],
                prompt=prompt,
                severity=case.metadata.get("severity", "medium"),
                duration_seconds=duration,
            )
        except Exception as exc:
            duration = time.perf_counter() - start
            result = Result(
                case_id=case.case_id,
                suite=case.suite,
                capability=case.capability,
                input=case.input,
                expected=case.expected,
                raw_output="",
                passed=None,
                error=f"{type(exc).__name__}: {exc}",
                prompt=prompt if "prompt" in locals() else case.input,
                severity=case.metadata.get("severity", "medium"),
                duration_seconds=duration,
            )
        results.append(result)
        if result_path:
            write_jsonl(result_path, [r.model_dump() for r in results])
    unload = getattr(runner, "unload", None)
    if unload:
        unload()
    return results


def compare(
    base: list[Result], adapted: list[Result], base_config: Any = None, adapted_config: Any = None
) -> dict[str, Any]:
    if base_config is not None and adapted_config is not None and base_config != adapted_config:
        raise ValueError("evaluation configurations differ; paired comparison is not valid")
    bmap, amap = {r.case_id: r for r in base}, {r.case_id: r for r in adapted}
    if len(bmap) != len(base) or len(amap) != len(adapted):
        raise ValueError("evaluation results contain duplicate case IDs")
    if bmap.keys() != amap.keys():
        raise ValueError("baseline and adapted results must have identical case IDs")
    cases, group = [], defaultdict(list)
    for cid in sorted(bmap):
        b, a = bmap[cid], amap[cid]
        if (b.input, b.expected, b.suite, b.capability) != (
            a.input,
            a.expected,
            a.suite,
            a.capability,
        ):
            raise ValueError(f"paired result content differs for case {cid}")
        if (
            b.passed is None
            or a.passed is None
            or b.error is not None
            or a.error is not None
            or not b.raw_output
            or not a.raw_output
        ):
            t = Transition.INCONCLUSIVE
        elif b.passed and a.passed:
            t = Transition.PRESERVED_SUCCESS
        elif b.passed:
            t = Transition.REGRESSION
        elif a.passed:
            t = Transition.IMPROVEMENT
        else:
            t = Transition.PRESERVED_FAILURE
        row = {
            "case_id": cid,
            "suite": b.suite,
            "capability": b.capability,
            "input": b.input,
            "expected": b.expected,
            "base_output": b.raw_output,
            "adapted_output": a.raw_output,
            "base_error": b.error,
            "adapted_error": a.error,
            "base_parsed": b.parsed,
            "adapted_parsed": a.parsed,
            "base_passed": b.passed,
            "adapted_passed": a.passed,
            "base_format_valid": b.format_valid,
            "adapted_format_valid": a.format_valid,
            "transition": t.value,
            "severity": b.severity if t == Transition.REGRESSION else None,
            "base_grading": b.grading,
            "adapted_grading": a.grading,
        }
        cases.append(row)
        group[b.capability].append(row)
    metrics = _metrics(cases)
    capabilities = {cap: _metrics(rows) for cap, rows in sorted(group.items())}
    return {
        "schema_version": "1",
        "generated_at": datetime.now(UTC).isoformat(),
        "compatible": True,
        "summary": metrics,
        "capabilities": capabilities,
        "target_analysis": _target_analysis(cases),
        "cases": cases,
        "methodology": {
            "accuracy": "Each model's successes divided by its own valid responses",
            "accuracy_change": "Paired adapted-minus-baseline accuracy on cases valid for both models",
            "forgetting_rate": "regressions / paired baseline successes; N/A if denominator is zero",
            "retention_rate": "preserved successes / paired baseline successes; N/A if denominator is zero",
            "interpretation": "Observed pass-to-fail transitions are operational regressions, not proof of permanent skill loss.",
        },
    }


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_valid = [
        r
        for r in rows
        if r["base_passed"] is not None and not r.get("base_error") and r["base_output"]
    ]
    adapted_valid = [
        r
        for r in rows
        if r["adapted_passed"] is not None and not r.get("adapted_error") and r["adapted_output"]
    ]
    paired_valid = [r for r in rows if r["transition"] != Transition.INCONCLUSIVE.value]
    bs = sum(r["base_passed"] is True for r in baseline_valid)
    ass = sum(r["adapted_passed"] is True for r in adapted_valid)
    paired_bs = sum(r["base_passed"] is True for r in paired_valid)
    paired_ass = sum(r["adapted_passed"] is True for r in paired_valid)
    reg = sum(r["transition"] == Transition.REGRESSION.value for r in rows)
    imp = sum(r["transition"] == Transition.IMPROVEMENT.value for r in rows)
    retained = sum(r["transition"] == Transition.PRESERVED_SUCCESS.value for r in rows)
    n = len(paired_valid)
    return {
        "total_cases": len(rows),
        "paired_valid": n,
        "inconclusive": len(rows) - n,
        "baseline_valid": len(baseline_valid),
        "adapted_valid": len(adapted_valid),
        "baseline_successes": bs,
        "adapted_successes": ass,
        "paired_baseline_successes": paired_bs,
        "paired_adapted_successes": paired_ass,
        "paired_baseline_accuracy": paired_bs / n if n else None,
        "paired_adapted_accuracy": paired_ass / n if n else None,
        "baseline_accuracy": bs / len(baseline_valid) if baseline_valid else None,
        "adapted_accuracy": ass / len(adapted_valid) if adapted_valid else None,
        "accuracy_change": (paired_ass - paired_bs) / n if n else None,
        "regressions": reg,
        "improvements": imp,
        "retention_rate": retained / paired_bs if paired_bs else None,
        "forgetting_rate": reg / paired_bs if paired_bs else None,
        "sample_warning": "Exploratory counts; small suites do not establish statistical certainty."
        if n < 30
        else None,
    }


def _target_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target = [row for row in rows if row["suite"] == "target"]
    output: dict[str, Any] = {"cases": len(target)}
    for side in ("base", "adapted"):
        valid = [row for row in target if row[f"{side}_passed"] is not None]
        field_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        missing_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        invented_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for row in valid:
            parsed = row.get(f"{side}_parsed")
            if not isinstance(parsed, dict):
                continue
            for field in FIELDS:
                expected = row["expected"].get(field)
                actual = parsed.get(field)
                field_counts[field][0] += actual == expected
                field_counts[field][1] += 1
                if expected is None:
                    missing_counts[field][0] += actual is None
                    missing_counts[field][1] += 1
                    invented_counts[field][0] += actual is not None
                    invented_counts[field][1] += 1
        output[side] = {
            "exact_record_accuracy": _ratio(
                sum(row[f"{side}_passed"] is True for row in valid), len(valid)
            ),
            "json_validity": _ratio(
                sum(row[f"{side}_format_valid"] is True for row in valid), len(valid)
            ),
            "schema_validity": _ratio(
                sum(row[f"{side}_grading"].get("schema_valid") is True for row in valid), len(valid)
            ),
            "field_accuracy": {
                key: _ratio(correct, total) for key, (correct, total) in field_counts.items()
            },
            "missing_field_correctness": {
                key: _ratio(correct, total) for key, (correct, total) in missing_counts.items()
            },
            "incorrect_non_null": {key: wrong for key, (wrong, _total) in invented_counts.items()},
            "unexpected_fields": sum(
                len(row[f"{side}_grading"].get("unexpected_fields", [])) for row in valid
            ),
            "extra_prose": sum(row[f"{side}_grading"].get("extra_prose") is True for row in valid),
        }
    return output


def aggregate_target(cases: list[dict[str, Any]]) -> dict[str, Any]:
    target = [c for c in cases if c["suite"] == "target"]
    fields = defaultdict(lambda: [0, 0])
    for row in target:
        details = row.get("adapted_grading", {})
        for k, v in details.get("field_accuracy", {}).items():
            fields[k][0] += bool(v)
            fields[k][1] += 1
    return {
        "exact_record_accuracy": _ratio(
            sum(r.get("adapted_passed") is True for r in target), len(target)
        ),
        "field_accuracy": {k: _ratio(ok, n) for k, (ok, n) in fields.items()},
        "cases": len(target),
    }


def _ratio(a: int, b: int) -> float | None:
    return a / b if b else None
