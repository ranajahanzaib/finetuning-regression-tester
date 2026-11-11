# Evaluation methodology

FineTuning Regression Tester evaluates a fixed set of inputs with identical prompts, decoding settings, and deterministic graders before and after adaptation. The evaluation definitions and expected answers must be fixed before inspecting model outputs.

## Paired outcomes

Each shared case is classified as preserved success, regression (pass to fail), improvement (fail to pass), preserved failure, or inconclusive. Execution errors and missing outputs are inconclusive rather than ordinary failures. Baseline and adapted accuracy each use that model's own valid scored responses. Accuracy change is the paired difference on cases valid for both models. Retention is preserved successes divided by paired baseline successes; operational forgetting is regressions divided by paired baseline successes. Both are N/A when there are no paired baseline successes.

Paired examples matter because the same model can exchange successes and failures while its aggregate accuracy stays unchanged. Therefore reports show transition counts and individual examples alongside score differences.

## Interpretation

Benchmark improvement alone says nothing about unrelated behavior. A baseline success is required to call a later failure an observed regression. FineTuning Regression Tester uses “forgetting” only as an operational label for a pass-to-fail transition in a particular evaluation. A changed deterministic output does not prove permanent loss of an underlying skill. Small samples are exploratory; a small model and synthetic task do not establish universal fine-tuning behavior.

JSON validity is evaluated separately from schema validity and field correctness. Extra prose is a strict format failure even if the JSON fragment is readable. The grader does not silently repair malformed output or use a second language model as judge.

The format suite tests JSON, a headerless CSV row, a lowercase boolean token, three numbered lines, an exact JSON key subset, and plain text without Markdown. It records syntax, semantic match, and instruction adherence separately. Missing-data abstention asks about facts absent from supplied descriptions; unsupported-capability cases ask for actions requiring unavailable tools or access. Neither suite establishes comprehensive safety behavior.

## Synthetic data and leakage

Generator splits use separate product identities and split-specific wording families. Exact and whitespace/case-normalized duplicate prompts are rejected. Cross-split sentence similarity at or above 0.92 is reported as a review flag; this conservative string threshold is not a semantic-equivalence test. Training and validation identities are also checked against held-out target cases. These checks reduce simple leakage but do not prove semantic independence; inspect the generated datasets and hashes before relying on a result.

Currency normalization is conservative: an explicit dollar symbol maps to USD in the generated English examples. No exchange rates, locale inference, or conversion is performed. Prices remain decimal strings.

When a listing explicitly names multiple colors, the synthetic target joins them in source order with comma and space (for example, `"black, silver"`). Pack wording such as “bundle contains 2 units” maps to quantity `2`; quantities of unrelated objects are not generated as product counts.
