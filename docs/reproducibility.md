# Reproducibility

Dataset generation records its seed and SHA-256 hashes. Evaluation persists prompts, expectations, raw outputs, grader outcomes, and case identity. Baseline and adapted runs must use matching model snapshots, decoding settings, dataset provenance, environment, and case hashes. A comparison is rejected when these recorded conditions differ. For repeatable runs, pin `model.revision` and the MLX-LM version. The default config records the recommended checkpoint as 4-bit MLX quantization; custom checkpoints should set `model.quantization` to their actual format or `null` when unknown.

The generator and decoding setup are deterministic by configuration, but bit-identical model output across MLX versions and hardware is not guaranteed. Freeze expected outputs before evaluation; changing them after inspecting model results invalidates the comparison. The manifest records incomplete stages and errors so an absent stage is never mistaken for a measured score.
