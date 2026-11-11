# Architecture

FineTuning Regression Tester separates five responsibilities: deterministic dataset generation and validation (`core`), inference (`mlx`), evaluation/grading (`core`), paired analysis (`core`), and presentation (`report`, `dashboard`). The evaluation/analysis modules have no MLX dependency. The CLI composes these layers and artifacts are portable JSON, JSONL, CSV, Markdown, and self-contained HTML.

An experiment writes a manifest, frozen evaluation cases, baseline and adapted raw outputs, a comparison, and offline report exports. Atomic replacement is used for JSON and JSONL writes. Re-running a case file resumes from completed case IDs. A run directory is never overwritten.

Model downloads are an explicit CLI operation. Baseline and training resolve only local model directories or cached Hugging Face snapshots, so those stages do not silently access the network. Cached commit revisions are recorded when available.
