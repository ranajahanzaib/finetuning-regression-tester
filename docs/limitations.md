# Limitations

- MLX execution requires macOS on Apple Silicon with Metal support. The portable test suite does not run model training.
- The default synthetic dataset is deliberately simple and is not representative of production product catalogs.
- Current deterministic graders cover objective text/JSON contracts; unsupported-request graders recognize a documented set of abstention phrases and can miss valid variants.
- The dashboard summarizes stored comparisons. It does not infer absent training metrics or claim safety alignment.
- Current confidence reporting is counts and denominator-aware metrics; no confidence interval is claimed.
- Evaluation and training require an explicitly downloaded local model snapshot; its resolved revision is recorded when the Hugging Face cache exposes it. Locally supplied directories may not expose a commit revision.
