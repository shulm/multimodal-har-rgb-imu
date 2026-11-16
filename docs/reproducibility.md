# Reproducibility and evidence boundaries

Use a clean Python 3.12 environment and the installation steps in HOW_TO_RUN.md. Public tests and all default notebooks run without credentials, source recordings or real weights. Synthetic model checks verify shape, parameter count, fusion, fallback and bundle validation; they do not validate trained accuracy.

The committed JSON is the source for the README table and comparison plot. It records the completed protocol, metric units and classifier sizes. No unpublished archive paths, per-record predictions or private repository access are required to interpret it. Exact neural retraining is outside this inference-and-education release: no new training infrastructure or datasets are shipped.

The optional local-artifact notebook defaults to a skip. Do not publish outputs containing real bundle paths, identities or predictions. The notebook execution script clears outputs and metadata before executing safe defaults and generating HTML previews. The full-class runtime never fits a sensor scaler on user inputs and never substitutes legacy rejection thresholds.

A trained-model parity result must use an authorized real checkpoint and the original implementation. A missing checkpoint must be reported as unavailable or skipped; a random-model smoke test is not a replacement. See test and release verification notes for the actual checks performed for this edition.
