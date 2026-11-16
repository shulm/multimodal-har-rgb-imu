# Verification

This edition was installed in a clean Python 3.12 environment with CPU PyTorch 2.6.0. Dependency checking (`python -m pip check`) passed. The test suite completed with **70 passed and one skipped**; the skip requires an optional trusted legacy bundle. All three notebooks executed successfully and their HTML previews were regenerated with synthetic examples and approved aggregates only.

Separate trained-weight parity checks covered six fixtures and 24 model/query cases. The port and reference produced exactly equal outputs when both ran on the same CPU backend. Comparisons with saved historical probabilities passed absolute tolerance `2e-5` and relative tolerance `1e-4`. Small internal differences against the historical CUDA computation were observed; these checks do not promise bitwise equality across hardware or numerical backends. Private trained fixtures and weights are not distributed.

Run `python -m pytest -q`, the four HAR demo model/fusion combinations in HOW_TO_RUN.md, and `python scripts/execute_notebooks.py` from a clean installed environment. Public checks use synthetic fixtures or approved aggregate results and need no private credentials. They do not reproduce model training or the historical benchmark.

Tests cover both encoder shapes and 27-class parameter counts, ordered labels, fusion normalization, missingness and malformed input, versioned bundles, CLI behavior and legacy compatibility. Random-model fixture tests check serialization, not trained recognition. Repeating trained-weight parity requires separately authorized trusted local artifacts.

HTML previews and saved notebook outputs contain no real recordings, checkpoint contents, per-record predictions or local artifact paths. Keep these exclusions when rerunning the notebooks with optional local artifacts.
