# Portfolio verification

Executed for this edition in a new Windows Python 3.12.4 CPU environment:

- Package wheel built and installed from this repository alone; pip check passed.
- Public tests: 22 passed, one optional artifact test skipped.
- No-network synthetic smoke: passed. Both-missing returns insufficient_input
  without loading weights. An explicitly missing bundle fails with exit code 2;
  omitted optional artifacts produce a clear SKIP message.
- Educational notebook: five code cells executed with saved outputs. Optional
  notebook: two setup/guard cells executed; real artifact loading skipped by default.
  HTML previews contain no scripts, remote assets, widgets or participant images.
- Separately, an explicitly supplied trusted local bundle validation test passed.
- Local numerical comparison newly executed across 18 compatible development
  bundles: 4,302 decisions exactly matched the retained original runtime outputs,
  including probabilities, sets and neighbors. Against saved development evidence,
  zero decision mismatches; maximum score error 2.345e-7 (tolerance 2e-6).
- One authorized known-action raw clip: six pattern/detector checks matched
  statuses, actions, sets and neighbor order. Probability maximum absolute errors
  were below 4e-9; embedding maximum absolute error was 1.145e-5, within the
  unchanged atol=2e-5 plus rtol=1e-4 comparison. This is a bounded consistency
  check, not a new accuracy experiment or full raw-data benchmark.
- Numerical function/class syntax trees match the retained implementation.
  Differences are removal of unused training/legacy code and one rewritten
  provenance docstring. No preprocessing, parameters, thresholds or policy changed.

Artifact-dependent evidence was checked locally and is not distributed. Public CI
runs synthetic checks only. CI actions are pinned to immutable official checkout
v4 and setup-python v5 revisions, verified through the publishers' Git refs.
No training occurred. The tested Windows dependency inventory is
../requirements-environment.txt; portable direct pins are ../requirements-lock.txt.
