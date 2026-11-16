# Evaluation

The principal table is [full_dataset_results.json](../results/full_dataset_results.json), with a CSV view in [classification_summary.csv](../results/classification_summary.csv). Accuracy and macro-F1 are fractions in machine files and percentages in README; NLL and multiclass Brier retain native units. Lower NLL/Brier is better. The comparison notebook reads the JSON directly, keeping the historical measured plot separate from synthetic model outputs.

Eight held-person folds cover all 861 usable paired recordings and all 27 actions. Three seeds are averaged within person, then people equally. There are 2,583 seed-recording decisions per fused system, not 2,583 independent recordings or people. The two encoder comparisons share a sensor branch. Product improves accuracy and Brier in these aggregates while worsening NLL relative to arithmetic.

No synthetic demonstration estimates these metrics. Independent full reproduction requires permitted recordings, preprocessing/training artifacts and the original experiment specification; this inference-focused repository distributes neither data nor weights and does not add a training runner. Optional local artifact checks establish compatibility only for the supplied artifact. Random-weight save/load tests are not trained-model parity tests.

The selected eight-action results remain [historical](legacy_eight_action.md). Reliability outputs are separate from closed-set classification; see [protocols](benchmark_protocols.md).
