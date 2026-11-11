# Development evidence

The selected study used eight known actions, six subjects, six leave-one-subject-out
folds and three seeds. It contains 191 unique known-action clips and 573 repeated
seed-trial predictions. Headline accuracy and macro-F1 average subjects within each
seed, then seeds. Repeated seeds are dependent; overlapping training folds also
limit uncertainty interpretations. No raw records are distributed here.

| Selected model | Accuracy | Macro-F1 | Classifier parameters |
|---|---:|---:|---:|
| Equal sensor CNN + HOF-GRU probability average | 98.96% | 98.86% | 66,672 |

These historical summaries are transcribed aggregate evidence, not a fresh benchmark
run by this export. Earlier evaluation subjects were inspected during development;
there is no historically untouched final-test claim. Learned missingness heads and
imbalance corrections were experimental comparisons, not replacements for this model.

For both inputs, accepted-known accuracy was about 99.62% for D1 and D2, while
unknown false acceptance was about 58.33% (D1) and 60.42% (D2). These policy rates
average subject/seed evaluations. Unknown false acceptance means an unknown was
accepted as a known action, not merely that the detector failed to reject it.
C2 pooled-crossfit construction achieved mean known coverage of about 90.2%,
a descriptive quantity. An observed throw→wave acceptance demonstrates why selective known
accuracy cannot be interpreted as precision in a deployment mixture.

Selected-study CPU completed-clip latency was approximately 381 ms median and
517 ms p95, with preprocessing included and model loading outside the timer.
Reference setup: Windows 10, Python 3.12.4, Torch 2.6 CPU, NumPy 2.1.3,
OpenCV 4.13; one Torch/OpenCV thread; Intel family 6 model 94, four physical/eight
logical cores and 15.8 GiB RAM. These are not streaming or end-to-end service
latencies, and will vary with clip length and hardware.

Machine-readable aggregates: [classification](../results/classification_summary.csv)
and [reliability](../results/reliability_summary.csv). See [limitations](limitations.md).
