# Model card

The selected classifier is a sensor CNN plus HOF-GRU with equal probability
averaging and explicit single-modality fallback. It has 66,672 parameters and
eight known actions. Intended use is offline research and code study on completed
RGB/IMU clips. It is not validated for safety-critical or deployed decisions.

[Method](docs/method.md) specifies preprocessing, raw decision probabilities,
alternative D1/D2 rejection and descriptive C2 sets. [Evaluation](docs/evaluation.md)
records the six-subject development protocol and historical aggregate metrics.
[Limitations](docs/limitations.md) includes unknown acceptance, subject exposure,
dependent seeds and missing-modality risks. No trained model is distributed.
Random weights used for shape checks make no accuracy claim. No broad reuse license
has been selected; see [licensing](LICENSING.md) and [data](DATA.md).
