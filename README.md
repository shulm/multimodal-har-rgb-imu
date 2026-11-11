# Lightweight RGB + IMU activity recognition

David Shulman · [shulm](https://github.com/shulm)

A small completed-clip classifier combining a six-channel inertial CNN with an
optical-flow-histogram GRU. Equal probability averaging gives a simple multimodal
decision; either branch can operate alone. Explicit uncertainty and rejection
states keep a missing input or an ambiguous prediction visible to the caller.

The selected development study reported **98.96% accuracy**, **98.86% macro-F1**
and **66,672 classifier parameters** over eight actions and six subjects using
leave-one-subject-out evaluation. These are historical aggregate results, not
measurements produced by the synthetic quick-start. There were 191 unique known
clips and three repeated seeds, not 573 independent recordings or people.

The central limitation is unknown activity: **throw can be accepted as wave**.
High accuracy among accepted known examples is not deployment precision.
This is a research portfolio, with no deployed-service or state-of-the-art claim.

## Run without data

Use Python 3.12 and a new virtual environment. Install the CPU wheel and package:

```sh
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-lock.txt
python -m pip install . --no-deps
python -m pytest -q
python scripts/smoke.py
python -m ai_research_work.f1 predict --bundle absent --detector D2
```

The last command returns `insufficient_input` without loading weights. The smoke
uses synthetic arrays and randomly initialized models: it verifies shapes and
plumbing, not recognition quality. No training or runtime download is included.

Start with [installation and local inference](HOW_TO_RUN.md), the
[teaching notebook](notebooks/01_understand_f1.ipynb), or its
[saved HTML preview](notebooks/01_understand_f1.html).
Read [method](docs/method.md), [evaluation](docs/evaluation.md),
[limitations](docs/limitations.md), and [model card](MODEL_CARD.md).

Real weights, reference banks, replay arrays and raw data are not distributed.
[Optional local verification](notebooks/02_review_local_artifacts.ipynb) requires
an explicitly supplied, trusted compatible bundle. No private account is required
for the synthetic mode. See [data](DATA.md) and [licensing](LICENSING.md).
