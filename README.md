# RGB-IMU activity recognition: GRU and Transformer

David Shulman · [shulm](https://github.com/shulm)

A completed-clip classifier combining a six-channel sensor CNN with RGB-derived optical-flow histograms. Choose a HOF-GRU or a tiny HOF-Transformer, then arithmetic or normalized-product probability fusion. Either branch can operate alone. The full-class API returns classification scores, not a claim that an activity is known, accepted or safe.

## Full-dataset benchmark

UTD-MHAD: **861 usable paired recordings, all 27 actions and eight people**. Eight-fold leave-one-person-out evaluation trains on seven people and tests on the remaining person at seeds **17, 29 and 43**. Results average seeds within person, then weight all eight people equally.

| System | Accuracy % | Macro-F1 % | NLL | Brier |
|---|---:|---:|---:|---:|
| GRU + arithmetic | 85.92 | 83.75 | 0.541258 | 0.239410 |
| GRU + product | 89.17 | 87.52 | 0.694219 | 0.180640 |
| Transformer + arithmetic | 87.85 | 86.19 | 0.488315 | 0.212298 |
| Transformer + product | 91.68 | 90.64 | 0.597152 | 0.148871 |

Source of the table and notebook plot: [machine-readable results](results/full_dataset_results.json). Lower NLL and Brier are better. Product improves these closed-set accuracy results but worsens NLL relative to arithmetic for each encoder. Sensor + GRU has **69,142** trainable classifier parameters; sensor + Transformer has **70,646**. The sensor branch is shared between the two comparisons.

“Full dataset” describes coverage under the declared folds. It does not mean training and testing on the same recordings, distributing the dataset, or supplying one final model trained on everyone. The benchmark is not a deployable all-subject checkpoint. The earlier [eight-action development result](docs/legacy_eight_action.md) uses a different task and is not directly comparable.

## Run without data or weights

Use Python 3.12 in a new virtual environment, then run from this repository:

```sh
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-lock.txt
python -m pip install . --no-deps
python -m ai_research_work.har demo --model hof_gru --fusion arithmetic
python -m ai_research_work.har demo --model hof_transformer --fusion product
python -m pytest -q
```

Demos use synthetic arrays and explicitly untrained models. They exercise inference shapes, fusion and fallback; they do not estimate accuracy or reproduce the benchmark. Arithmetic is the backward-compatible default. Inference performs no downloads or training.

See [installation and optional local inference](HOW_TO_RUN.md), [model definitions](docs/models.md), [benchmark protocols](docs/benchmark_protocols.md), and the [comparison notebook](notebooks/03_compare_models.ipynb) with its [HTML preview](notebooks/03_compare_models.html). The [first teaching notebook](notebooks/01_understand_f1.ipynb) and [optional local-artifact notebook](notebooks/02_review_local_artifacts.ipynb) remain available.

## Included and excluded

Included: authored inference code, tests, synthetic teaching examples and historical aggregate results. Excluded: source RGB/IMU recordings, participant images, checkpoints, reference banks and per-record predictions. Optional real inference requires a trusted compatible local bundle and permitted local inputs. No private account is needed for the synthetic mode.

Missing one modality returns its surviving branch probabilities unchanged. Neither input yields `insufficient_input`; malformed supplied input is an error. Classification and acceptance are distinct: no full-class acceptance policy is bundled, and legacy thresholds must not be reused. One small benchmark, sensor-placement/action confounding and the lack of operational unknown-action guarantees limit deployment conclusions. No state-of-the-art, external-validation, real-time-streaming or measured product-pipeline-latency claim is made.

Read the [model card](MODEL_CARD.md), [data notes](DATA.md), [limitations](docs/limitations.md), [reproducibility](docs/reproducibility.md), and [licensing terms](LICENSING.md).
