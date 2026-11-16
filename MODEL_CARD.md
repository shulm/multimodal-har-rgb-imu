# Model card

## Intended use

Educational and research inference on completed RGB-IMU action clips. Two classifiers are supported: sensor CNN + HOF-GRU and sensor CNN + tiny HOF-Transformer. Both use the same sensor branch and allow arithmetic or normalized-product probability fusion. This is not an autonomous safety system, streaming service or validated unknown-action detector.

## Model and data

The sensor input uses the released six-channel inertial stream, resampled to 128 steps with training-only normalization. RGB becomes Farneback optical flow and a [batch,16,128] histogram-of-flow tensor. Twenty-seven-class parameter counts are 29,883 sensor, 39,259 GRU video and 40,763 Transformer video; totals 69,142 and 70,646. See [architecture](docs/models.md).

The principal benchmark covers 861 paired UTD-MHAD recordings, 27 actions and eight people, with seven-train/one-test LOSO at three seeds. [Results and protocol](results/full_dataset_results.json) are historical aggregates, not synthetic demonstrations or a distributed final checkpoint. The data use wrist placement for actions 1-21 and thigh placement for actions 22-27, confounding mounting and action group.

## Behavior and limitations

Both modalities use selected fusion; one modality returns its branch probabilities unchanged; neither returns insufficient input. Invalid supplied data raises an error. Full-class outputs expose scores without calling them accepted or safe. Closed-set accuracy does not measure unknown rejection or prevalence-dependent deployment precision. No operational acceptance policy is supplied for these models.

LOSO estimates transfer to held people within this small benchmark, not arbitrary environments, cameras, sensor mounts or activities. Three training seeds do not increase the number of people. CPU inference and historical CUDA LOSO paths are not claimed bitwise identical. Product's higher recorded accuracy coexists with worse NLL; no single score determines reliability. Latency for a product-fusion pipeline has not been measured here.

No recordings, participant images, trained weights or per-record predictions are distributed. Users need permitted local data and a trusted compatible bundle. Licensing and third-party notices remain separate and unchanged. The legacy eight-action API has its own protocol and calibration limitations; do not reuse its policy thresholds.
