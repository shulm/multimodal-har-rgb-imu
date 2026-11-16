# Method

Sensor -> 128-step resampling -> training-only normalization -> sensor CNN.

RGB -> Farneback optical flow -> 16-by-128 HOF tensor -> HOF-GRU or tiny HOF-Transformer.

Each branch returns ordered class probabilities. Arithmetic averaging is the backward-compatible default; normalized product is an explicit alternative with a 1e-12 probability floor and stable float64 log-domain normalization. Class IDs, labels, batch size and bundle metadata must agree. Single-modality input returns unchanged surviving-branch probabilities; neither input produces insufficient input.

The inference path reuses the measured network and preprocessing definitions without redesign, augmentation or normalization fitting. Local bundles identify which fitted fold/model they contain. The three-seed LOSO aggregate is not one bundle or an instruction to choose the best fold.

Classification and acceptance are distinct. Full-class inference returns predictions and scores without an operational rejection policy. See [models](models.md), [protocols](benchmark_protocols.md), and [limitations](limitations.md). The separate legacy API retains its own descriptive detector/set workflow for compatible old bundles.
