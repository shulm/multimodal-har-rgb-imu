# Data and local inputs

The benchmark is [UTD-MHAD](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html), introduced by Chen, Jafari and Kehtarnavaz (ICIP 2015). It provides RGB, depth, skeleton and inertial streams; this project uses **RGB and the released six-channel inertial stream only**. Obtain source data from the original provider under applicable terms. This repository does not redistribute recordings or participant images.

The full benchmark has 861 usable paired recordings, all 27 actions and eight people. Three expected recordings are absent/corrupted in the released set. Each LOSO fold trains on seven people and tests on one; seeds 17, 29 and 43 are averaged within person before equal-person averaging. This is not one model trained and evaluated on all eight people.

The IMU is worn on the right wrist for actions 1-21 and the right thigh for actions 22-27. Sensor placement therefore varies with action identity. Preserve the released channel order; do not invent new physical-unit conversions or add depth, skeleton or identity-derived features.

The no-data demos generate synthetic tensors and need no download. Optional full-class inference reads a permitted completed RGB clip and numeric NPY sensor array of shape N-by-6, using the compatible local bundle's training-only scaler. The legacy interface separately reads MATLAB `d_iner`. Never fit normalization on inference examples. The software validates structure but cannot establish that a real recording follows the benchmark's physical capture assumptions.

See [LICENSING.md](LICENSING.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Dataset availability is not a blanket permission to redistribute it or to deploy a classifier on people.
