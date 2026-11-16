# Benchmark protocols must stay separate

| Study | Classes and people | Roles and seeds | Interpretation |
|---|---|---|---|
| Principal full-dataset classification | 27 actions, eight people, 861 paired clips | Eight LOSO folds, seven train/one test; seeds 17,29,43 | Closed-set held-person classification |
| Reliability research | 20 known / seven unknown actions, eight people | Four train / three calibration / one test, fixed overlapping class episodes; seed 17 | Conditional acceptance-policy transfer |
| Historical development | Eight actions, six people, 191 known clips | Development LOSO, three repeated seeds | Smaller selected task; not comparable to 27-action accuracy |

Principal metrics average seeds within person, then give equal weight to people. Full dataset means all usable recordings are covered under these folds, not that training and testing share people or that a final all-person checkpoint exists. The historical classifier comparison used a CUDA LOSO path; the separate open-set study used the historical CPU recipe. No bitwise cross-backend identity is assumed.

The reliability study's identically recalibrated arithmetic confidence-versus-singleton comparisons gave reductions of 8.93 and 10.61 percentage points in unknown acceptance at deployment-specific original useful-output counts. Some deployments lost correct-known utility. Fixed-target sensitivity showed smaller, operating-point-dependent differences and shrinking composite feasibility; this is not a uniform benefit or safe unknown detection. It is not acceptance performance of the 91.68% classifier. Additional open-set seeds 29/43 have not been executed as part of the completed evidence presented here.
