# Historical eight-action development system

The earlier sensor CNN + HOF-GRU development study used eight selected actions, six people and 191 unique known clips. Its three-seed LOSO aggregate reported 98.96% accuracy, 98.86% macro-F1 and 66,672 classifier parameters. [Historical classification CSV](../results/legacy_eight_action_classification.csv) and [historical reliability CSV](../results/reliability_summary.csv) preserve that scope.

This is a smaller task with a different protocol. Comparing 98.96% to the full 27-action 91.68% result as deterioration would be misleading. Three repeated seeds did not create 573 independent recordings.

`ai_research_work.f1` remains compatible with trusted `f1-development-v1` bundles. Its detector/set/retrieval features require those bundles' own class ordering, calibration evidence and metadata. They are not transferred to the new full-class models or a different fusion. Unknown actions can be accepted as known ones, including the historical throw/wave failure. Descriptive pooled calibration is not a new-person safety guarantee.
