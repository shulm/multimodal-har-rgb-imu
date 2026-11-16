# Limitations

One small benchmark, eight people and structured indoor clips limit generalization. Wrist/thigh mounting changes with action group. There is no external-dataset validation, state-of-the-art claim or real-time streaming validation. Completed-clip processing does not establish end-to-end streaming latency; product-pipeline latency is not measured.

Closed-set accuracy says nothing by itself about whether a novel activity should be accepted. Full-class inference exposes scores and does not ship an operational acceptance policy. Historical reliability research is a separate seed-17, 20-known/seven-unknown protocol with people reused across deployments; training, calibration and test roles remain disjoint within each deployment. Its conclusions are operating-point-dependent and include individual losses of correct-known utility. It does not establish universal superiority, noninferiority or safe unknown-action handling.

Product's best recorded accuracy coexists with worse NLL than arithmetic. The benchmark aggregates LOSO folds; no single all-person production model is supplied. Models are untrained in the synthetic demo. Trusted local checkpoints, permissions and acquisition compatibility remain the user's responsibility; hashes verify bytes, not the publisher's authority or model validity.
