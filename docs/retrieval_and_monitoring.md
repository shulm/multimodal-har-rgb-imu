# Retrieval and monitoring

Normalize each available 64-dimensional embedding, concatenate available blocks,
then normalize again. Search only the matching saved fold and modality space using
exact cosine distance. Ties use stable trial-ID order; explicit query IDs exclude
self-matches. Reference labels come from training data. Do not mix coordinates from
separately fitted encoders. Retrieval explains similarity, not causality, and never
adds classes or changes the classifier decision.

Local telemetry allowlists fields and excludes raw media and input paths. Summary
denominators distinguish requests, reviewed events, and reviewed accepted events.
Reviewed subsets are not population estimators. Monitoring is a proposal: a future
service would need governed sampling, human review, privacy controls and explicit
drift criteria. This repository does not implement that service or retraining.
