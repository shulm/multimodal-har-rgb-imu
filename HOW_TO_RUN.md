# Running the portfolio

Use Python 3.12. Create `python -m venv .venv`, then activate it
(`.venv\Scripts\Activate.ps1` on PowerShell; `source .venv/bin/activate` on POSIX).
Run the installation and smoke commands in README.md from the repository root.
Then run `python -m pip check`. Dependency installation needs a package index or
an independently supplied wheelhouse; inference and demonstrations make no downloads.
The direct lock pins portable dependencies; requirements-environment.txt records
the full tested Windows environment and is not a cross-platform lock.

Execute the notebooks with `python scripts/execute_notebooks.py`. It saves outputs
and HTML next to each notebook. The educational notebook uses synthetic data only.
The local-artifact notebook defaults to an explicit skip and contains no training.

## Optional trusted local inputs

Obtain compatible artifacts only if you have independent permission to use them.
There is no model download in this repository. A compatible `f1-development-v1`
bundle consists of bundle.json, sensor.pt, hof.pt, bank.npz and replay.npz, with
valid hashes, class order, preprocessing and fold metadata. Existing integrity
checks are retained. Do not edit manifests to force compatibility. Trust the
publisher before loading checkpoints; checksums do not authenticate a publisher.

```sh
python scripts/verify_artifacts.py --bundle <local-bundle-directory>
python -m ai_research_work.f1 predict --bundle <local-bundle-directory> --detector D1 --sensor <local-recording.mat> --video <local-clip.avi>
```

MAT input must contain finite `d_iner` of shape N×6, N≥2. Video is decoded as a
completed clip. Omit one modality flag for surviving-branch fallback. Supplied
malformed inputs fail; they are not treated as missing. Select D1 or D2 explicitly.
Use `--query-id` only when a known compatible identity should be excluded from
retrieval. `--explain-k 0` hides neighbors without disabling D1's top-five score.
Opt-in `--telemetry <local-log.jsonl>` writes local summaries only; protect these
files and do not commit them. They can contain model identities and review labels.

For pytest's optional bundle validation, run
`python -m pytest tests -q -m local_artifacts --bundle=<local-bundle-directory>`.
Without it, the artifact-dependent test is skipped. For the optional notebook,
edit its empty BUNDLE setting and set TRUST_BUNDLE=True locally. Its outputs may
contain sensitive artifact information: do not publish an executed real-data copy.
