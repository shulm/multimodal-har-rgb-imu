# Running the portfolio

Use Python 3.12. Create `python -m venv .venv`, then activate it (`.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on POSIX). Run the installation commands in README.md, followed by `python -m pip check`. Installing dependencies requires a package index or local wheelhouse; inference itself performs no network access or training. The direct lock is the installation specification; the environment inventory is platform-specific.

## Synthetic mode

```sh
python -m ai_research_work.har demo --model hof_gru --fusion arithmetic
python -m ai_research_work.har demo --model hof_gru --fusion product
python -m ai_research_work.har demo --model hof_transformer --fusion arithmetic
python -m ai_research_work.har demo --model hof_transformer --fusion product
python scripts/execute_notebooks.py
```

All four demos use untrained models and synthetic inputs. Notebook outputs are synthetic or committed aggregate results. The runner executes all three notebooks and regenerates their HTML previews; no private data is needed.

## Optional trusted local full-class bundle

The repository does not distribute weights. Obtain a compatible versioned HAR bundle only with permission and from a trusted publisher. File hashes establish integrity, not publisher trust. Architecture, ordered action IDs/labels, class count, fusion configuration, preprocessing, training-only sensor scaler, seed and fold metadata must agree. Do not rewrite a manifest to disguise an eight-class checkpoint as a 27-class model.

```sh
python -m ai_research_work.har predict --bundle <trusted-local-bundle> --trust-local-bundle --model hof_transformer --fusion product --sensor <local-sensor.npy> --video <local-clip.avi>
```

Sensor NPY input contains a finite numeric N-by-6 inertial array. The local bundle supplies the training-only scaler; inference must never fit one to the user's examples. Video is a completed RGB clip. Omit a modality flag for surviving-branch fallback. Neither input returns `insufficient_input` without unnecessary weight loading. Supplied malformed input is an explicit error. The full-class result is classification-only, without an operational rejection guarantee.

## Legacy eight-action compatibility

`ai_research_work.f1` remains for compatible `f1-development-v1` bundles. This is a separate API with historical descriptive reliability logic, not a full-class acceptance policy.

```sh
python -m ai_research_work.f1 predict --bundle absent --detector D2
python scripts/verify_artifacts.py --bundle <trusted-legacy-bundle>
python -m ai_research_work.f1 predict --bundle <trusted-legacy-bundle> --detector D1 --sensor <local-recording.mat> --video <local-clip.avi>
```

The legacy MAT key is `d_iner`, shape N-by-6, N at least two. Its bundle includes its own sensor/video files, reference bank, replay evidence, hashes and role metadata. Do not transplant its labels, thresholds or scaler into the full-class path. The optional local-artifact notebook defaults to skipping real artifacts. Do not publish outputs after enabling it with real local files.
