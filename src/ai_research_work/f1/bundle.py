"""Strict, checksummed episode-A bundles; trusted artifacts only."""
import hashlib
import json
import re
from numbers import Real
from pathlib import Path

import numpy as np

CLASSES = [1, 2, 3, 4, 22, 23, 24, 25]
NAMES = ["swipe left", "swipe right", "wave", "clap", "jog", "walk", "sit to stand", "stand to sit"]
PATTERNS = ("both", "sensor_only", "video_only")
SCHEMA = "f1-development-v1"
PREPROCESS = {
    "sensor": {"key": "d_iner", "channels": 6, "steps": 128,
               "resampling": "linear_numpy_interp_unit_interval", "normalization": "checkpoint_training_mean_std"},
    "hof": {"resize": [160, 120], "interpolation": "INTER_AREA", "frames": "all",
            "flow": "Farneback", "parameters": [0.5, 3, 9, 3, 5, 1.1, 0],
            "magnitude_strict_min": 0.5, "bins": [16, 4, 4, 8], "normalization": "clip_L1_plus_1e-6"},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError(f"Nonstandard JSON numeric token: {value}")
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"Duplicate JSON key: {key}")
            out[key] = value
        return out
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid, object_pairs_hook=unique)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def encode_q(q):
    if not isinstance(q, Real) or isinstance(q, bool):
        raise ValueError("Quantile must be a number or explicit positive_infinity")
    q = float(q)
    if np.isposinf(q):
        return "positive_infinity"
    if not np.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("Invalid conformal quantile")
    return q


def decode_q(q):
    if q == "positive_infinity":
        return float("inf")
    encode_q(q)
    return float(q)


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite embedding")
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def feature(sensor=None, video=None):
    blocks = [unit(x) for x in (sensor, video) if x is not None]
    if not blocks or any(x.shape != (64,) for x in blocks):
        raise ValueError("Expected available 64-dimensional embeddings")
    return unit(np.concatenate(blocks))


def trial_identity(trial):
    match = re.fullmatch(r"a(\d+)_s(\d+)_t(\d+)", trial)
    if match is None:
        raise ValueError("Invalid bank trial ID")
    return tuple(map(int, match.groups()))


class Bundle:
    def __init__(self, path):
        self.path = Path(path)
        m = self.meta = read_json(self.path / "bundle.json")
        if m["schema_version"] != SCHEMA or m["class_order"] != CLASSES or m["action_names"] != NAMES:
            raise ValueError("Unsupported bundle schema or class order/names")
        if m["episode"] != "A" or m["seed"] not in (17, 29, 43) or m["held_subject"] not in range(1, 7):
            raise ValueError("Unsupported research identity")
        if m["bundle_id"] != f'A_seed{m["seed"]}_held{m["held_subject"]}':
            raise ValueError("Bundle ID disagrees with research identity")
        if m["preprocessing"] != PREPROCESS:
            raise ValueError("Incompatible preprocessing")
        if m["training_subjects"] != [s for s in range(1, 7) if s != m["held_subject"]]:
            raise ValueError("Training subject leakage")
        for name in ("selection", "missingness", "policy"):
            if not re.fullmatch(r"[0-9a-f]{40}", m["source_commits"][name]):
                raise ValueError("Missing exact source commit")
        self.version = sha(self.path / "bundle.json")
        required = {"sensor.pt", "hof.pt", "bank.npz", "replay.npz"}
        if set(m["files"]) != required:
            raise ValueError("Unexpected bundle file inventory")
        for name, digest in m["files"].items():
            if sha(self.path / name) != digest:
                raise ValueError(f"Checksum mismatch: {name}")
        self.bank = dict(np.load(self.path / "bank.npz", allow_pickle=False))
        ids = self.bank["ids"].tolist()
        labels = self.bank["actions"].tolist()
        if len(ids) < 5 or len(ids) != len(set(ids)) or ids != m["training_ids"] or len(ids) != len(labels):
            raise ValueError("Empty, duplicate or incompatible bank IDs")
        if hashlib.sha256(",".join(ids).encode()).hexdigest() != m["training_ids_sha256"]:
            raise ValueError("Training-ID provenance mismatch")
        for tid, action in zip(ids, labels):
            a, s, _ = trial_identity(tid)
            if a != action or a not in CLASSES or s not in m["training_subjects"]:
                raise ValueError("Bank label mismatch or held/unknown training leakage")
        if m["label_provenance"] != "UTD-MHAD_dataset_labels_verified_against_manifest":
            raise ValueError("Missing verified label provenance")
        if set(m["patterns"]) != set(PATTERNS):
            raise ValueError("Missing availability pattern parameters")
        for pattern, dim in zip(PATTERNS, (128, 64, 64)):
            bank = self.bank[pattern]
            if bank.shape != (len(ids), dim) or not np.isfinite(bank).all():
                raise ValueError("Incompatible embedding bank shape/values")
            norms = np.linalg.norm(bank, axis=1)
            if not np.allclose(norms, 1, atol=1e-6, rtol=0):
                raise ValueError("Bank embeddings must have unit norm")
            p = m["patterns"][pattern]
            expected_encoder = ":".join(m["files"][n] for n in
                ({"both": ["sensor.pt", "hof.pt"], "sensor_only": ["sensor.pt"], "video_only": ["hof.pt"]}[pattern]))
            if p["encoder_id"] != expected_encoder or p["normalization"] != "block_L2_then_joint_L2":
                raise ValueError("Encoder/bank space mismatch")
            for det in ("D1", "D2"):
                d = p[det]
                expected = "mean_cosine_distance_top5" if det == "D1" else "1-max_raw_probability"
                if d["score_definition"] != expected or d["orientation"] != "higher_is_unknown" or d["rule"] != "score>threshold":
                    raise ValueError("Incompatible detector rule")
                if not np.isfinite(d["threshold"]) or not 0 <= d["threshold"] <= (2 if det == "D1" else 1):
                    raise ValueError("Invalid detector threshold")
            c = p["conformal"]
            decode_q(c["q"])
            if c["alpha"] != .1 or c["method"] != "descriptive_pooled_crossfit" or c["n_calibration"] != len(ids):
                raise ValueError("Incompatible C2 calibration")
            if not np.isfinite(p["temperature"]) or p["temperature"] <= 0:
                raise ValueError("Invalid temperature")
        # Cross-check paired coordinates with the two single-branch spaces.
        paired = unit(np.concatenate([self.bank["sensor_only"], self.bank["video_only"]], axis=1))
        if not np.allclose(paired, self.bank["both"], atol=2e-7, rtol=0):
            raise ValueError("Paired bank and modality spaces disagree")
        with np.load(self.path / "replay.npz", allow_pickle=False) as replay:
            if set(replay["ids"].tolist()) & set(ids):
                raise ValueError("Evaluation query found in reference bank")

    def neighbors(self, pattern, embedding, k, query_id=None):
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("explain_k must be a positive integer")
        bank = self.bank[pattern]
        embedding = np.asarray(embedding, dtype=np.float64)
        if embedding.shape != (bank.shape[1],) or not np.isfinite(embedding).all():
            raise ValueError("Query embedding incompatible with bank")
        indices = np.array([i for i, tid in enumerate(self.bank["ids"]) if tid != query_id], dtype=int)
        if k > len(indices):
            raise ValueError("k exceeds eligible bank size")
        distances = 1 - unit(bank[indices]) @ unit(embedding)
        # Stable ID ordering resolves exact distance ties reproducibly.
        order = np.lexsort((self.bank["ids"][indices], distances))[:k]
        p = self.meta["patterns"][pattern]
        return [{"trial_id": str(self.bank["ids"][indices[j]]),
                 "action_id": int(self.bank["actions"][indices[j]]),
                 "action_name": NAMES[CLASSES.index(int(self.bank["actions"][indices[j]]))],
                 "distance": float(distances[j]), "encoder_id": p["encoder_id"],
                 "bank_version": self.meta["files"]["bank.npz"],
                 "label_provenance": self.meta["label_provenance"]} for j in order]
