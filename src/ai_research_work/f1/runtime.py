"""Raw clip -> frozen F1 -> D1 OR D2 -> raw C2 -> diagnostic retrieval."""
import time
import numpy as np
from .bundle import Bundle, CLASSES, SCHEMA, decode_q, feature


def vector(p):
    p = np.asarray(p, dtype=np.float64)
    if p.shape != (8,) or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1, atol=1e-6, rtol=0):
        raise ValueError("Expected eight finite nonnegative normalized probabilities")
    return p


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def missing():
    return {"schema_version": SCHEMA, "status": "insufficient_input", "condition": "both_missing",
            "availability": [False, False], "action_id": None, "bundle_id": None, "model_version": None,
            "probabilities": None, "prediction_set": None, "neighbors": [], "timing_seconds": {"total": 0.0}}


def decide(bundle, detector, sensor_probs=None, hof_probs=None, sensor_emb=None, hof_emb=None, explain_k=5, query_id=None):
    if detector not in ("D1", "D2"):
        raise ValueError("detector must be D1 or D2")
    if sensor_probs is None and hof_probs is None:
        return missing()
    s = vector(sensor_probs) if sensor_probs is not None else None
    v = vector(hof_probs) if hof_probs is not None else None
    condition = "both" if s is not None and v is not None else "sensor_only" if s is not None else "video_only"
    p = .5*s + .5*v if condition == "both" else (s if s is not None else v)
    pp = bundle.meta["patterns"][condition]
    emb = feature(sensor_emb if s is not None else None, hof_emb if v is not None else None)
    if not isinstance(explain_k, int) or isinstance(explain_k, bool) or explain_k < 0:
        raise ValueError("explain_k must be a nonnegative integer")
    neighbors = bundle.neighbors(condition, emb, explain_k, query_id) if explain_k else []
    score = float(np.mean([n["distance"] for n in bundle.neighbors(condition, emb, 5, query_id)])) if detector == "D1" else float(1-p.max())
    c = pp["conformal"]
    pset = [CLASSES[i] for i in np.flatnonzero(1-p <= decode_q(c["q"]))]
    threshold = pp[detector]["threshold"]
    status = "unknown" if score > threshold else "accepted" if len(pset) == 1 else "uncertain"
    # Report fitted T for every pattern, but explicitly identify fallback scaling as unrecommended.
    pT = softmax(np.log(np.clip(p, 1e-6, 1))/pp["temperature"])
    return {"schema_version": SCHEMA, "status": status, "condition": condition,
            "availability": [s is not None, v is not None], "bundle_id": bundle.meta["bundle_id"],
            "model_version": bundle.version, "detector": detector, "score": score, "threshold": threshold,
            "score_definition": pp[detector]["score_definition"], "decision_probabilities": "raw",
            "probabilities": p.tolist(), "max_probability": float(p.max()),
            "diagnostic_argmax_action_id": CLASSES[int(p.argmax())], "prediction_set": pset,
            "prediction_set_role": "diagnostic_after_rejection" if status == "unknown" else "decision",
            "conformal": c, "action_id": pset[0] if status == "accepted" else None,
            "calibrated_confidence": {"probabilities": pT.tolist(), "max_probability": float(pT.max()),
                "temperature": pp["temperature"], "role": "reporting_only",
                "recommended_for_pattern": condition == "both",
                "provenance": pp["temperature_provenance"]},
            "decision_margin": float(threshold-score),
            "conformal_min_abs_margin": None if c["q"] == "positive_infinity" else float(np.min(np.abs(1-p-decode_q(c["q"])))),
            "neighbors": neighbors, "timing_seconds": {}}


class Predictor:
    def __init__(self, bundle):
        self.bundle_path = bundle
        self.bundle = None
        self.models = {}

    def load(self):
        if self.bundle is not None:
            return
        import torch
        from ..models import SensorModel, VideoModel
        b = Bundle(self.bundle_path)
        cp = torch.load(b.path / "sensor.pt", map_location="cpu", weights_only=True)
        if cp["class_order"] != CLASSES or cp["identity"]["seed"] != b.meta["seed"] or cp["identity"]["held"] != b.meta["held_subject"] or cp["identity"]["train_ids"] != b.meta["training_ids"]:
            raise ValueError("Sensor checkpoint identity mismatch")
        mean, std = cp["mean"].numpy().reshape(6), cp["std"].numpy().reshape(6)
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Invalid training normalizer")
        s, v = SensorModel(8), VideoModel(8, 128)
        s.load_state_dict(cp["state"])
        v.load_state_dict(torch.load(b.path / "hof.pt", map_location="cpu", weights_only=True))
        self.models = {"S": s.eval(), "V": v.eval()}
        self.mean, self.std = mean, std
        self.bundle = b

    def outputs(self, sensor_array=None, hof_array=None):
        import torch
        self.load()
        kw = {}
        with torch.inference_mode():
            for name, x, prefix in (("S", sensor_array, "sensor"), ("V", hof_array, "hof")):
                if x is None:
                    continue
                x = np.asarray(x, dtype=np.float32)
                if x.shape != ((128, 6) if name == "S" else (16, 128)) or not np.isfinite(x).all():
                    raise ValueError("Invalid preprocessed input")
                if name == "S":
                    x = (x-self.mean)/self.std
                emb = self.models[name].embed(torch.from_numpy(x[None]))
                logits = self.models[name].head(emb)[0].numpy()
                kw[prefix+"_emb"] = emb[0].numpy()
                kw[prefix+"_probs"] = softmax(logits)
        return kw

    def predict(self, sensor=None, video=None, detector="D1", explain_k=5, query_id=None):
        start = time.perf_counter()
        if detector not in ("D1", "D2"):
            raise ValueError("detector must be D1 or D2")
        if sensor is None and video is None:
            return missing()
        from . import preprocessing as prep
        timing = {}
        t = time.perf_counter()
        sx = prep.sensor(sensor) if sensor is not None else None
        timing["sensor_preprocessing"] = time.perf_counter()-t
        t = time.perf_counter()
        frames = prep.decode(video) if video is not None else None
        timing["video_decode"] = time.perf_counter()-t
        t = time.perf_counter()
        hx = prep.hof(frames) if frames is not None else None
        timing["hof_extraction"] = time.perf_counter()-t
        t = time.perf_counter()
        self.load()
        timing["bundle_load"] = time.perf_counter()-t
        t = time.perf_counter()
        kw = self.outputs(sx, hx)
        timing["model_inference"] = time.perf_counter()-t
        t = time.perf_counter()
        out = decide(self.bundle, detector, **kw, explain_k=explain_k, query_id=query_id)
        timing["policy_retrieval"] = time.perf_counter()-t
        timing["total"] = time.perf_counter()-start
        out["timing_seconds"] = timing
        return out


def predict(bundle, sensor=None, video=None, detector="D1", explain_k=5, query_id=None):
    return Predictor(bundle).predict(sensor, video, detector, explain_k, query_id)
