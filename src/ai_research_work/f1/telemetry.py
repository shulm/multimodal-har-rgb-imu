"""Opt-in local records. No raw input paths and no inferred population metrics."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
import numpy as np
from .bundle import SCHEMA, read_json

FIELDS = {"schema_version", "status", "condition", "availability", "bundle_id", "model_version",
          "action_id", "detector", "score", "threshold", "score_definition", "decision_probabilities",
          "probabilities", "max_probability", "prediction_set", "conformal", "calibrated_confidence",
          "timing_seconds", "diagnostic_argmax_action_id"}


def append(path, record, event_id=None, review=None, error_code=None):
    out = {key: record[key] for key in FIELDS if key in record}
    out.update(schema_version=SCHEMA, event_id=event_id or str(uuid.uuid4()),
               timestamp=datetime.now(timezone.utc).isoformat())
    if error_code is not None:
        if error_code not in ("invalid_input_or_bundle", "io_error"):
            raise ValueError("Unknown telemetry error code")
        out["error_code"] = error_code
    if review is not None:
        if set(review) != {"action_id", "provenance", "reviewed_at"} or not isinstance(review["action_id"], int) or not review["provenance"] or not review["reviewed_at"]:
            raise ValueError("Review needs integer action_id, provenance and reviewed_at")
        out["review"] = review
    line = json.dumps(out, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line+"\n")
    return out["event_id"]


def summarize(path):
    def invalid(x):
        raise ValueError(f"Invalid telemetry JSON: {x}")
    records = [json.loads(line, parse_constant=invalid) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(r.get("schema_version") != SCHEMA for r in records):
        raise ValueError("Unsupported telemetry schema")
    result = {"records": len(records), "scope": "logged subset only; no population performance or drift inference", "patterns": {}}
    for pattern in sorted({r["condition"] for r in records}):
        rows = [r for r in records if r["condition"] == pattern]
        times = [r["timing_seconds"]["total"] for r in rows if "total" in r.get("timing_seconds", {})]
        labeled = [r for r in rows if "review" in r]
        accepted = [r for r in labeled if r["status"] == "accepted"]
        correct = sum(r["action_id"] == r["review"]["action_id"] for r in accepted)
        result["patterns"][pattern] = {"n": len(rows), "statuses": dict(Counter(r["status"] for r in rows)),
            "errors": sum("error_code" in r for r in rows), "latency_n": len(times),
            "latency_seconds": {"p50": float(np.median(times)), "p95": float(np.quantile(times, .95))} if times else None,
            "reviewed_n": len(labeled), "reviewed_accepted_n": len(accepted), "reviewed_accepted_correct": correct,
            "reviewed_accepted_accuracy": correct/len(accepted) if accepted else None,
            "reviewed_total_correct_fraction": correct/len(labeled) if labeled else None}
    return result
