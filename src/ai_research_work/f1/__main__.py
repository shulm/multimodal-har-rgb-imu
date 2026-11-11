import argparse
import json
import sys
import time
from .bundle import Bundle, SCHEMA, read_json
from .runtime import predict
from .telemetry import append, summarize


def main():
    parser = argparse.ArgumentParser(description="Frozen F1 development inference; no training or downloads")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--bundle", required=True)
    p.add_argument("--sensor")
    p.add_argument("--video")
    p.add_argument("--detector", required=True, choices=["D1", "D2"])
    p.add_argument("--explain-k", type=int, default=5)
    p.add_argument("--query-id")
    p.add_argument("--telemetry")
    p.add_argument("--review-json", help="Optional reviewed action_id, provenance, reviewed_at object")
    p = sub.add_parser("verify-bundle")
    p.add_argument("--bundle", required=True)
    p = sub.add_parser("summary")
    p.add_argument("--telemetry", required=True)
    a = parser.parse_args()
    start = time.perf_counter()
    try:
        if a.command == "predict":
            # Frozen replay environment uses one CPU thread. No GPU allocation.
            if a.sensor is not None or a.video is not None:
                import torch
                import cv2
                torch.set_num_threads(1)
                cv2.setNumThreads(1)
            out = predict(a.bundle, a.sensor, a.video, a.detector, a.explain_k, a.query_id)
            if a.telemetry:
                append(a.telemetry, out, review=read_json(a.review_json) if a.review_json else None)
        elif a.command == "verify-bundle":
            b = Bundle(a.bundle)
            out = {"valid": True, "bundle_id": b.meta["bundle_id"], "model_version": b.version,
                   "bank_count": len(b.bank["ids"])}
        else:
            out = summarize(a.telemetry)
        print(json.dumps(out, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        if a.command == "predict" and a.telemetry:
            available = [a.sensor is not None, a.video is not None]
            condition = "both" if all(available) else "sensor_only" if available[0] else "video_only" if available[1] else "both_missing"
            append(a.telemetry, {"status": "error", "condition": condition, "availability": available,
                                "timing_seconds": {"total": time.perf_counter()-start}}, error_code="invalid_input_or_bundle")
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "message": str(exc)}, allow_nan=False), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
