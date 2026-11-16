"""Independent synthetic runtime checks; no trained-model quality claim."""
import copy
import hashlib
import json
import socket
import subprocess
import sys

import cv2
import numpy as np
import pytest
import torch

from ai_research_work.har import SensorCNN, HOFGRU, HOFTransformer, fuse_probabilities, save_bundle, load_bundle, predict
from ai_research_work.har.runtime import outputs
from ai_research_work.models import SensorModel, VideoModel
from ai_research_work.har.preprocessing import TrainingScaler, resample_sensor, hof
from ai_research_work.f1.preprocessing import hof as legacy_hof


@pytest.mark.parametrize("constructor,shape,count", [
    (SensorCNN, (2, 128, 6), 29883),
    (HOFGRU, (2, 16, 128), 39259),
    (HOFTransformer, (2, 16, 128), 40763),
])
def test_encoder_parameter_and_embedding_contract(constructor, shape, count):
    torch.set_num_threads(1)
    net = constructor(classes=27).eval()
    assert sum(p.numel() for p in net.parameters() if p.requires_grad) == count
    x = torch.linspace(-1, 1, int(np.prod(shape))).reshape(shape)
    with torch.inference_mode():
        logits, embedding = net(x), net.embed(x)
    assert logits.shape == (2, 27)
    assert embedding.shape == (2, 64)
    assert torch.isfinite(logits).all() and torch.isfinite(embedding).all()
    # Batch members are independent examples, not an accidentally shared pooled axis.
    with torch.inference_mode():
        individual = torch.cat([net(x[i:i + 1]) for i in range(2)])
    torch.testing.assert_close(logits, individual, atol=2e-6, rtol=2e-5)


def test_legacy_eight_class_constructors_remain_separate():
    sensor, video = SensorModel(), VideoModel(8, 128)
    assert sum(p.numel() for m in (sensor, video) for p in m.parameters()) == 66672
    assert sensor.head.out_features == video.head.out_features == 8


@pytest.mark.parametrize("constructor,shape", [
    (SensorCNN, (128, 6)), (HOFGRU, (16, 128)), (HOFTransformer, (16, 128))
])
def test_encoders_reject_invalid_supplied_tensors(constructor, shape):
    net = constructor().eval()
    bad = [torch.zeros(*shape), torch.zeros(0, *shape),
           torch.zeros(1, shape[0] - 1, shape[1]),
           torch.full((1, *shape), float("nan")), torch.zeros(1, *shape, dtype=torch.int64)]
    for x in bad:
        with pytest.raises(ValueError):
            net(x)


def test_sensor_preprocessing_and_fixed_training_scaler():
    raw = np.array([[1, 2, 3, 4, 5, 6], [11, 12, 13, 14, 15, 16]], dtype=np.float32)
    x = resample_sensor(raw)
    expected = raw[0] + np.linspace(0, 10, 128)[:, None]
    np.testing.assert_allclose(x, expected, atol=1e-6, rtol=0)
    mean = np.arange(6, dtype=np.float32).reshape(1, 1, 6)
    std = np.full((1, 1, 6), 2, dtype=np.float32)
    scaler = TrainingScaler(mean, std)
    mean[:] = 100  # Constructor must own its supplied normalization statistics.
    one = scaler.transform(x[None])
    np.testing.assert_allclose(one, (x[None] - np.arange(6)) / 2, rtol=0, atol=1e-6)
    batch = scaler.transform(np.stack([x, x + 500]))
    np.testing.assert_array_equal(one[0], batch[0])
    assert not hasattr(scaler, "fit")
    with pytest.raises(ValueError):
        TrainingScaler(np.zeros((1, 1, 6)), np.zeros((1, 1, 6)))
    for bad in [np.zeros((1, 6)), np.zeros((4, 5)), np.full((3, 6), np.inf)]:
        with pytest.raises(ValueError):
            resample_sensor(bad)


def test_synthetic_flow_recipe_matches_legacy_preprocessing():
    # A translated synthetic textured patch exercises real flow and histogram bins.
    rng = np.random.default_rng(103)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[25:85, 30:95] = rng.integers(0, 256, (60, 65, 3), dtype=np.uint8)
    frames = np.stack([np.roll(frame, shift, axis=1) for shift in (0, 3, 6)])
    actual = hof(frames)
    np.testing.assert_array_equal(actual, legacy_hof(frames))
    assert actual.shape == (16, 128) and actual.dtype == np.float32
    assert np.isfinite(actual).all() and (actual >= 0).all()
    assert actual.sum() == pytest.approx(1, abs=2e-6)


def record(p, ids=(9, 2, 17)):
    return {"probabilities": np.asarray(p, dtype=np.float64), "action_ids": list(ids),
            "labels": [f"synthetic_{i}" for i in ids]}


@pytest.mark.parametrize("fusion", ["arithmetic", "product"])
def test_fusion_and_explicit_order(fusion):
    s = record([[.6, .3, .1], [1., 0., 0.]])
    v = record([[.1, .2, .7], [0., 1., 0.]])
    out = fuse_probabilities(s, v, fusion=fusion)
    if fusion == "arithmetic":
        expected = np.array([[.35, .25, .4], [.5, .5, 0.]])
    else:
        # Independent normalized multiplication is stable for this selected fixture.
        expected = np.maximum(s["probabilities"], 1e-12) * np.maximum(v["probabilities"], 1e-12)
        expected /= expected.sum(axis=1, keepdims=True)
        assert out["probabilities"].dtype == np.float64
    np.testing.assert_allclose(out["probabilities"], expected, atol=1e-15, rtol=1e-14)
    np.testing.assert_allclose(out["probabilities"].sum(axis=1), 1, atol=1e-15)
    assert np.isfinite(out["probabilities"]).all()
    assert out["action_ids"] == [9, 2, 17]
    assert out["predicted_action_ids"] == [17, 9]
    assert out["status"] == "prediction_only" and out["acceptance"] == "not_evaluated"


@pytest.mark.parametrize("fusion", ["arithmetic", "product"])
@pytest.mark.parametrize("branch", ["sensor", "video"])
def test_single_branch_probabilities_are_unchanged(fusion, branch):
    r = record([[0., .7, .3]])
    r["probabilities"] = r["probabilities"].astype(np.float32)
    out = fuse_probabilities(**{branch: r}, fusion=fusion)
    np.testing.assert_array_equal(out["probabilities"], r["probabilities"])
    assert out["probabilities"].dtype == np.float32
    assert out["modalities"] == [branch]


@pytest.mark.parametrize("mutation", ["order", "labels", "batch", "negative", "nan", "zero", "unnormalized", "integer"])
def test_malformed_branch_is_an_error_not_missingness(mutation):
    s = record([[.6, .3, .1]])
    v = copy.deepcopy(s)
    if mutation == "order":
        v["action_ids"] = [2, 9, 17]
    elif mutation == "labels":
        v["labels"][0] = "different_label"
    elif mutation == "batch":
        v["probabilities"] = np.repeat(v["probabilities"], 2, axis=0)
    else:
        values = {"negative": [[1.1, -.1, 0]], "nan": [[np.nan, .3, .1]],
                  "zero": [[0., 0., 0.]], "unnormalized": [[.3, .3, .1]],
                  "integer": [[1, 0, 0]]}
        v["probabilities"] = np.array(values[mutation])
    with pytest.raises(ValueError):
        fuse_probabilities(s, v)


@pytest.mark.parametrize("model", ["hof_gru", "hof_transformer"])
def test_no_input_avoids_loading_and_network(monkeypatch, model):
    def forbidden(*args, **kwargs):
        raise AssertionError("No-input inference attempted I/O")
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr("ai_research_work.har.runtime.load_bundle", forbidden)
    out = predict("does-not-exist", model=model, fusion="product")
    assert out["status"] == "insufficient_input" and out["probabilities"] is None


def synthetic_bundle(path, model, classes=27):
    ctor = HOFGRU if model == "hof_gru" else HOFTransformer
    ids = list(range(classes, 0, -1))
    labels = [f"synthetic_action_{i}" for i in ids]
    sensor, video = SensorCNN(classes).eval(), ctor(classes).eval()
    scaler = TrainingScaler(np.zeros((1, 1, 6)), np.full((1, 1, 6), 2.))
    save_bundle(path, sensor_model=sensor, video_model=video, model=model,
                action_ids=ids, labels=labels, scaler=scaler,
                provenance={"seed": 17, "protocol": "synthetic_untrained", "fold": "synthetic",
                            "training_subjects": [], "test_subjects": []})
    return sensor, video, scaler, ids, labels


@pytest.mark.parametrize("model", ["hof_gru", "hof_transformer"])
def test_synthetic_bundle_roundtrip_and_runtime(tmp_path, monkeypatch, model):
    sensor, video, scaler, ids, labels = synthetic_bundle(tmp_path / "bundle", model)
    actual_load = torch.load
    calls = []
    def checked_load(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs.get("weights_only") is True and kwargs.get("map_location") == "cpu"
        return actual_load(*args, **kwargs)
    monkeypatch.setattr(torch, "load", checked_load)
    b = load_bundle(tmp_path / "bundle", model=model, action_ids=ids, labels=labels, trusted=True)
    assert len(calls) == 2
    x = np.ones((2, 128, 6), dtype=np.float32)
    h = np.zeros((2, 16, 128), dtype=np.float32)
    s = outputs(sensor, scaler.transform(x), action_ids=ids, labels=labels)
    v = outputs(video, h, action_ids=ids, labels=labels)
    for fusion in ("arithmetic", "product"):
        out = predict(b, model=model, fusion=fusion, sensor_input=x, video_input=h)
        np.testing.assert_array_equal(out["probabilities"], fuse_probabilities(s, v, fusion=fusion)["probabilities"])
        assert out["status"] == "prediction_only" and out["acceptance"] == "not_evaluated"
        for kwargs, expected in (({"sensor_input": x}, s), ({"video_input": h}, v)):
            np.testing.assert_array_equal(predict(b, model=model, fusion=fusion, **kwargs)["probabilities"], expected["probabilities"])
    with pytest.raises(ValueError, match="trusted"):
        load_bundle(tmp_path / "bundle", model=model)
    with pytest.raises(ValueError, match="ordered"):
        load_bundle(tmp_path / "bundle", model=model, action_ids=ids[::-1], trusted=True)
    with pytest.raises(ValueError, match="batch"):
        predict(b, model=model, sensor_input=x, video_input=h[:1])
    with pytest.raises(ValueError):
        predict(b, model=model, sensor_input=np.full_like(x, np.nan))


@pytest.mark.parametrize("field", ["schema", "class_count", "preprocessing", "integrity", "filename"])
def test_bundle_rejects_incompatible_metadata_or_tampering(tmp_path, field):
    path = tmp_path / "bundle"
    synthetic_bundle(path, "hof_gru")
    m = json.loads((path / "bundle.json").read_text())
    if field == "schema":
        m["schema"] = "unsupported-version"
    elif field == "class_count":
        m["class_count"] = 8
    elif field == "preprocessing":
        m["preprocessing"]["sensor_shape"] = [64, 6]
    elif field == "filename":
        m["files"]["sensor"]["filename"] = "../outside.pt"
    else:
        raw = bytearray((path / "sensor.pt").read_bytes())
        raw[-1] ^= 1
        (path / "sensor.pt").write_bytes(raw)
    (path / "bundle.json").write_text(json.dumps(m))
    with pytest.raises(ValueError):
        load_bundle(path, model="hof_gru", trusted=True)


def test_eight_class_checkpoint_cannot_be_relabelled_as_27(tmp_path):
    path = tmp_path / "bundle"
    synthetic_bundle(path, "hof_gru", classes=8)
    m = json.loads((path / "bundle.json").read_text())
    m.update(class_count=27, action_ids=list(range(1, 28)), labels=[f"action_{i}" for i in range(1, 28)])
    # Even rewriting both manifest and embedded identity cannot enlarge an 8-class head.
    identity = {k: m[k] for k in ("schema", "model", "class_count", "action_ids", "labels", "preprocessing", "sensor_scaler", "provenance")}
    for branch in ("sensor", "video"):
        f = path / f"{branch}.pt"
        checkpoint = torch.load(f, weights_only=True, map_location="cpu")
        checkpoint["identity"] = identity
        torch.save(checkpoint, f)
        raw = f.read_bytes()
        m["files"][branch].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    (path / "bundle.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="shape"):
        load_bundle(path, model="hof_gru", trusted=True)


@pytest.mark.parametrize("model", ["hof_gru", "hof_transformer"])
@pytest.mark.parametrize("fusion", ["arithmetic", "product"])
def test_cli_demo_is_explicitly_untrained(model, fusion):
    p = subprocess.run([sys.executable, "-m", "ai_research_work.har", "demo", "--model", model,
                        "--fusion", fusion], capture_output=True, text=True, check=True, timeout=60)
    out = json.loads(p.stdout)
    assert out["evidence"] == "synthetic_untrained_demonstration_not_benchmark"
    assert set(out["cases"]) == {"both", "sensor_only", "video_only", "neither"}
    assert out["cases"]["neither"]["status"] == "insufficient_input"
    assert "accuracy" not in out


def test_cli_local_bundle_and_explicit_bad_input(tmp_path):
    path = tmp_path / "bundle"
    synthetic_bundle(path, "hof_gru")
    sensor = tmp_path / "synthetic.npy"
    np.save(sensor, np.ones((3, 6), dtype=np.float32))
    cmd = [sys.executable, "-m", "ai_research_work.har", "predict", "--model", "hof_gru",
           "--fusion", "product", "--bundle", str(path), "--sensor", str(sensor)]
    untrusted = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert untrusted.returncode == 2 and "trusted" in untrusted.stderr
    p = subprocess.run(cmd + ["--trust-local-bundle"], capture_output=True, text=True, check=True, timeout=60)
    out = json.loads(p.stdout)
    assert out["status"] == "prediction_only" and out["modalities"] == ["sensor"]
    assert len(out["probabilities"][0]) == 27 and out["action_ids"][0] == 27
    video = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10., (160, 120))
    assert writer.isOpened(), "Synthetic CLI test requires the OpenCV MJPG writer"
    try:
        for intensity in (0, 70, 140):
            writer.write(np.full((120, 160, 3), intensity, dtype=np.uint8))
    finally:
        writer.release()
    both = subprocess.run(cmd + ["--trust-local-bundle", "--video", str(video)],
                          capture_output=True, text=True, check=True, timeout=60)
    assert json.loads(both.stdout)["modalities"] == ["sensor", "video"]
    np.save(sensor, np.zeros((3, 5)))
    bad = subprocess.run(cmd + ["--trust-local-bundle"], capture_output=True, text=True, timeout=60)
    assert bad.returncode == 2 and "error:" in bad.stderr
