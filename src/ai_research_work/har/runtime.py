"""Explicit CPU classification API; no acceptance policy or automatic model retrieval."""
import numpy as np
import torch
from .models import SensorCNN, MODELS
from .bundles import Bundle, load_bundle
from .fusion import fuse_probabilities, validate_order, FUSIONS
from .preprocessing import validate_batch, TrainingScaler


def outputs(network, x, *, action_ids, labels):
    """Return logits/probabilities [B,K], embeddings [B,64], and ordered identities."""
    ids, names = validate_order(action_ids, labels)
    if type(network) is SensorCNN:
        shape = (128,6)
    elif type(network) in MODELS.values():
        shape = (16,128)
    else:
        raise ValueError('Unsupported encoder')
    if network.head.out_features != len(ids):
        raise ValueError('Output class identity/count mismatch')
    a = validate_batch(x, shape)
    if next(network.parameters()).device.type != 'cpu':
        raise ValueError('This inference interface requires a CPU model')
    network.eval()
    with torch.inference_mode():
        embedding = network.embed(torch.from_numpy(a))
        logits = network.head(embedding)
        probabilities = torch.softmax(logits, dim=1)
    if not torch.isfinite(logits).all() or not torch.isfinite(embedding).all():
        raise ValueError('Model produced nonfinite outputs')
    return {'action_ids':ids, 'labels':names, 'logits':logits.numpy(),
            'embeddings':embedding.numpy(), 'probabilities':probabilities.numpy()}


def predict(bundle_path=None, *, model='hof_gru', fusion='arithmetic',
            sensor_input=None, video_input=None, trusted=False):
    """Classify resampled sensor [B,128,6] and/or HOF [B,16,128].

    Sensor input is not yet standardized; the bundle's training-only statistics
    are applied once. None means missing; malformed supplied arrays are errors.
    Pass a trusted-local bundle path or a previously validated Bundle instance.
    """
    if model not in MODELS or fusion not in FUSIONS:
        raise ValueError('Unknown model or fusion')
    if sensor_input is None and video_input is None:
        return fuse_probabilities(fusion=fusion)
    if sensor_input is not None:
        sensor_input = validate_batch(sensor_input, (128,6))
    if video_input is not None:
        video_input = validate_batch(video_input, (16,128))
    if sensor_input is not None and video_input is not None and len(sensor_input) != len(video_input):
        raise ValueError('Modality batch-size mismatch')
    if isinstance(bundle_path, Bundle):
        bundle = bundle_path
        if bundle.metadata['model'] != model:
            raise ValueError('Requested model does not match loaded bundle')
    else:
        if bundle_path is None:
            raise ValueError('A trusted compatible bundle is required for supplied inputs')
        bundle = load_bundle(bundle_path, model=model, trusted=trusted)
    identity = {'action_ids':bundle.metadata['action_ids'], 'labels':bundle.metadata['labels']}
    s = None if sensor_input is None else outputs(bundle.sensor_model, bundle.scaler.transform(sensor_input), **identity)
    v = None if video_input is None else outputs(bundle.video_model, video_input, **identity)
    return fuse_probabilities(s, v, fusion=fusion)


def demo(*, model='hof_gru', fusion='arithmetic', seed=17):
    """Synthetic untrained shape/fusion/fallback demonstration, not an accuracy test."""
    if model not in MODELS or fusion not in FUSIONS:
        raise ValueError('Unknown model or fusion')
    # fork_rng preserves the caller's torch RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        sensor = SensorCNN(27).eval(); video = MODELS[model](27).eval()
    rng = np.random.default_rng(seed)
    identity = {'action_ids':list(range(1,28)), 'labels':[f'action_{i:02d}' for i in range(1,28)]}
    s = outputs(sensor, rng.normal(size=(2,128,6)).astype(np.float32), **identity)
    v = outputs(video, rng.normal(size=(2,16,128)).astype(np.float32), **identity)
    result = {name:fuse_probabilities(a,b,fusion=fusion) for name,a,b in
              [('both',s,v),('sensor_only',s,None),('video_only',None,v),('neither',None,None)]}
    return {'evidence':'synthetic_untrained_demonstration_not_benchmark', 'model':model,
            'fusion':fusion, 'sensor_parameters':sum(p.numel() for p in sensor.parameters()),
            'video_parameters':sum(p.numel() for p in video.parameters()), 'cases':result}
