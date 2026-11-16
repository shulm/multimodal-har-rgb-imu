"""Versioned trusted-local state bundles. Integrity is not authenticity or release permission."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import io
import json
import copy
import torch
from .models import SensorCNN, MODELS
from .preprocessing import PREPROCESSING, TrainingScaler
from .fusion import validate_order, FUSIONS

SCHEMA = 'rgb-imu-har-bundle-1'
MAX_WEIGHTS_BYTES = 16 * 1024 * 1024


def _provenance(value):
    if not isinstance(value, dict):
        raise ValueError('Explicit training provenance required')
    required = {'seed', 'protocol', 'fold', 'training_subjects', 'test_subjects'}
    if not required <= value.keys() or type(value['seed']) is not int:
        raise ValueError('Provenance needs seed, protocol, fold, training_subjects, test_subjects')
    if value['protocol'] not in ('cpu_431', 'cuda_loso', 'synthetic_untrained'):
        raise ValueError('Unknown fitting protocol identity')
    if not isinstance(value['fold'], str) or not value['fold'].strip():
        raise ValueError('Explicit fold identity required')
    for field in ('training_subjects', 'test_subjects'):
        v = value[field]
        if not isinstance(v, list) or any(type(x) is not int or x < 1 for x in v) or len(v) != len(set(v)):
            raise ValueError('Invalid subject identities')
    if set(value['training_subjects']) & set(value['test_subjects']):
        raise ValueError('Training/test subjects must be disjoint')
    if value['protocol'] != 'synthetic_untrained' and (not value['training_subjects'] or not value['test_subjects']):
        raise ValueError('Fitted bundle requires explicit training/test subjects')
    return copy.deepcopy(value)


def _identity(model, ids, labels, scaler, provenance):
    if model not in MODELS:
        raise ValueError('Unknown model architecture')
    ids, labels = validate_order(ids, labels)
    return dict(schema=SCHEMA, model=model, class_count=len(ids), action_ids=ids,
                labels=labels, preprocessing=copy.deepcopy(PREPROCESSING),
                sensor_scaler=scaler.to_dict(), provenance=_provenance(provenance))


def _state(model, expected):
    state = model.state_dict(); shapes = expected.state_dict()
    if state.keys() != shapes.keys():
        raise ValueError('Checkpoint architecture/state-key mismatch')
    for k, v in state.items():
        if not isinstance(v, torch.Tensor) or v.shape != shapes[k].shape or v.dtype != shapes[k].dtype or not torch.isfinite(v).all():
            raise ValueError('Checkpoint shape/dtype/finite-value mismatch')
    return {k: v.detach().cpu().clone() for k,v in state.items()}


@dataclass
class Bundle:
    sensor_model: SensorCNN
    video_model: torch.nn.Module
    scaler: TrainingScaler
    metadata: dict


def save_bundle(path, *, sensor_model, video_model, model, action_ids, labels,
                scaler, provenance, fusion='arithmetic'):
    """Write an explicitly supplied local artifact; never trains or exports it remotely.

    Use only weights/statistics you are entitled to store. Synthetic examples must
    identify protocol='synthetic_untrained'; real models are not shipped here.
    """
    if fusion not in FUSIONS or not isinstance(scaler, TrainingScaler):
        raise ValueError('Invalid fusion/scaler')
    identity = _identity(model, action_ids, labels, scaler, provenance)
    k = identity['class_count']
    if type(sensor_model) is not SensorCNN or type(video_model) is not MODELS[model]:
        raise ValueError('Architecture identity mismatch')
    states = {'sensor': _state(sensor_model, SensorCNN(k)), 'video': _state(video_model, MODELS[model](k))}
    dest = Path(path)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise ValueError('Bundle destination must be new or empty; refusing overwrite')
    dest.mkdir(parents=True, exist_ok=True)
    files = {}
    for branch, state in states.items():
        data = io.BytesIO(); torch.save({'identity': identity, 'branch': branch, 'state_dict': state}, data)
        raw = data.getvalue(); filename = branch + '.pt'; (dest/filename).write_bytes(raw)
        files[branch] = {'filename': filename, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    manifest = dict(identity, branches={'sensor':'sensor_cnn', 'video':model},
                    fusion={'default':fusion, 'supported':list(FUSIONS), 'product_floor':1e-12}, files=files)
    (dest/'bundle.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return dest


def load_bundle(path, *, model, action_ids=None, labels=None, trusted=False):
    """Load CPU-only weights with weights_only=True. No unsafe fallback.

    trusted=True is a caller assertion about artifact origin, not a signature
    verification. SHA256 only checks consistency within that trusted artifact.
    """
    if trusted is not True:
        raise ValueError('Only explicitly trusted local bundles may be loaded')
    if model not in MODELS:
        raise ValueError('Unknown model architecture')
    path = Path(path)
    mp = path/'bundle.json'
    if mp.stat().st_size > 1024*1024:
        raise ValueError('Bundle manifest too large')
    try:
        meta = json.loads(mp.read_text(encoding='utf-8'))
        if meta['schema'] != SCHEMA or meta['model'] != model or meta['preprocessing'] != PREPROCESSING:
            raise ValueError('Bundle version, architecture or preprocessing mismatch')
        ids, names = validate_order(meta['action_ids'], meta['labels'])
        if type(meta['class_count']) is not int or meta['class_count'] != len(ids):
            raise ValueError('Bundle class-count mismatch')
        if action_ids is not None and list(action_ids) != ids:
            raise ValueError('Requested ordered action IDs mismatch')
        if labels is not None and list(labels) != names:
            raise ValueError('Requested ordered labels mismatch')
        if meta['branches'] != {'sensor':'sensor_cnn', 'video':model}:
            raise ValueError('Branch identity mismatch')
        if meta['fusion'] != {'default':meta['fusion']['default'], 'supported':list(FUSIONS), 'product_floor':1e-12} or meta['fusion']['default'] not in FUSIONS:
            raise ValueError('Fusion configuration mismatch')
        scaler = TrainingScaler(**meta['sensor_scaler'])
        identity = _identity(model, ids, names, scaler, meta['provenance'])
        models = {'sensor':SensorCNN(len(ids)), 'video':MODELS[model](len(ids))}
        if set(meta['files']) != set(models):
            raise ValueError('Required branch file inventory mismatch')
        for branch, network in models.items():
            info = meta['files'][branch]
            if info['filename'] != branch+'.pt' or type(info['bytes']) is not int or not 0 < info['bytes'] <= MAX_WEIGHTS_BYTES:
                raise ValueError('Invalid local checkpoint filename/size')
            f = path/info['filename']
            if f.is_symlink() or f.stat().st_size != info['bytes']:
                raise ValueError('Checkpoint file/size mismatch')
            raw = f.read_bytes()
            if hashlib.sha256(raw).hexdigest() != info['sha256']:
                raise ValueError('Checkpoint integrity mismatch')
            value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
            if value['identity'] != identity or value['branch'] != branch:
                raise ValueError('Checkpoint embedded identity mismatch')
            state = value['state_dict']; expected = network.state_dict()
            if not isinstance(state, dict):
                raise ValueError('Checkpoint state must be a tensor mapping')
            if state.keys() != expected.keys():
                raise ValueError('Checkpoint state-key mismatch')
            for key,tensor in state.items():
                if not isinstance(tensor, torch.Tensor) or tensor.shape != expected[key].shape or tensor.dtype != expected[key].dtype or not torch.isfinite(tensor).all():
                    raise ValueError('Checkpoint shape/dtype/finite-value mismatch')
            network.load_state_dict(state, strict=True); network.eval()
    except (KeyError, TypeError, json.JSONDecodeError) as e:
        raise ValueError('Malformed bundle metadata/checkpoint') from e
    return Bundle(models['sensor'], models['video'], scaler, meta)
