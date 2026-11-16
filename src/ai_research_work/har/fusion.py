"""Classification probability combination, deliberately separate from acceptance."""
import numpy as np

FUSIONS = ('arithmetic', 'product')


def validate_order(action_ids, labels):
    ids = list(action_ids); names = list(labels)
    if (len(ids) < 2 or len(ids) != len(names)
            or any(type(x) is not int or x < 1 for x in ids)
            or len(set(ids)) != len(ids)
            or any(not isinstance(x, str) or not x.strip() for x in names)
            or len(set(names)) != len(names)):
        raise ValueError('Explicit unique positive action IDs and unique nonempty labels required')
    return ids, names


def _record(record):
    if not isinstance(record, dict):
        raise ValueError('Branch output must be a record')
    try:
        ids, labels = validate_order(record['action_ids'], record['labels'])
        p = np.asarray(record['probabilities'])
    except (KeyError, TypeError) as e:
        raise ValueError('Missing branch output fields') from e
    if (p.dtype.kind != 'f' or p.ndim != 2 or p.shape[1] != len(ids)
            or not len(p) or not np.isfinite(p).all() or (p < 0).any()
            or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=2e-6, rtol=0)):
        raise ValueError('Expected finite normalized floating [batch, ordered class] probabilities')
    return ids, labels, p


def fuse_probabilities(sensor=None, video=None, *, fusion='arithmetic'):
    if fusion not in FUSIONS:
        raise ValueError('Unknown fusion rule')
    if sensor is None and video is None:
        return {'status': 'insufficient_input', 'action_ids': None, 'labels': None,
                'probabilities': None, 'acceptance': 'not_evaluated'}
    present = [r for r in (sensor, video) if r is not None]
    ids, labels, p = _record(present[0])
    if len(present) == 2:
        other_ids, other_labels, v = _record(present[1])
        if ids != other_ids or labels != other_labels or p.shape != v.shape:
            raise ValueError('Branch batch size or ordered class identity mismatch')
        if fusion == 'arithmetic':
            p = .5 * p + .5 * v
        else:
            z = np.log(np.maximum(p.astype(np.float64), 1e-12)) + np.log(np.maximum(v.astype(np.float64), 1e-12))
            z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    return {'status': 'prediction_only', 'action_ids': ids, 'labels': labels,
            'probabilities': p, 'predicted_action_ids': [ids[i] for i in p.argmax(1)],
            'predicted_labels': [labels[i] for i in p.argmax(1)],
            'fusion': fusion, 'modalities': [name for name,r in [('sensor',sensor),('video',video)] if r is not None],
            'acceptance': 'not_evaluated'}
