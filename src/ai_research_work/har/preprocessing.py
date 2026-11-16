"""Completed-clip preprocessing; no normalization fitting or physical-axis guessing."""
import cv2
import numpy as np

PREPROCESSING = {
    'version': 'rgb-imu-hof-1', 'sensor_shape': [128, 6],
    'sensor_resampling': 'float32-linear-normalized-sample-index',
    'sensor_columns': 'native-six-column-order-no-unit-conversion',
    'scaler': 'training-only-channel-mean-std-ddof0-floor1e-5',
    'rgb_size': [160, 120], 'resize': 'INTER_AREA',
    'color': 'RGB-to-gray', 'decode': 'all-frames-reported-count-checked',
    'farneback': [0.5, 3, 9, 3, 5, 1.1, 0],
    'hof_shape': [16, 128], 'spatial_cells': [4, 4], 'angle_bins': 8,
    'magnitude_threshold': 0.5, 'normalization_epsilon': 1e-6,
    'temporal_bins': 'min((i-1)*16//(N-1),15)',
}


def validate_batch(x, shape):
    a = np.asarray(x)
    if a.dtype.kind not in 'fiu' or a.ndim != 3 or a.shape[1:] != shape or not len(a):
        raise ValueError(f'Expected numeric nonempty batch with trailing shape {shape}')
    a = a.astype(np.float32)
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite input')
    return a


def resample_sensor(x):
    a = np.asarray(x)
    if a.dtype.kind not in 'fiu':
        raise ValueError('Sensor input must be numeric')
    a = a.astype(np.float32)
    if a.ndim != 2 or a.shape[1] != 6 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError('Expected finite N>=2 by six sensor channels')
    source = np.linspace(0, 1, len(a)); target = np.linspace(0, 1, 128)
    return np.stack([np.interp(target, source, a[:, j]) for j in range(6)], axis=1).astype(np.float32)


class TrainingScaler:
    """Immutable inference statistics supplied by a compatible training artifact.

    No fit method is exposed: never estimate these values on inference queries.
    """
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32, copy=True)
        self.std = np.array(std, dtype=np.float32, copy=True)
        if (self.mean.shape != (1, 1, 6) or self.std.shape != (1, 1, 6)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all()
                or (self.std < 1e-5).any()):
            raise ValueError('Scaler requires finite [1,1,6] mean and std >= 1e-5')
        self.mean.flags.writeable = False; self.std.flags.writeable = False

    def transform(self, x):
        out = (validate_batch(x, (128, 6)) - self.mean) / self.std
        if not np.isfinite(out).all():
            raise ValueError('Scaling produced nonfinite values')
        return out

    def to_dict(self):
        return {'mean': self.mean.tolist(), 'std': self.std.tolist()}


def decode_video(path):
    cap = cv2.VideoCapture(str(path)); frames = []
    try:
        if not cap.isOpened():
            raise ValueError('Cannot open supplied video')
        expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.resize(bgr, (160, 120), interpolation=cv2.INTER_AREA)[..., ::-1])
    finally:
        cap.release()
    if not frames or (expected > 0 and len(frames) != expected):
        raise ValueError('Supplied video is empty or truncated')
    return np.stack(frames)


def hof(rgb):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[1:] != (120, 160, 3) or not len(rgb):
        raise ValueError('Expected nonempty uint8 RGB frames [N,120,160,3]')
    grey = np.stack([cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2GRAY) for f in rgb])
    hist = np.zeros((16, 4, 4, 8), np.float32)
    for i in range(1, len(grey)):
        flow = cv2.calcOpticalFlowFarneback(grey[i-1], grey[i], None, .5, 3, 9, 3, 5, 1.1, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1]); keep = mag > .5
        b = min((i-1)*16//(len(grey)-1), 15)
        if keep.any():
            yy, xx = np.nonzero(keep)
            np.add.at(hist[b], (yy*4//120, xx*4//160,
                np.minimum((ang[keep]/(2*np.pi)*8).astype(int), 7)), mag[keep])
    hist = hist.reshape(16, 128); hist /= hist.sum() + 1e-6
    if not np.isfinite(hist).all():
        raise ValueError('Optical flow produced nonfinite HOF')
    return hist
