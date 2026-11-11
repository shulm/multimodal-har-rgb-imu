"""MAT sensor reader and relative-time resampling."""
import numpy as np
from scipy.io import loadmat

def sensor_array(path, steps=None):
    data = loadmat(path)
    if "d_iner" not in data:
        raise ValueError(f"Missing d_iner: {path}")
    x = np.asarray(data["d_iner"], dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 6 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError(f"Invalid sensor array {path}: {x.shape}")
    if steps is not None:
        source = np.linspace(0, 1, len(x))
        target = np.linspace(0, 1, steps)
        x = np.stack([np.interp(target, source, x[:, j]) for j in range(6)], axis=1).astype(np.float32)
    return x
