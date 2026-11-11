"""Completed-clip inertial resampling and optical-flow histograms."""
from pathlib import Path
import cv2
import numpy as np
from ..data import sensor_array


def sensor(path):
    try:
        return sensor_array(Path(path), 128)
    except Exception as exc:
        raise ValueError("Invalid supplied sensor file (expected finite d_iner Nx6, N>=2)") from exc


def decode(path):
    cap = cv2.VideoCapture(str(path))
    small = []
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        if not cap.isOpened():
            raise ValueError("Invalid supplied video: cannot open")
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            small.append(cv2.resize(bgr, (160, 120), interpolation=cv2.INTER_AREA)[..., ::-1])
    finally:
        cap.release()
    if not small or (expected > 0 and len(small) != expected):
        raise ValueError("Invalid supplied video: empty or truncated decode")
    return np.stack(small)


def hof(rgb):
    grey = np.stack([cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2GRAY) for f in rgb])
    hist = np.zeros((16, 4, 4, 8), np.float32)
    for i in range(1, len(grey)):
        flow = cv2.calcOpticalFlowFarneback(grey[i-1], grey[i], None, .5, 3, 9, 3, 5, 1.1, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        keep = mag > .5
        b = min((i-1)*16//(len(grey)-1), 15)
        if keep.any():
            yy, xx = np.nonzero(keep)
            np.add.at(hist[b], (yy*4//120, xx*4//160,
                      np.minimum((ang[keep]/(2*np.pi)*8).astype(int), 7)), mag[keep])
    hist = hist.reshape(16, 128)
    hist /= hist.sum() + 1e-6
    return hist
