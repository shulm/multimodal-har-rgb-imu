"""RGB-IMU classification with explicit architectures, fusion and trusted bundles."""
from .models import SensorCNN, HOFGRU, HOFTransformer
from .preprocessing import PREPROCESSING, TrainingScaler, resample_sensor, decode_video, hof
from .fusion import fuse_probabilities
from .bundles import SCHEMA, Bundle, save_bundle, load_bundle
from .runtime import outputs, predict, demo

__all__ = ['SensorCNN','HOFGRU','HOFTransformer','TrainingScaler','PREPROCESSING',
           'resample_sensor','decode_video','hof','fuse_probabilities','SCHEMA',
           'Bundle','save_bundle','load_bundle','outputs','predict','demo']
