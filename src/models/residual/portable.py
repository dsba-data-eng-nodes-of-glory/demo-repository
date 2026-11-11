"""Portable, weights-only residual autoencoder checkpoint conversion and loading."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

class PortableRobustScaler:
    def __init__(self, center, scale):
        self.center = np.asarray(center, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
    def transform(self, data):
        return (np.asarray(data, dtype=np.float64) - self.center) / self.scale

def safe_tensors(value):
    if isinstance(value, np.ndarray): return torch.as_tensor(value)
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: safe_tensors(v) for k, v in value.items()}
    if isinstance(value, list): return [safe_tensors(v) for v in value]
    if isinstance(value, tuple): return tuple(safe_tensors(v) for v in value)
    if isinstance(value, (str, int, float, bool, type(None), torch.Tensor)): return value
    raise TypeError(f"Unsupported checkpoint object: {type(value)}")

def save_portable(bundle: dict, destination: str | Path) -> None:
    data = dict(bundle)
    scaler = data.pop("scaler")
    data["scaler_center"] = scaler.center_
    data["scaler_scale"] = scaler.scale_
    torch.save(safe_tensors(data), destination)

def load_bundle(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix.lower() != '.pt':
        raise ValueError("Only weights-only .pt residual-load checkpoints are accepted")
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model_state = saved.pop('model_state_dict')
    def numpy(value):
        if isinstance(value, torch.Tensor): return value.cpu().numpy()
        if isinstance(value, dict): return {k: numpy(v) for k,v in value.items()}
        if isinstance(value, list): return [numpy(v) for v in value]
        if isinstance(value, tuple): return tuple(numpy(v) for v in value)
        return value
    saved = numpy(saved)
    saved['model_state_dict'] = model_state
    saved['scaler'] = PortableRobustScaler(saved.pop('scaler_center'), saved.pop('scaler_scale'))
    if saved.get('schema') != 'residual_ae_v2_capacity':
        raise ValueError('Checkpoint must be capacity-aware residual_ae_v2_capacity')
    from src.models.residual.residual_ae_common import FEATURE_NAMES
    if saved.get('feature_names') != FEATURE_NAMES:
        raise ValueError('Residual checkpoint has incompatible feature schema')
    return saved
