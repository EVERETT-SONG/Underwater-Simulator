"""Replace load_data or provide --loader module:function for future CFD formats."""
import importlib
from pathlib import Path
import numpy as np

WRENCH_COLUMNS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")


def load_data(path, branch, loader=None):
    """Return finite float32 arrays: velocity (N,3), wrench (N,6).

    NPZ: keys velocity/wrench. CSV: header vx,vy,vz or wx,wy,wz,
    followed by Fx,Fy,Fz,Mx,My,Mz (column order is arbitrary).
    Custom callable signature: loader(path: str, branch: str) -> (x, y).
    """
    if loader:
        module, function = loader.split(":", 1)
        x, y = getattr(importlib.import_module(module), function)(str(path), branch)
    elif Path(path).suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            x, y = data["velocity"], data["wrench"]
    elif Path(path).suffix.lower() == ".csv":
        data = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
        columns = ("vx", "vy", "vz") if branch == "translation" else ("wx", "wy", "wz")
        missing = set(columns + WRENCH_COLUMNS) - set(data.dtype.names or ())
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        x = np.column_stack([data[c] for c in columns])
        y = np.column_stack([data[c] for c in WRENCH_COLUMNS])
    else:
        raise ValueError("Use .csv, .npz or --loader module:function")
    x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 3 or y.shape != (len(x), 6):
        raise ValueError(f"Expected (N,3)/(N,6), got {x.shape}/{y.shape}")
    if len(x) < 10 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("At least 10 finite samples are required")
    return x, y
