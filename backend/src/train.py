"""Legacy training entry point.

Run this module if an older deployment still points at ``backend/src/train.py``.
The real training pipeline now lives in :mod:`backend.ml.train_model`.
"""

from backend.ml.train_model import train_and_evaluate


if __name__ == "__main__":
    train_and_evaluate()

