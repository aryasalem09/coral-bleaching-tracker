"""Legacy data-prep entry point.

The observed-data preparation logic now lives in :mod:`backend.ml.build_dataset`.
"""

from backend.ml.build_dataset import build_modeling_dataset


if __name__ == "__main__":
    build_modeling_dataset()

