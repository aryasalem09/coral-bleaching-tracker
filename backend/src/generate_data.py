"""Deprecated synthetic-data generator.

Synthetic label generation is intentionally not part of the production pipeline.
This module is kept only to make the deprecation explicit for older workflows.
"""


if __name__ == "__main__":
    raise SystemExit(
        "Synthetic data generation is disabled for this project. "
        "Use the real observed-data pipeline in backend/ml/build_dataset.py instead."
    )
