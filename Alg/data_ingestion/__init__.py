"""Data-ingestion stage: load each application's raw / processed data into the
single unified ``TaskSet`` representation that every algorithm consumes.

Modules
-------
loaders          per-application adapters -> TaskSet (S_real, S_synth, theta*, n_j, N_j)
simulate         generator for simulated exchangeable Bernoulli tasks
build_data_info  CLI: write Data/<app>/info.json (task counts, n_j / N_j)
"""

from .loaders import Task, TaskSet

__all__ = ["Task", "TaskSet"]
