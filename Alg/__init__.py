"""Task-exchangeability pipeline, organized in three stages:

    Alg/data_ingestion/   raw / processed data  ->  unified TaskSet format
    Alg/inference/        TaskSet + algorithms  ->  Results/
    Alg/result_process/   Results/              ->  Plots/ and summary tables

Shared filesystem locations are defined here so every submodule agrees on where
``Data/``, ``Results/`` and ``Plots/`` live (repo root = parent of ``Alg/``).
"""

from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "Data")
RESULTS_DIR = os.path.join(ROOT, "Results")
PLOTS_DIR = os.path.join(ROOT, "Plots")

__all__ = ["ROOT", "DATA_DIR", "RESULTS_DIR", "PLOTS_DIR"]
