"""Inference stage: the algorithms and the driver that writes ``Results/``.

Modules
-------
core        algorithm_1 / algorithm_2, InferenceResult, conformal helpers, decompose
ci_methods  closed-form / bootstrap / quantile CI builders for the functionals
methods     uniform per-task wrappers (alg1, alg2, alg3 one-piece, synth_only) +
            the Pew multidimensional path
registry    declarative apps / task-defs / fixed alpha allocations / algorithms
run_inference  CLI driver: data + registry -> Results/
"""

from .core import InferenceResult, algorithm_1, algorithm_2, decompose

__all__ = ["InferenceResult", "algorithm_1", "algorithm_2", "decompose"]
