"""LENS interaction-graph extraction core.

The package exposes two statistical contracts built on the same finite-grid
interaction vectors:

* ``LENS-Model`` conditions on a fitted teacher and uses independent inference
  data.
* ``LENS-Stable`` uses independent discovery, confirmation, and evaluation
  samples with label/refit calibration.
"""

from .grids import Grid1D, PairGrid, build_grid
from .interaction_vectors import InteractionSample, compute_interaction_sample
from .model_inference import (
    ModelInferenceResult,
    ReplicationInferenceResult,
    run_model_inference,
    run_replication_inference,
    run_replication_inference_streaming,
)
from .pipeline import StableLensConfig, StableLensResult, run_stable_lens
from .ranking import rank_interactions
from .shared_confirmation import (
    SharedExactConfirmationResult,
    run_shared_exact_confirmation,
)
from .splits import ThreeWaySplit, make_three_way_split
from .teachers import (
    SoftplusMLPTeacherFactory,
    TeacherFactory,
    XGBoostTeacherFactory,
)

__all__ = [
    "Grid1D",
    "PairGrid",
    "InteractionSample",
    "ModelInferenceResult",
    "ReplicationInferenceResult",
    "SharedExactConfirmationResult",
    "SoftplusMLPTeacherFactory",
    "StableLensConfig",
    "StableLensResult",
    "TeacherFactory",
    "ThreeWaySplit",
    "XGBoostTeacherFactory",
    "build_grid",
    "compute_interaction_sample",
    "make_three_way_split",
    "rank_interactions",
    "run_model_inference",
    "run_replication_inference",
    "run_replication_inference_streaming",
    "run_shared_exact_confirmation",
    "run_stable_lens",
]
