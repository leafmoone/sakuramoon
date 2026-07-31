"""Explicit evaluation entry points for SakuraMoon."""

from sakuramoon.eval.artifacts import (
    CheckpointMetricComparison,
    EvaluationArtifact,
    write_evaluation_artifact,
)
from sakuramoon.eval.manual_quality import (
    ManualQualityObservation,
    ManualQualityReport,
    summarize_manual_quality,
)
from sakuramoon.eval.metrics import (
    FeatureStats,
    InceptionScore,
    frechet_inception_distance,
    inception_score,
)
from sakuramoon.eval.schedule import ScheduledEvaluation, scheduled_evaluations
from sakuramoon.eval.spec import (
    CheckpointRef,
    EvaluationCost,
    EvaluationJob,
    PromptCase,
    PromptManifest,
)
from sakuramoon.eval.vae_reconstruction import (
    ReconstructionBatch,
    ReconstructionBatchSpec,
    ReconstructionCohortManifest,
    ReconstructionCohortMember,
    ReconstructionEvaluation,
    ReconstructionMetricIdentity,
    ReconstructionObservation,
    ReconstructionQualityReport,
    canonical_reconstruction_cohort_manifest_bytes,
    evaluate_reconstruction_batches,
    summarize_reconstruction_quality,
    write_reconstruction_evaluation,
)

__all__ = [
    "CheckpointMetricComparison",
    "CheckpointRef",
    "EvaluationArtifact",
    "EvaluationCost",
    "EvaluationJob",
    "FeatureStats",
    "InceptionScore",
    "ManualQualityObservation",
    "ManualQualityReport",
    "PromptCase",
    "PromptManifest",
    "ReconstructionBatch",
    "ReconstructionBatchSpec",
    "ReconstructionCohortManifest",
    "ReconstructionCohortMember",
    "ReconstructionEvaluation",
    "ReconstructionMetricIdentity",
    "ReconstructionObservation",
    "ReconstructionQualityReport",
    "ScheduledEvaluation",
    "canonical_reconstruction_cohort_manifest_bytes",
    "evaluate_reconstruction_batches",
    "frechet_inception_distance",
    "inception_score",
    "scheduled_evaluations",
    "summarize_manual_quality",
    "summarize_reconstruction_quality",
    "write_evaluation_artifact",
    "write_reconstruction_evaluation",
]
