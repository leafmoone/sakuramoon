"""Coordinate production observers after a successful training update."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sakuramoon.config.schema import RuntimeConfig
    from sakuramoon.distributed import DistributedProgress
    from sakuramoon.eval.runtime import TrainingEvaluator
    from sakuramoon.train.benchmark import BenchmarkTrace
    from sakuramoon.train.runtime import SuccessfulTrainingObservation
    from sakuramoon.train.sampling import TrainingSampler


class ProductionObserverTelemetry(Protocol):
    """The observer and publication surface shared by real and no-op telemetry."""

    def observer(self, observation: SuccessfulTrainingObservation, /) -> None: ...

    def submit_wandb_images(
        self,
        image_paths: tuple[Path, ...],
        captions: tuple[str, ...],
        *,
        successful_update: int,
    ) -> None: ...

    def submit_wandb_metrics(
        self,
        metrics: Mapping[str, int | float],
        *,
        successful_update: int,
    ) -> None: ...


def observe_production_successful_update(
    observation: SuccessfulTrainingObservation,
    *,
    config: RuntimeConfig,
    benchmark_trace: BenchmarkTrace | None,
    training_sampler: TrainingSampler | None,
    telemetry: ProductionObserverTelemetry,
    progress: DistributedProgress,
    evaluator: TrainingEvaluator | None,
    evaluation_is_splits: int | None,
    is_main_process: bool,
    get_growth_alpha: Callable[[int], float],
) -> None:
    """Run sampling, telemetry, and evaluation in the shared rank-stage order."""

    update = observation.loop.update.state.successful_updates
    if benchmark_trace is not None:
        benchmark_trace.append(observation)
    sampling_due = (
        config.sampling.training.enabled
        and update > 0
        and update % config.sampling.training.every_updates == 0
    )

    def sample_and_publish() -> None:
        if training_sampler is None:
            raise RuntimeError("rank zero training sampler is unavailable")
        with observation.phase_timer.record("sample"):
            training_sampler.set_growth_alpha(get_growth_alpha(update))
            samples = training_sampler.sample(update, observation.microbatches)
        telemetry.submit_wandb_images(
            samples.paths,
            samples.captions,
            successful_update=update,
        )
        telemetry.submit_wandb_metrics(
            {
                "training_samples/count": len(samples.paths),
                **{
                    f"condition_diagnostics/{key}": value
                    for key, value in samples.diagnostics.items()
                },
            },
            successful_update=update,
        )

    if sampling_due:
        progress.run_on_rank(
            f"observer/update-{update}/training-samples",
            0,
            sample_and_publish,
        )
    progress.run_on_rank(
        f"observer/update-{update}/telemetry",
        0,
        lambda: telemetry.observer(observation),
    )
    if evaluator is not None and evaluator.due(update):
        if evaluation_is_splits is None:
            raise RuntimeError("evaluation config is unavailable")
        evaluator.set_growth_alpha(get_growth_alpha(update))
        evaluation = evaluator.evaluate(update)
        # Every rank participates in the same named stage.
        # Long rank-zero-only generation must keep a heartbeat
        # alive while the other ranks wait.
        current_evaluator = evaluator
        suite_metrics = progress.run_on_rank(
            f"observer/update-{update}/concept-suite",
            0,
            lambda: current_evaluator.run_concept_suite(update),
        )
        if is_main_process:
            if evaluation is None:
                raise RuntimeError("rank zero evaluation is unavailable")
            telemetry.submit_wandb_metrics(
                {
                    "evaluation/fid": evaluation.fid,
                    "evaluation/inception_score_mean": (
                        evaluation.inception_score_mean
                    ),
                    "evaluation/inception_score_std": (evaluation.inception_score_std),
                    "evaluation/kid_mean": evaluation.kid_mean,
                    "evaluation/kid_std": evaluation.kid_std,
                    "evaluation/cmmd": evaluation.cmmd,
                    "evaluation/sample_count": evaluation.sample_count,
                    "evaluation/real_sample_count": (evaluation.real_sample_count),
                    "evaluation/is_splits": evaluation_is_splits,
                },
                successful_update=update,
            )

            if suite_metrics:
                telemetry.submit_wandb_metrics(
                    {
                        f"evaluation/concept/{key}": value
                        for key, value in suite_metrics.items()
                    },
                    successful_update=update,
                )
    progress.synchronize(f"observer/update-{update}/complete")
