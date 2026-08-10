from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict

import pandas as pd
import torch

from xai_pipeline.contracts import Example, MetricRecord, Prediction


def explain_with_timing(explainer, examples: Sequence[Example], model, predictions: Sequence[Prediction]):
    """Runs an explainer one example at a time (not batched), timing each
    call individually, so per-example compute cost can be reported
    alongside the other comparison metrics.

    Returns (explanations, timing_dataframe) -- explanations are exactly
    what explainer.explain() would normally return; timing_dataframe has
    one row per example with wall-clock seconds and peak GPU memory
    (if running on CUDA).

    NOTE / ASSUMPTION: measures wall-clock time only, not separating
    out data-loading vs. pure-compute time, and peak memory is the
    device-wide peak since the last reset (not perfectly isolated to
    just this one call if other work is happening concurrently) --
    accurate enough for relative method-to-method comparison, which is
    the actual goal here, but worth flagging as an approximation.
    """
    all_explanations = []
    timing_records: list[MetricRecord] = []

    for example, prediction in zip(examples, predictions):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        example_explanations = explainer.explain([example], model, [prediction])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        peak_memory_mb = None
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        all_explanations.extend(example_explanations)

        n_labels = max(len(example_explanations), 1)
        timing_records.append(
            MetricRecord(
                run_id="task2_processing_time",
                example_id=example.example_id,
                method=explainer.name,
                metric="seconds_total",
                value=elapsed,
                slice_name="n_labels_explained",
                slice_value=str(n_labels),
                metadata={"peak_memory_mb": peak_memory_mb},
            )
        )
        timing_records.append(
            MetricRecord(
                run_id="task2_processing_time",
                example_id=example.example_id,
                method=explainer.name,
                metric="seconds_per_label",
                value=elapsed / n_labels,
                slice_name="n_labels_explained",
                slice_value=str(n_labels),
                metadata={"peak_memory_mb": peak_memory_mb},
            )
        )
        if peak_memory_mb is not None:
            timing_records.append(
                MetricRecord(
                    run_id="task2_processing_time",
                    example_id=example.example_id,
                    method=explainer.name,
                    metric="peak_memory_mb",
                    value=peak_memory_mb,
                    slice_name="n_labels_explained",
                    slice_value=str(n_labels),
                    metadata={},
                )
            )

    timing_df = pd.DataFrame([asdict(r) for r in timing_records])
    return all_explanations, timing_df