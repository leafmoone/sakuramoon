from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline-tune GEMM shapes recorded by DTK TunableOp."
    )
    parser.add_argument("--untuned", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--max-duration-ms", type=int, default=50)
    parser.add_argument(
        "--max-operations",
        type=int,
        default=0,
        help="Tune at most this many supported unique GEMMs (0 means all).",
    )
    args = parser.parse_args()
    untuned = args.untuned.resolve(strict=True)
    results = args.results.resolve()
    if not 1 <= args.max_duration_ms <= 1000:
        raise ValueError("max duration must be in [1, 1000] milliseconds")
    if args.max_operations < 0:
        raise ValueError("max operations must be non-negative")
    results.parent.mkdir(parents=True, exist_ok=True)

    tuned: set[tuple[str, str]] = set()
    if results.is_file():
        for raw_line in results.read_text(encoding="utf-8").splitlines():
            fields = raw_line.split(",", maxsplit=2)
            if len(fields) == 3 and fields[0].startswith(("Gemm", "ScaledGemm")):
                tuned.add((fields[0], fields[1]))

    # PyTorch's offline tuner does not support strided-batched GEMMs.  Preserve
    # first-seen order because the recorder emits the hot model GEMMs first,
    # and avoid spending hours retuning duplicate rows from repeated runs.
    operations: list[str] = []
    seen: set[str] = set()
    for raw_line in untuned.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        fields = line.split(",", maxsplit=1)
        if (
            not line.startswith(("GemmTunableOp_", "ScaledGemm"))
            or len(fields) != 2
            or line in seen
            or (fields[0], fields[1]) in tuned
        ):
            continue
        seen.add(line)
        operations.append(line)
        if args.max_operations and len(operations) >= args.max_operations:
            break
    if not operations:
        raise ValueError("untuned file contains no supported unique GEMMs")

    tunable = torch.cuda.tunable
    tunable.set_filename(str(results), insert_device_ordinal=False)
    tunable.enable(True)
    tunable.tuning_enable(True)
    tunable.record_untuned_enable(False)
    tunable.set_max_tuning_duration(args.max_duration_ms)
    tunable.write_file_on_exit(True)
    if results.is_file() and results.stat().st_size and not tunable.read_file(
        str(results)
    ):
        raise RuntimeError("TunableOp could not load the existing results file")
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="sakuramoon-tunableop-",
        suffix=".csv",
        dir=results.parent,
    ) as selected:
        selected.write("\n".join(operations) + "\n")
        selected.flush()
        print(f"[tunableop] selected_operations={len(operations)}", flush=True)
        tunable.tune_gemm_in_file(selected.name)
    if not tunable.write_file(str(results)):
        raise RuntimeError("TunableOp did not write a results file")
    print(f"[tunableop] results={results}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
