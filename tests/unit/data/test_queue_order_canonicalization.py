"""Queue order must be bit-identical across processes (PYTHONHASHSEED).

Regression for the MBS corrected-100U rerun finding (09-08):
``_QueueStore.new`` previously started its seeded shuffle from
``list(frozenset)``. frozenset[str] iteration order depends on per-process
string hash randomization, so two fresh cycle-0 runs (different processes)
consumed different sample sequences despite identical manifest and seed —
the "deterministic matched-data" reset was only same-process deterministic.

The fix canonicalizes the shuffle input with ``sorted(paths)``. This test
proves the observable contract: same manifest + same cycle + different
PYTHONHASHSEED values (subprocesses, so the interpreter hash seed is
genuinely different) yield a bit-identical row order.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import os
import subprocess
import sys
from pathlib import Path

from sakuramoon.data.manifest import (
    DatasetManifest,
    DatasetSourceIdentity,
    ShardRecord,
)
from sakuramoon.data.service import _QueueStore
from sakuramoon.data.validation import ValidationSelection

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SHARD_COUNT = 64

# Runs in a child interpreter so PYTHONHASHSEED genuinely differs. Prints
# the full cycle-0 row order (one path per line).
_SNIPPET = """
import sys
from pathlib import Path

sys.path.insert(0, str(Path("__REPO_ROOT__") / "src"))
from sakuramoon.data.manifest import (
    DatasetManifest,
    DatasetSourceIdentity,
    ShardRecord,
)
from sakuramoon.data.service import _QueueStore
from sakuramoon.data.validation import ValidationSelection

paths = [f"data/set-{i:03d}/shard-{i:04d}.tar" for i in range(__SHARD_COUNT__)]
paths.append("data/validation.tar")
paths.sort()
records = tuple(ShardRecord(path=path, bytes=1) for path in paths)
manifest = DatasetManifest.from_shards(
    DatasetSourceIdentity(repo_id="leafmoone/webdataset_danbooru_v2", revision="master"),
    records,
)
validation = ShardRecord(path="data/validation.tar", bytes=1)
selection = ValidationSelection(
    selection_id="validation",
    dataset_id=manifest.dataset_id,
    seed=44,
    shards=(validation,),
)
store = _QueueStore(Path("/nonexistent/mainset.json"), manifest, selection)
state = store.new(0)
print("\\n".join(row.path for row in state.rows))
""".replace("__REPO_ROOT__", str(_REPO_ROOT)).replace("__SHARD_COUNT__", str(_SHARD_COUNT))


def _run_order(seed: str) -> list[str]:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    result = subprocess.run(
        [sys.executable, "-c", _SNIPPET],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, (
        f"child queue-order probe failed (PYTHONHASHSEED={seed}):\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    return result.stdout.splitlines()


def test_cycle_order_bit_identical_across_hash_seeds() -> None:
    orders = {seed: _run_order(seed) for seed in ("1", "2", "12345")}

    for seed, rows in orders.items():
        assert len(rows) == _SHARD_COUNT, (
            f"PYTHONHASHSEED={seed}: expected {_SHARD_COUNT} training rows, "
            f"got {len(rows)}"
        )
        assert len(set(rows)) == _SHARD_COUNT, f"PYTHONHASHSEED={seed}: duplicate rows"

    reference = orders["1"]
    for seed in ("2", "12345"):
        assert orders[seed] == reference, (
            f"cycle-0 queue order differs between PYTHONHASHSEED=1 and "
            f"PYTHONHASHSEED={seed}: first divergence at index "
            f"{next(i for i, (a, b) in enumerate(zip(reference, orders[seed])) if a != b)}"
        )

    # The shuffle must actually permute (guard against a no-op shuffle
    # making this test pass trivially).
    assert reference != sorted(reference), "cycle-0 order equals sorted order: shuffle is a no-op"


def test_same_process_same_cycle_is_stable_and_cycle_dependent(tmp_path: Path) -> None:
    validation = ShardRecord(path="data/validation.tar", bytes=1)
    training = tuple(
        ShardRecord(path=f"data/set-{i:03d}/shard-{i:04d}.tar", bytes=1) for i in range(32)
    )
    manifest = DatasetManifest.from_shards(
        DatasetSourceIdentity(repo_id="leafmoone/webdataset_danbooru_v2", revision="master"),
        tuple(sorted((validation, *training), key=lambda record: record.path)),
    )
    selection = ValidationSelection(
        selection_id="validation",
        dataset_id=manifest.dataset_id,
        seed=44,
        shards=(validation,),
    )
    store = _QueueStore(tmp_path / "mainset.json", manifest, selection)

    zero_first = [row.path for row in store.new(0).rows]
    zero_second = [row.path for row in store.new(0).rows]
    one_first = [row.path for row in store.new(1).rows]

    assert zero_first == zero_second, "same cycle must be stable within a process"
    assert zero_first != one_first, "different cycles must not produce the same order"
