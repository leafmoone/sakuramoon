"""Regression-contract helpers for the Camera Coordinate Causal audit tooling.

These small pure-Python helpers express the exact semantic contracts that the
forensic snapshot scripts in ``final_snapshot/`` depend on. They are a
regression-contract *representation*, not a reimplementation of the audit:

* They must NOT be used to recompute the final scientific numbers.
* They are deliberately torch-free so the contracts can be tested anywhere.
* The snapshot scripts remain the sole authoritative executed code; this file
  exists so the bug classes found while producing the final numbers
  (checkpoint-key collision, variant-cardinality collapse, latent 3-D
  contract, bootstrap subspace, strata masking, margin sign) have permanent,
  executable contracts.

Mapping to final-snapshot code (file:line at snapshot time):
* ledger keying         -> final_snapshot/cc_stage2.py:224    (done[unit][ck_name] = row)
* unit stratum means    -> final_snapshot/cc_stage3.py:139-161
* margin definition     -> final_snapshot/cc_stage3.py:37-40
* band thresholds       -> final_snapshot/cc_stage3.py:291-302
* finite mask + sel     -> final_snapshot/cc_stage3.py:233-241  (v5)
* bootstrap resample    -> final_snapshot/cc_stage3.py:233-261  (v3/v5)
* latent 3-D contract   -> final_snapshot/cc_stage2.py:111-112, 149, 179
* latent 3-D contract   -> final_snapshot/cc_stage2b.py:78, 91
"""
from __future__ import annotations

import re

# ---------------- 10A: ledger keying (checkpoint tag is part of the key) ----


def ledger_key(unit: int, checkpoint: str) -> tuple[int, str]:
    """Stage2 ledger key: (unit, checkpoint).

    The checkpoint tag is part of the key. Matches the final
    ``cc_stage2.py`` storage ``done.setdefault(unit, {})[ck_name] = row``:
    rows are nested under the checkpoint name, never flat per unit, so the
    same sample/timestep/arm under PRE, MID and POST cannot collide.
    """
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("checkpoint tag must be a nonempty string (the key includes it)")
    return (int(unit), checkpoint)


def aggregate_ledger_units(ledger: dict) -> list[tuple[int, str]]:
    """Distinct (unit, checkpoint) entries of a stage2-style ledger.

    A ledger shaped ``{unit: {checkpoint: row}}`` yields one entry per
    (unit, checkpoint) pair: identical sample/timestep/arm content under
    three checkpoints aggregates to THREE separate entries.
    """
    keys: set[tuple[int, str]] = set()
    for unit, per_ckpt in ledger.items():
        for ck in per_ckpt:
            keys.add(ledger_key(unit, ck))
    return sorted(keys)


# ---------------- 10B: base-id vs variant cardinality ----------------------

_VARIANT_RE = re.compile(r"^(?P<base>.+?)-v(?P<n>\d+)_derived$")


def split_base_variant(name: str) -> tuple[str, int | None]:
    """Split a row id into (base_id, variant_index or None).

    Final naming convention: a base id plus optional ``-v1_derived`` /
    ``-v2_derived`` suffixes. ``case-X`` -> ``("case-X", None)``;
    ``case-X-v1_derived`` -> ``("case-X", 1)``.
    """
    m = _VARIANT_RE.match(name)
    if m is None:
        return name, None
    return m.group("base"), int(m.group("n"))


def variant_cardinality(names: list[str]) -> dict:
    """Cardinality report: logical bases vs observed variant rows.

    A base-id filter must never collapse N variant rows to 1: grouping by
    base keeps every observed (base, variant) row distinct.
    """
    bases: set[str] = set()
    variant_keys: set[tuple[str, int | None]] = set()
    for n in names:
        base, v = split_base_variant(n)
        bases.add(base)
        variant_keys.add((base, v))
    return {
        "logical_base_count": len(bases),
        "observed_variant_rows": len(variant_keys),
        "input_rows": len(names),
        "all_rows_preserved": len(variant_keys) == len(set(names)),
    }


# ---------------- 10C: image/noise identity uniqueness ---------------------


def assert_distinct_identity_hashes(hashes: list[str]) -> None:
    """Fail closed if any image/noise identity hash repeats across seed variants.

    Before multi-seed aggregation the identity hash of each seed variant
    must be distinct; a duplicate means two rows are secretly the same
    sample and the aggregate is invalid.
    """
    seen: set[str] = set()
    for h in hashes:
        if h in seen:
            raise ValueError(f"duplicate image/noise identity hash across seed variants: {h[:16]}...")
        seen.add(h)


# ---------------- 10D: latent 3-D contract ----------------------------------


def squeeze_batch(shape: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """The exact transform the final scripts apply to frozen 4-D bundles.

    Frozen unit bundles store ``x0``/``states`` as ``(1, C, H, W)``; the
    final ``cc_stage2.py``/``cc_stage2b.py`` do ``.squeeze(0)`` before
    building the latents tuple. A tensor that is already 3-D passes through.
    """
    s = tuple(shape)
    if len(s) == 4 and s[0] == 1:
        return s[1:]
    return s


def validate_latent_tuple(latent_shapes, in_channels: int) -> None:
    """Pure shape/layout mirror of the PackedDiT latents contract.

    The model contract (src/sakuramoon/model/dit.py, prepare_packed_sequences)
    requires each latents-tuple element to be 3-D ``[C, H, W]`` with
    ``shape[0] == input channels`` on the token device and dtype. The final
    scripts depend on passing squeezed 3-D tensors; a 4-D ``(1, C, H, W)``
    element (the stage2/2b crash) or a transposed layout must be rejected.
    """
    for i, shape in enumerate(latent_shapes):
        s = tuple(shape)
        if len(s) != 3:
            raise ValueError(
                f"latent {i} must be 3-D [C,H,W] on the token device/dtype; got ndim={len(s)} shape={s}"
            )
        if s[0] != in_channels:
            raise ValueError(f"latent {i} channel dim {s[0]} != input channels {in_channels}")
        if s[1] <= 0 or s[2] <= 0:
            raise ValueError(f"latent {i} spatial dims must be positive: {s}")


# ---------------- 10E: bootstrap subspace -----------------------------------


def bootstrap_checkpoint_means(
    margin_rows_by_checkpoint: dict[str, list[float]],
    selected_units: list,
    resample_by_checkpoint: dict[str, list[list[int]]],
) -> dict[str, list[float]]:
    """Paired cluster-bootstrap means with the final v3/v5 subspace semantics.

    ``margin_rows_by_checkpoint``: ``{checkpoint: [per-unit margin]}`` where
    the rows are in ``selected_units`` order (final ``cc_stage3.py`` builds
    ``m = (w - c)[sel_idx]``). ``resample_by_checkpoint``: per-checkpoint
    resample matrices with indices into ``[0, k)`` of the SELECTED subspace
    (the v3 fix: the resample space is the selected subset, not the global
    unit space).

    Contracts:
      * each checkpoint reads ONLY its own rows (PRE never reads MID/POST);
      * indices are bounds-checked against the selected subspace and fail
        closed (the original bug: global-space index 2047 into a 2040-row
        subspace raised an uncontrolled IndexError deep in aggregation);
      * paired structure: identical resample matrices for two checkpoints
        produce identically paired bootstrap replicates (the paired
        POST-PRE delta of the final audit uses the same resample).
    """
    k = len(selected_units)
    if k < 2:
        raise ValueError("bootstrap needs >= 2 selected units")
    out: dict[str, list[float]] = {}
    for ck, rows in margin_rows_by_checkpoint.items():
        if len(rows) != k:
            raise ValueError(f"checkpoint {ck}: {len(rows)} rows != {k} selected units")
        if ck not in resample_by_checkpoint:
            raise ValueError(f"missing resample matrix for checkpoint {ck}")
        vals: list[float] = []
        for rep in resample_by_checkpoint[ck]:
            if len(rep) != k:
                raise ValueError(f"checkpoint {ck}: replicate draws {len(rep)} != k")
            total = 0.0
            for j in rep:
                if j < 0 or j >= k:
                    raise ValueError(
                        f"checkpoint {ck}: resample index {j} outside subspace [0, {k}) (bootstrap-subspace regression)"
                    )
                total += rows[j]
            vals.append(total / k)
        out[ck] = vals
    return out


def per_stratum_rows(
    margins_by_checkpoint_stratum: dict[str, dict[int, list[float]]],
    checkpoint: str,
    stratum: int,
) -> list[float]:
    """Exact per-stratum row view: reads ONLY ``margins[checkpoint][stratum]``.

    Final-code equivalent: the timestep-strata table of ``cc_stage3.py``
    (per-``k`` slicing of the per-stratum loss arrays). A per-stratum view
    must never mix strata; a missing checkpoint/stratum fails closed.
    """
    if checkpoint not in margins_by_checkpoint_stratum:
        raise KeyError(f"unknown checkpoint {checkpoint}")
    strata = margins_by_checkpoint_stratum[checkpoint]
    if stratum not in strata:
        raise KeyError(f"checkpoint {checkpoint} has no stratum {stratum}")
    return list(strata[stratum])


# ---------------- 10F: strata masks ------------------------------------------


def band_of(
    zoom: float, latent_shift: float, orientation: str, norm_offset: float
) -> dict[str, str]:
    """Exact final geometry-band thresholds (``cc_stage3.py:291-302``, v5).

    Boundary semantics follow the final code: ``<`` is strict (e.g.
    ``norm_offset == 0.5`` is the ``high`` band; ``zoom == 1.20`` is
    ``medium``).
    """
    return {
        "zoom": "mild" if zoom < 1.20 else ("medium" if zoom < 1.35 else "strong"),
        "shift": "lt2" if latent_shift < 2.0 else ("2to4" if latent_shift < 4.0 else "ge4"),
        "orientation": "horizontal" if orientation == "horizontal" else "vertical",
        "offset": "low" if norm_offset < 0.5 else "high",
        "edge": "L" if norm_offset < 1 / 3 else ("C" if norm_offset < 2 / 3 else "R"),
    }


def stratum_mask(bands: list[dict[str, str]], stratum_name: str, level: str) -> list[int]:
    """Indices of units whose band equals ``level`` for ``stratum_name``.

    A wrong/missing stratum key fails closed (KeyError) instead of silently
    returning an all-true / empty mask.
    """
    if not bands:
        return []
    if stratum_name not in bands[0]:
        raise KeyError(f"stratum '{stratum_name}' not in band schema (fail closed, never silent all-true)")
    return [i for i, b in enumerate(bands) if b.get(stratum_name) == level]


# ---------------- 10G: causal-margin sign ------------------------------------


def causal_margin(wrong_loss: float, correct_loss: float) -> float:
    """M = Loss(wrong) - Loss(correct) (final ``cc_stage3.py:37-40``).

    Positive = wrong coordinates are penalized (the model reads absolute
    coordinates). Locks the sign convention so a future refactor cannot
    silently invert it.
    """
    return wrong_loss - correct_loss
