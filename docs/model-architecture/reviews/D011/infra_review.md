# D011 Infra/performance review

Status: validation bundle expansion passed main-agent Infra review after one durability
finding was remediated. Fresh independent rereview is unavailable after two direct
agent-start failures.

The added bucket-key check is constant-time per metadata row and does not change the
in-memory selection algorithm's asymptotic cost. D011 still performs no network, disk
index, database, model, or GPU work.

The approximately 11M-row production scan and its memory/time evidence remain pending;
synthetic tests do not close that gate. No DDP/NCCL, multi-GPU work, training long run,
or performance placeholder was used. Final acceptance is by the main agent because
direct independent re-review startup was unavailable.

Main-agent review found that the temporary directory was renamed before parent fsync,
but an fsync failure left the final visible while returning an error. Publication now
tracks rename completion, removes only `validation_manifest.jsonl`, `validation.tar`
and their task-owned directory, then best-effort fsyncs the parent. The fourth-fsync
injection contract leaves neither final nor temporary.

The manifest is fixed at 2,000 lines; tar writing keeps one caller-provided sample in
memory at a time. Selection itself intentionally remains an in-memory tuple/dict scan;
its approximately 11M-row memory/time measurement is still required before production.
No database, network, GPU or training hot-path work was added. No-clobber remains a
single-writer operational boundary and is not an adversarial filesystem claim.

The 32 targeted CPU contracts passed in 1.22 seconds; Ruff and strict Pyright passed.
Two direct fresh-review starts failed with `agent thread limit reached`; this is a
main-agent remediation conclusion, not an independent PASS.
