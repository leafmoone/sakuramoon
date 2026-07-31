# R001 AI/model review

Status: the original independent AI review passed commit
`664fda71faed5e5d7d26d5fd06754af1a20b721f`. The later evidence-only remediation
passed main-agent acceptance; no fresh independent rereview is claimed.

The repository entrypoint and agent policy establish the current documents as the only
normative architecture source and explicitly prevent archived recommendations from
becoming implementation or configuration. Training semantics remain TOML-owned,
`all_condition=0.10` is the only fixed dropout probability, and every other dropout
value remains unresolved. Single-GPU evidence is explicitly barred from closing any
four-GPU gate.

R001 does not implement model, data, optimizer or training behavior. Its reference
metadata is historical source/license evidence only and is not imported or executed by
production code or tests. The later removal of ordinary-task performance placeholders
and correction of immutable-commit test counts did not change any model or training
contract.

Main-agent verification found no AI/model correctness issue in the current R001
boundary. Two direct attempts to start a fresh independent reviewer failed with
`agent thread limit reached`; per user direction work continued without agents.
