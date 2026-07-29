# A002 Independent Review Remediation

## Initial verdicts

- AI/model correctness: **FAIL / changes required**. The first contract lived entirely in a test helper, accepted any `from_pretrained` local-looking argument without provenance from A001, gave broad dataset-path download exceptions, and did not close aliases, `getattr`, `partial`, cache paths, or reference taint flows.
- Infra/performance: **FAIL / changes required**. The first scan skipped itself and could be bypassed through symlinks, composed paths, read-then-exec, process/search-path APIs, and hostile reference-repository Git configuration. Git metadata inspection did not disable fsmonitor, hooks, pager, external diff, interactive filters, and prompts.

These verdicts remain failures until the original independent reviewers rereview this remediation. This file is not review approval.

## Remediation

- Moved policy logic into `tools/asset_execution_boundary.py`, scanned that tool together with every Python file below `src/`, `tests/`, and `tools/`, and made enumeration/read fail closed on out-of-root paths, symlinked files/directories, non-regular files, decode errors, and before/during-read identity drift.
- Added AST import/call provenance and reference taint propagation. Parameterized negative contracts cover loader aliases, `getattr`, `partial`, remote repo/cache/arbitrary paths, missing `local_files_only`, `trust_remote_code`, download aliases, reference static/dynamic imports, composed paths, read-then-exec, `subprocess`/`os`/`asyncio`, `sys.path`, and `site`; positive contracts cover the A001-bound loader and unrelated method/text names.
- Added `VerifiedAssetSelection.verified_root(asset_id)`. A model loader source is accepted only when it comes from `require_runtime_assets_ready` or an exact `sakuramoon.assets.VerifiedAssetSelection` and names `qwen_text_encoder` or `mage_vae`; every locked file and manifest identity is revalidated before returning the root.
- Removed all model-download exceptions. Only `src/sakuramoon/data/modelscope.py::fetch_dataset_shard` may call the ModelScope download API, and only with repo `leafmoone/webdataset_danbooru`, a literal 40-lowercase-hex revision, and `repo_type="dataset"`.
- Restricted reference Git inspection to exact read-only `rev-parse --verify HEAD^{commit}`, `remote get-url origin`, and `status --porcelain=v1 --untracked-files=no`. The subprocess disables system/global config, fsmonitor, hooks, pager, external diff, interactive diff filters, optional locks, terminal prompts, and stdin; hostile local-config tests assert no sentinel executes. Unlisted commands return before subprocess creation.

## Scope and remaining gates

- CPU/static remediation only. It did not read `.env`, model, DB, dataset, cache, or ignored reference contents; did not access the network; and did not use a GPU.
- It proves execution-boundary enforcement, not real Qwen/Mage-VAE load, forward, VAE posterior mean/round-trip, performance, or any four-GPU requirement. Those gates remain assigned to their later tasks.
- Status: remediation complete, pending original independent AI and Infra rereview and main-agent acceptance.

## Second independent review verdicts

- AI/model correctness: **FAIL / changes required**. The first remediation kept global, flow-insensitive root sets, so reassignment could retain stale trust; unknown expanded kwargs and ModelScope loader paths were incomplete. The dataset exception and evidence inventory also needed stricter provenance and clean-commit counts.
- Infra/performance: **FAIL / changes required**. The first remediation could be spoofed by annotations/casts, missed computed loader forms and several reference flows, allowed overly broad test Git shapes, and conflated same-name variables across functions.

These verdicts also remain failures until the original independent reviewers rereview this second remediation. No local status or evidence file represents approval.

## Second remediation

- Replaced global provenance sets with per-module/per-function environments evaluated in statement order. Assignment and reassignment kill a target and its descendants; conservative branch merges retain only valid facts. Function summaries propagate symbolic parameter and return taint across local calls without leaking same-name facts between scopes.
- Added a runtime capability gate, `require_verified_selection`, which accepts only a real `VerifiedAssetSelection` and immediately calls `require_unchanged()`. The static gate trusts only `require_runtime_assets_ready` results or narrow wrapper parameters passed through this exact gate; annotations and casts do not create selection/root provenance.
- Expanded model-loader recognition to Transformers, Diffusers, and ModelScope, including aliases, computed `getattr`, `functools.partial`, and `.from_pretrained.__call__`. `trust_remote_code`, `cache_dir`, non-literal policy values, and unknown expanded kwargs fail closed. Hugging Face download APIs include their qualified module paths and remain forbidden.
- Restricted the dataset exception to the exact `modelscope.hub.snapshot_download.snapshot_download` callable inside `src/sakuramoon/data/modelscope.py::fetch_dataset_shard`, with the exact dataset repo, revision obtained from the locked `load_config().config.data.source.revision` path, and literal `repo_type="dataset"`. Hugging Face lookalikes receive no exception.
- Extended reference taint through shell command strings, attributes, subscripts, list/dict containers, mutator calls, helper returns/parameters, computed process calls, and `sys.path` slice operations. Reassignment and cross-function same-name benign cases demonstrate the intended kill and scope isolation.
- Restricted the test-only Git exception to exact test fixture/function identities and exact allowed argv shapes. Calls containing `-c`, alias setup, unknown argv, or appended commands are rejected; the synthetic repository fixture now configures identity through explicit allowed calls.
- Excluded ignored notebook checkpoint files from scanner inventory so evidence matches a clean tracked checkout: 28 scanned Python sources and 12 tracked production modules. The governance checker may observe an ignored local checkpoint in a dirty developer workspace, but that file is neither scanned nor counted as clean-commit evidence.

## Second-remediation scope and remaining gates

- CPU/static work only. It did not read `.env`, model, DB, dataset, cache, or reference payloads; did not access the network; and did not use a GPU.
- It still proves only enforcement and provenance. Real Qwen/Mage-VAE load, forward, VAE posterior/round-trip, performance, canary, and all multi-GPU gates remain outside A002.
- Status: second remediation complete, pending original independent AI and Infra rereview and main-agent acceptance.

## Third independent review verdicts

- AI/model correctness: **FAIL / changes required**. Callable facts could still be lost through containers/subscripts, helper returns, class attributes, and dynamic attributes; static capability provenance needed to reject forged construction/reflection/subclasses; the test Git `-C` allowlist had to prove its repository argument came from the synthetic temporary root; and D010's final typed Hub facade replaced the obsolete snapshot-download exception.
- Infra/performance: **FAIL / changes required**. The scanner missed forward helper-to-process flows, class methods, lambdas, definition-time expressions, `match`, `assert`, `try`/`except*`, star imports, loops/comprehensions, tuple destructuring, sensitive subclasses, and cross-module/higher-order wrappers. The recommended policy was a bounded, deny-by-default production language rather than another open-ended API denylist.
- The reviewers also demonstrated runtime prerequisites outside A002: subclass overrides, direct constructors, forged nested files, post-issuance field mutation, and cross-selection grafting could bypass a type/identity-only gate. A001 owns the runtime fix and sealed exact capability types plus factory-issued identity and field fingerprints in commit `f7ceb2229874986492f807863b33f803ebec9566`.

These remain failure verdicts until the original independent reviewers rereview the third remediation. The clean local evidence below is implementation evidence, not review approval.

## Third remediation

- Reworked facts and summaries to retain callable, capability-class, selection/root, object/class, literal, tuple/list item, and dict mapping provenance. Fixed-point passes resolve forward declarations and helper returns; branch merges preserve or explicitly mark ambiguous sensitive callables instead of dropping them.
- Covered executable definition and statement positions: function/class decorators, defaults, keyword defaults, annotations, bases, `assert`, `raise`, `delete`, `match` patterns/guards, `try`/`except*` handler expressions, loops, async loops, comprehensions, class methods, lambdas, and tuple/list destructuring. Sensitive star imports fail without guessing their export sets.
- Added deny-by-default production handling for opaque higher-order calls, sensitive callable escape, parameterized process/dynamic/search wrappers, reference/model-root arguments crossing unknown `sakuramoon.*` module boundaries, and unknown external returns entering execution sinks. Assignment kill and precise benign forward/container/match contracts remain green.
- Preserved callable provenance through literal containers and `append`/`extend`/`insert`/`update`, helper return values, class/instance attributes, and literal `getattr`/`setattr`. Dynamic attribute assignment carrying a sensitive callable or reference provenance fails closed.
- Statically sealed `VerifiedAssetFile`/`VerifiedAssetSelection`: only the exact `_inspect_file` and `_selection` factory constructor shapes in `src/sakuramoon/assets/inspect.py` are allowed. Direct construction, subclassing, `object.__new__`, `type.__call__`, dynamic `type`, `object.__setattr__`, `dataclasses.replace`, copying, and computed capability reflection are rejected in production. Test adversarial fixtures remain runtime-testable because this construction/reflection rule is scoped to production sources.
- Replaced the legacy ModelScope `snapshot_download` allowance with D010's exact `ModelScopeDatasetTransport.list_locked_files` and `download_locked_shard_to_staging` methods. Only `self._client.list_repo_files`/`download_file` with the frozen positional and keyword provenance is allowed. `expected_sha256` must be present as literal `None`, leaving the one complete-file hash to project code under its safe staging identity; omitted, non-`None`, or dynamic values fail. Direct `HubApi`, unknown client methods, altered literals, overwritten arguments, or the same call outside the exact class/method/file also fail.
- Required the synthetic Git fixture's `repo`/`-C` argument to preserve parameter-zero temporary-root provenance. The allowlist remains exact by path, function, argv shape, and synthetic identity literals.
- Added one narrow production callable-parameter audit entry for `_IdentityWeakRegistry.contains` dereferencing its local weak reference. The tuple `(file, class, method, local-name)` must match exactly; an equivalent method in another class is rejected.

## Third-remediation clean validation and remaining gates

- The final clean candidate used base `f7ceb2229874986492f807863b33f803ebec9566` and overlaid only `tools/asset_execution_boundary.py` plus `tests/contracts/assets/test_asset_execution_boundary.py`. D001/D010 shared-worktree edits and all A002 documentation edits were excluded from the code candidate.
- Results: 126/126 boundary contracts, 293/293 full tests, 28 Python sources with zero scanner violations, full Ruff pass, full strict Pyright with zero errors/warnings, and traceability checker `ok=true` for 221 requirements/source nodes, 16 archives, 12 production modules, and 235 runtime config keys.
- The first isolated `uv sync --frozen` attempt timed out at 120 seconds while materializing the already locked multi-gigabyte CUDA dependency set; it did not fail a code check. Final commands ran in the isolated source checkout while reusing the main checkout's same locked `.venv`. No model, dataset, database, `.env`, or reference payload was read; no model/data download or GPU execution occurred.
- This remains CPU/static enforcement evidence only. Real Qwen/Mage-VAE loading, forward/posterior/round-trip, WebDataset remote streaming validation, canary, performance, and every multi-GPU gate remain assigned to later tasks.
- Status: third remediation complete, pending original independent AI and Infra rereview and main-agent acceptance.

## Fourth remediation: frozen D010 stdlib HTTPS boundary

- D010 removed the proposed `modelscope_hub`/`HubApi` facade before integration and froze a standard-library HTTPS transport instead. A002 therefore removed every SDK/client-method allowance; importing ModelScope SDKs, alternate HTTP libraries, sockets, logging or patch helpers in the transport is denied.
- The only allowed network constructor is `http.client.HTTPSConnection` in `src/sakuramoon/data/modelscope.py::ModelScopeDatasetTransport._open_get`, with exact `target.host`, `target.port`, configured connect timeout and `ssl.create_default_context()`. Only exact GET/target/body/header/chunked arguments, `getresponse()` and configured socket read timeout are accepted; changed receiver, method alias, positional/keyword argument, `**kwargs`, class, method or file is denied.
- `_ValidatedHttpTarget` construction is restricted to `_listing_target`, `_shard_target` and `_redirect_target` exact shapes. Calls into those factories require verified manifest/shard provenance, and private transport helpers require the returned target capability; overwriting a target with an unverified value kills trust.
- `_request_headers` is locked to fixed Accept, identity encoding and User-Agent fields, with Authorization/Cookie only under `target.send_authorization`. `_open_get` can only add the exact nonnegative resume `Range`; arbitrary mappings, header mutation methods or alternate request header provenance are rejected.
- Response access is limited to `Location`, `Content-Length`, `Content-Encoding`, `Content-Range`, `response.read(length)` and exact cleanup positions. TLS verification mutation, dynamic patching and HTTP diagnostics are denied. Ordinary D010 file-descriptor `os.read/os.close` calls remain distinct and do not create false positives.
- Negative contracts now cover changed method, receiver, request target, args, `**kwargs`, detached method, arbitrary headers, overwritten target, direct target construction, private helper/factory calls, response read/header locations, SDK/alternate network imports, TLS mutation and otherwise identical HTTPS code outside the locked file/class.

## Fourth-remediation clean validation and remaining gates

- The final clean candidate used A001 PASS base `fa435ee72d2d905911ea296c07d1ed3743667a05` and overlaid only `tools/asset_execution_boundary.py` plus `tests/contracts/assets/test_asset_execution_boundary.py`. Concurrent D001/D010 implementation and all A002 evidence edits were excluded.
- Results: 151/151 boundary contracts in 5.17 seconds wall time, 349/349 full tests in 25.10 seconds, 28 Python sources with zero violations in 4.21 seconds, full Ruff in 0.27 seconds, full strict Pyright with zero errors/warnings in 3.79 seconds, and traceability verification in 1.15 seconds for 221 requirements/source nodes, 16 archives, 12 production modules and 235 runtime config keys.
- A separate compatibility scan of the shared-tree frozen D010 HTTP AST reported zero violations. D010 still owns its file-safety, pagination, real remote-streaming and task evidence gates; A002 does not close them.
- No `.env`, model, database, dataset/cache or `reference/` payload was read. No model/data download, GPU work or performance artifact was created. The isolated candidate reused the main checkout's existing locked `.venv`.
- Status: fourth remediation complete, pending independent AI and Infra rereview and main-agent acceptance.

## Fifth remediation: post-`bcf792e` independent review

- Network capabilities now propagate recursively through containers. Passing a verified D010 target, headers, response, or connection into an opaque, cross-module, or higher-order call reports `network_capability_escape_forbidden` and invalidates the capability before later statements are analyzed. This closes bound/unbound `__setitem__`, `dict.__setitem__`, `__ior__`, `operator.setitem`, mapping aliases, custom helpers, `operator.methodcaller`, `operator.attrgetter`, and unknown adapter routes.
- Header writes are accepted only for the exact `_open_get` Range assignment backed by verified `_request_headers`; every other assignment, augmented assignment, or method mutation is rejected. Response/connection names cannot be overwritten by unverified values and then passed into the audited helpers.
- The frozen D010 graph is structurally locked: `_open_get` must construct `HTTPSConnection` in the exact `try` block that raises the redacted initialization error, `_follow_redirects` must clean up through `_close_response`, and `_read_response` length expressions must use the exact bounded-remaining shapes. `_ValidatedHttpTarget` aliases and helper construction remain denied.
- Dynamic `__import__`, dynamic subscript selection from sensitive containers, nested higher-order callable escape, `vars`, `inspect.getattr_static`, and `from sakuramoon.assets import *` are fail closed. `for` and `while` now converge loop-carried facts to a fixed point; branch and exception merges exclude paths that definitely terminate.
- Synthetic Git provenance carries an explicit safe test-root fact and only safe relative `/` operands. Absolute operands and any `..` component, including `../../reference`, are rejected. The callable-parameter exception for the data-manifest identity registry is the same exact `(file, class, method, local-name)` tuple as the assets registry and does not generalize to another class.
- Source reads are rooted with per-component directory descriptors opened using `O_DIRECTORY | O_NOFOLLOW`; leaf stat/open/read/post-stat all remain anchored to the opened parent. Contracts cover a parent replaced by a symlink after lexical precheck and a replacement immediately before leaf open, closing the parent-symlink race.

## Fifth-remediation clean validation and remaining gates

- The detached clean candidate used base `bcf792ef2968f8fb901bc65b1c289c7b8aa57f17` and overlaid only `tools/asset_execution_boundary.py` and `tests/contracts/assets/test_asset_execution_boundary.py`; concurrent D001/D010 and all evidence edits were excluded.
- Results: 191/191 boundary contracts in 7.48 seconds pytest-reported time, 389/389 full tests in 31.79 seconds, 28 Python sources with zero violations, full Ruff pass, strict Pyright with zero errors/warnings, and traceability verification for 221 requirements/source nodes, 16 archives, 12 production modules, and 235 runtime config keys.
- The shared tree containing frozen D010 separately scanned 36 Python sources with zero violations. D010 still owns real remote WebDataset streaming, byte/range/file-safety, and task evidence; the compatibility scan does not close those gates.
- No `.env`, model, database, dataset/cache, or `reference/` payload was read. No model/data download, GPU work, or performance artifact was performed. The candidate reused the main checkout's locked uv environment; an earlier isolated environment materialization was terminated and is conservatively recorded as dependency-network use only.
- Status: fifth remediation complete, pending independent AI and Infra rereview and main-agent acceptance.

## Sixth remediation: call expansion, control flow, reflection, and bounded reads

- `ast.Starred` now evaluates to the full operand fact, and nonliteral keyword expansions are retained as full facts rather than taint-only summaries. Exact and unknown calls, class constructors, and partial bindings recursively inspect expanded list/tuple/mapping/generator children for network capabilities, model roots, asset/dataset capabilities, and sensitive callables.
- Container joins use a finite, idempotent shape-unknown widening. Once list length or mapping keys differ, later iterations cannot oscillate back to an exact shape; security summaries are lifted to the parent fact. `set.pop`, `dict.pop`, and `popitem` preserve extracted loader/capability facts while updating the bounded abstract container state.
- Loop control flow now routes `continue` environments to the next fixed-point iteration and `break` environments to loop exits. Blocks stop after break/continue/return/raise, and loop `else` applies only to the normal exhaustion state. Contracts cover sensitive loader propagation through both break and continue without a convergence violation.
- Production dynamic namespaces and code are denied: no-argument `locals`, `vars`, and `globals`; `eval`, `exec`, `compile`, and other dynamic-code APIs; frame namespace attributes and frame discovery; and operator callable adapters. Reflected D010 private methods and class members are denied. `object.__getattribute__` is limited to exact audited asset/data fingerprint-reader tuples already required by runtime capability sealing.
- Relative assets star imports are rejected. Synthetic Git remains limited by exact fixture/function/argv shape, and `make_reference` now additionally verifies its second argument at each analyzed call site as a safe relative literal without absolute or `..` components.
- The D010 listing read allowance requires a live `bounded_nonnegative` fact for `remaining`. Only the frozen `_LISTING_RESPONSE_LIMIT_BYTES + 1 - len(payload)` assignment issues it; overwrite, augmented assignment, or uncertain branch merge removes it even when the final read expression is textually unchanged.

## Sixth-remediation clean validation and remaining gates

- The detached candidate used base `70981e19510c5a6c7d7889d6042b5e8a55887931` and overlaid only `tools/asset_execution_boundary.py` and `tests/contracts/assets/test_asset_execution_boundary.py`; D001/D010 shared changes and all evidence edits were excluded.
- Results: 232/232 boundary contracts in 19.56 seconds pytest-reported time (20.186 wall), 430/430 full tests in 40.05 seconds (40.693 wall), 28 Python sources with zero violations in 18.682 seconds, full Ruff pass in 0.279 seconds, strict Pyright with zero errors/warnings in 4.244 seconds, and traceability verification in 1.176 seconds for 221 requirements/source nodes, 16 archives, 12 production modules, and 235 runtime config keys.
- The shared tree containing frozen D010 separately scanned 36 Python sources with zero violations in 28.165 seconds. The scanner is a preflight/validation check and does not enter the training forward/backward/update hot path. D010 still owns real remote WebDataset streaming and task-level byte/range/file evidence.
- No `.env`, model, database, dataset/cache, or `reference/` payload was read. No model/data download, GPU work, or performance artifact was performed; the clean candidate reused the existing locked uv environment.
- Status: sixth remediation complete, pending independent AI and Infra rereview and main-agent acceptance.

## Seventh remediation: post-`9081104` independent review

- The frozen `ModelScopeDatasetTransport` class now has a whole-class normalized-AST SHA-256 pin (`226d16422aa57d5fcd8c7b1a05ef4cc07f52296f1d21c29e78c40ef23b1567f9`). Formatting and source locations are excluded, but any structural rewrite fails. The generic provenance analysis remains an independent second layer.
- Namespace recovery is denied through closure vars/cells, generator or coroutine locals, callable defaults/kwdefaults, signatures and execution namespace attributes. `type.__getattribute__`, `functools.reduce`, and `inspect.getattr_static` cannot select production callables. A helper or lambda parameter cannot select `from_pretrained`.
- Nonliteral `getattr` fails closed for local classes, unknown `sakuramoon` modules, builtins, and D010 methods. The only allowances are exact asset-binding comparisons; the only argument-taking `vars` allowances are exact asset/data fingerprint readers. Builtin namespace recovery through `vars(builtins)` remains denied.
- The synthetic Git `make_reference` function is a non-escapable security capability. Passing it through `invoke(fn, *args)` or recovering it through the fixture module's `globals()` is rejected, while exact direct calls still require a proven safe relative literal.
- D010 listing length provenance now proves the payload upper bound. Empty `bytearray()` initialization creates the fact, container mutation kills it, and only the exact terminating oversize guard reissues it on the continuing path. The textual `remaining` expression alone can no longer authorize a negative or otherwise unbounded read.

## Seventh-remediation clean validation and remaining gates

- The detached candidate used base `9081104ba0a87ce72efbeac3125f975b7e3fb71d` and overlaid only `tools/asset_execution_boundary.py` and `tests/contracts/assets/test_asset_execution_boundary.py`; concurrent D001/D010 and all evidence edits were excluded.
- Results: 271/271 boundary contracts in 21.41 seconds pytest-reported time (22.012 wall), 469/469 full tests in 46.55 seconds (47.145 wall), 28 Python sources with zero violations in 20.488 seconds, full Ruff pass in 0.257 seconds, strict Pyright with zero errors/warnings in 3.713 seconds, and traceability verification in 1.075 seconds for 221 requirements/source nodes, 16 archives, 12 production modules, and 235 runtime config keys.
- The shared tree containing frozen D010 separately scanned 36 Python sources with zero violations in 30.096 seconds. This remains static compatibility evidence only; D010 owns real remote WebDataset streaming, byte/range/file-safety, and task evidence.
- No `.env`, model, database, dataset/cache, or `reference/` payload was read. No network access, model/data download, GPU work, or performance artifact was performed. Status remains pending independent AI and Infra rereview and main-agent acceptance.
