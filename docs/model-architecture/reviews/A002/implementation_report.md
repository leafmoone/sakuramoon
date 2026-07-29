# A002 Implementation Report

## 结果

A002 已将用户确认的两条资产执行边界写入现行决定、仓库规则、路线图与资产策略，并通过合同测试建立机器保护。实现没有读取或执行 `reference/` 内容、没有读取 `.env`、没有下载/哈希/加载模型、没有使用 GPU，也没有创建性能占位。

首轮、第二轮、第三轮以及提交 `bcf792ef2968f8fb901bc65b1c289c7b8aa57f17`、`70981e19510c5a6c7d7889d6042b5e8a55887931`、`9081104ba0a87ce72efbeac3125f975b7e3fb71d` 后的独立 AI/Infra 审查均提出 changes required。现已完成 `review_remediation.md` 所列第七轮修复、D010 冻结标准库 HTTPS 接口兼容和 A002-only 隔离验证；状态仍是等待独立 reviewer 复审，而不是已通过。

## 实现摘要

- `model/qwen_3.5_2B` 与 `model/vae` 明确为已准备的唯一 Qwen TE/Mage-VAE 本地目录；缺失或漂移硬失败，禁止自动下载、补下载、联网替换、隐式 cache 命中或 fallback。
- `reference/` 明确只供人工理解/对照且可完全不使用；生产代码、测试、preflight、训练、评估和运行时禁止静态 import、动态加载、`sys.path` 注入、执行、调用或以子进程启动其中代码。
- 独立 scanner 扫描 `src/`、`tests/`、`tools/` 及其自身，并忽略版本控制已忽略的 notebook checkpoint；读取前拒绝越界、symlink、非普通文件和身份漂移。AST provenance/taint 在模块、函数和类作用域内按语句顺序传播，赋值会 kill 旧 provenance；固定点摘要处理前向 helper、lambda 与 method，事实穿透 tuple/list/dict、subscript、class attribute、literal `getattr`/`setattr`、循环、推导与 pattern binding。
- `from_pretrained` 覆盖 Transformers、Diffusers 与 ModelScope，以及 alias、computed `getattr`、`partial` 和 `.__call__`。来源必须是 `require_runtime_assets_ready` 返回的 capability，或由 `require_verified_selection` 验证的窄 wrapper 参数，再取 Qwen/Mage-VAE 的 `verified_root()`；annotation/cast 不构成证明。调用必须显式 `local_files_only=True`，拒绝 `trust_remote_code`、`cache_dir` 和来源未知的 `**kwargs`。
- 任何模型下载 API 都禁止，旧 `snapshot_download`、SDK/HubApi dataset 例外也已删除。唯一 dataset transport 是 `src/sakuramoon/data/modelscope.py::ModelScopeDatasetTransport` 的固定标准库 HTTPS 图：target 只能来自三个审计 factory，入口 manifest/shard 必须经验证；连接只能使用默认 TLS context、固定 host/port/timeout；请求只能是 GET、锁定 request target、空 body、审计 headers、`encode_chunked=False`。header/Range、redirect、response header/read/close 均限制在精确函数和参数位置，changed receiver/method alias/参数/`**kwargs`/位置无例外。
- Reference taint 覆盖 shell 字符串、路径组合、文件内容、属性/容器、函数参数/返回值、动态 `setattr`、动态 loader、进程、跨模块 wrapper 与 search-path API（包括 `sys.path` slice）。测试 Git 例外仅接受精确测试文件/函数及固定命令 shape，拒绝 `-c`、alias、未知 argv 或附加命令；`git -C` 的 repo 参数还必须证明来自 synthetic `tmp_path` 根。
- Production 分析采用受限语言策略：function decorators/defaults/annotations、`assert`/`raise`/`delete`、`match`、`try`/`except*`、class bases/decorators、star imports、tuple destructuring、循环与推导都被求值；不可证明的高阶 callable、敏感 callable escape、parameterized execution wrapper、未知跨模块返回进入执行 sink 均硬失败。
- Static capability sealing 只允许 `src/sakuramoon/assets/inspect.py::_inspect_file` 与 `_selection` 的精确构造 shape，并拒绝直接构造、subclass、`object.__new__`、动态 `type`、`object.__setattr__`、`dataclasses.replace` 和 computed reflection。真正的运行时 exact-type、identity issuance 与字段 fingerprint gate 归属 A001 commit `fa435ee72d2d905911ea296c07d1ed3743667a05`，A002 不把它重复记为自身实现。
- 参考 Git metadata helper 只允许三个命令，并用命令行配置、隔离环境、关闭 stdin 的组合阻断 local/system/global fsmonitor、hook、pager、external diff、interactive filter 与 prompt。
- 追踪将两条要求拆为独立 profile：`local_model_execution_boundary` 映射模型配置与 asset/encoder 模块；`reference_execution_boundary` 明确 config N/A，并覆盖现在及未来全部 `src/sakuramoon/**` 生产模块。
- A002 初始实现将 confirmed source revision 从 1 递增到 2、registry 从 5 递增到 6；后续治理提交已把 registry 递增到 11。稳定 ID 仍为 `DOC-006` 与 `DOC-007`。第三轮没有新增/移动 requirement，也没有改变 mapping/path/status，因此不为纯证据改写 registry revision、source fingerprint 或 SHA-256 changelog；`current/` 与 archive 均未修改。

## AI / 模型正确性自检

- manifest 中两条 ready model 的逻辑路径与用户确认值精确一致，合同没有读取模型 payload。
- 本任务只证明资产路径/执行策略和静态调用边界，不把它表述为 Qwen load、Mage-VAE posterior mean、round-trip、forward 或质量证据；这些门槛仍属于 T020/T021。
- Dataset 的 ModelScope 网络流式职责仍隔离在 D010 标准库 HTTPS transport；A002 只验证精确静态调用边界，该例外不授权任何模型下载，也不等于 D010 已完成远端数据验证。

## Infra / 性能自检

- Scanner 只在测试/验证阶段解析仓库 Python 文本；固定点上限随 AST 中函数/class/lambda 数量线性设定，在最终 clean candidate 扫描 28 个 Python sources 用时 4.21 秒。`require_verified_selection`/`verified_root` 只在模型构造前重验已锁文件身份，不进入训练 step 热路径。
- 检查不访问网络、GPU、大模型、数据库、ignored reference 内容或 `.env`。
- 本地模型不合约时维持 fail-closed，不新增网络恢复、远端替换或慢路径 fallback。

## 独立审查重点

- AI reviewer：复验 capability 工厂/identity 边界、容器/subscript/helper/class/dynamic-attribute callable provenance、`git -C` synthetic temp-root provenance，以及 D010 target/manifest/header 的端到端 provenance。
- Infra reviewer：复验 forward helper、class/lambda、definition expressions、`match`/`assert`/`except*`、star import、循环/推导、tuple destructuring、跨模块/高阶 wrapper、精确 weakref 例外和 D010 changed receiver/args/kwargs/location 拒绝矩阵；确认 scanner 保持 fail closed 且不进入训练热路径。

## 第五轮修复摘要

- `_Fact` 递归保存并检查敏感 callable 与 D010 target/header/response/connection capability。任何网络 capability 进入 opaque、跨模块或高阶调用都报告 escape，并从环境和嵌套容器中失效；因此失败调用后的 request/read/close 不会沿用旧信任。
- 对 assignment、augmented assignment、bound/unbound mutation、operator helper、mapping alias 与自定义 helper 统一执行 header mutation 拒绝；对 `operator.methodcaller`、`operator.attrgetter`、未知 receiver/method 与网络 binding 覆盖统一 fail closed。
- D010 exact graph 与冻结实现一致：HTTPS constructor 必须位于 redacted-error `try` 中，redirect cleanup 只走 `_close_response`，response reads 使用冻结 bounded length，target construction 只允许三个直接 factory。changed wrapper/helper/alias/location 均有负合同。
- 动态 import、动态/嵌套容器下标、`vars`、`inspect.getattr_static`、assets star import 与 callable escape 已封闭；`for`/`while` 采用循环携带事实 fixed point，控制流 merge 排除确定终止的分支。
- Git 测试例外验证 synthetic root path 的安全相对组合并拒绝绝对路径和 `..`；data manifest 的 weakref 解引用只增加与 assets inspect 同等精确的 tuple 审计项。
- Scanner 文件读取使用逐组件 no-follow directory descriptors 锚定仓库根，leaf 的 stat/open/read/post-stat 均绑定同一父目录 descriptor；合同模拟预检后父目录替换和 leaf open 前替换，均硬失败或继续读取已锚定的安全 inode。

## 第五轮验证结论

- Clean candidate：base `bcf792ef2968f8fb901bc65b1c289c7b8aa57f17`，只叠加 `tools/asset_execution_boundary.py` 与 `tests/contracts/assets/test_asset_execution_boundary.py`。
- 结果：191/191 boundary contracts、389/389 full tests、28 Python sources/0 violations、全仓 Ruff PASS、strict Pyright 0 errors/0 warnings、traceability 221 requirements/source nodes PASS。
- Shared compatibility：并行 D010 冻结实现所在共享树扫描 36 Python sources、0 violations；这不是 D010 远端流式或任务放行证据。
- 本轮没有读取 `.env`、模型、数据库、dataset/cache 或 `reference/` payload，没有模型/数据下载、GPU 工作或性能 artifact。状态保持第五轮修复完成、等待独立 AI/Infra 复审。

## 第六轮修复摘要

- Call expansion 不再把 `ast.Starred` 或 nonliteral `**` 降为 taint-only。完整 container children 进入 network、model-root、capability 与 sensitive-callable 递归检查；known/unknown calls、class construction 和 `partial` binding 使用同一 security fact 集合。
- Branch/loop join 对不同 container shape 做不可逆 widening，并在容器顶层保存 network、model-root、sensitive callable、asset capability 与 dataset capability summary，保证 join 幂等且不会静默丢失历史安全事实。`pop`/`popitem` 返回候选 value fact，并同步有限 mutation state。
- Loop analysis 分开收集 break exit 与 continue back-edge；block 在确定终止语句后停止，normal exhaustion 才执行 loop `else`。敏感 loader 的 break/continue 负合同同时验证 fact 传播和 fixed-point convergence。
- Production restricted language 新增 dynamic namespace/code/frame 与 callable reflection deny。无参 locals/vars/globals、eval/exec/compile、frame/f_locals、operator adapters、D010 reflected private helpers、非审计 object member extraction 全部失败；runtime capability fingerprint 的既有 object access 由精确 path/function/receiver/attribute tuple 审计。
- Relative assets star import 纳入敏感 import；synthetic Git helper call-site 要求第二参数为 proven safe relative literal。D010 listing `remaining` 获得独立 bounded-nonnegative fact，精确表达式只负责签发，重赋值与 merge 会移除信任。

## 第六轮验证结论

- Clean candidate：base `70981e19510c5a6c7d7889d6042b5e8a55887931`，只叠加 `tools/asset_execution_boundary.py` 与 `tests/contracts/assets/test_asset_execution_boundary.py`。
- 结果：232/232 boundary contracts（19.56 秒 pytest、20.186 秒 wall）、430/430 full tests（40.05/40.693 秒）、28 Python sources/0 violations（18.682 秒）、Ruff PASS（0.279 秒）、strict Pyright 0 errors/0 warnings（4.244 秒）、traceability 221 requirements/source nodes PASS（1.176 秒）。
- Shared compatibility：冻结 D010 所在共享树扫描 36 Python sources、0 violations，用时 28.165 秒。Scanner 只在 preflight/验证阶段运行，不进入训练 step 热路径；该结果不是 D010 远端 WebDataset 流式证据。
- 本轮没有读取 `.env`、模型、数据库、dataset/cache 或 `reference/` payload，没有模型/数据下载、GPU 工作或性能 artifact。状态保持第六轮修复完成、等待独立 AI/Infra 复审。

## 第七轮修复摘要

- `src/sakuramoon/data/modelscope.py::ModelScopeDatasetTransport` 的完整 normalized AST 被固定 SHA-256 锁定；任意方法、closure、default、descriptor 或控制流结构变化都会先触发结构违规。该 pin 忽略格式/source location，同时保留全部通用 provenance 规则作为第二层防护。
- Production namespace/reflection 限制覆盖 closure vars、generator/coroutine locals、callable defaults/kwdefaults/signature parameters、closure cell、`type.__getattribute__`、`functools.reduce` 与 `inspect.getattr_static`。参数化 helper/lambda 选择 `from_pretrained` 和所有非 literal `getattr` 默认失败。
- `vars` 与动态 `getattr` 仅给当前 fingerprint reader 和 asset binding 比较的精确调用 tuple 放行；`vars(builtins)["eval"]`、`getattr(builtins, input())`、未知本地/`sakuramoon` callable 与 D010 dynamic method 均被负合同覆盖。
- Synthetic Git `make_reference` 作为专用 security capability 传播，不能进入 `invoke(fn, *args)` 等高阶调用；fixture 中 `globals()` 等 execution namespace 恢复也被拒绝。直接安全相对路径调用保持正例。
- Listing payload 新增 upper-bound fact：空 `bytearray()` 初始有界，任何 append/extend/insert/update mutation 清除该 fact；只有精确 `len(payload) > limit` 且超限分支确定终止时，继续分支重新获得 bound。`remaining` 必须同时满足精确算式和 payload bound 才能成为 nonnegative read length。

## 第七轮验证结论

- Clean candidate：base `9081104ba0a87ce72efbeac3125f975b7e3fb71d`，只叠加 `tools/asset_execution_boundary.py` 与 `tests/contracts/assets/test_asset_execution_boundary.py`；并行 D001/D010 与全部 A002 evidence edits 均排除。
- 结果：271/271 boundary contracts（21.41 秒 pytest、22.012 秒 wall）、469/469 full tests（46.55/47.145 秒）、28 Python sources/0 violations（20.488 秒）、Ruff PASS（0.257 秒）、strict Pyright 0 errors/0 warnings（3.713 秒）、traceability 221 requirements/source nodes PASS（1.075 秒）。
- Shared compatibility：冻结 D010 所在共享树扫描 36 Python sources、0 violations，用时 30.096 秒。Scanner 只在 preflight/验证阶段运行，不进入 forward/backward/update；该结果不是 D010 远端 WebDataset 流式证据。
- 本轮没有读取 `.env`、模型、数据库、dataset/cache 或 `reference/` payload，没有联网、模型/数据下载、GPU 工作或性能 artifact。状态保持第七轮修复完成、等待独立 AI/Infra 复审。
