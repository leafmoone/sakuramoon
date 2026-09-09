# Camera v2：位移方视口（shifted-square）数据策略

纯数据几何策略：不改模型、loss、优化器、LR、batch、checkpoint 格式，也不改
采样 / 评估入口（本轮 `src/sakuramoon/eval/` 与采样链路保持冻结，仅在既有
`eval/runtime.py` 的 `coordinate_map` 接线上传递相机裁剪坐标）。

## 几何（crop-after-scale）

选中相机路径的样本**直接按 post-EXIF 源尺寸规划**，不再先调用普通
`prepare_image` / `assign_bucket`（普通 admission 完全不参与）：

1. `R` = `train.resolution` 对应的唯一方形 stage bucket（构造期 fail-fast）；
2. 短边按比例缩放到 `R`（LANCZOS），长边按 round-half-up 量化；
3. 在量化后的长轴上均匀抽取一个**含端点的整数偏移**；
4. 输出 = 完整画布上 `R × R` 的裁剪（先缩放后裁剪，不是原图直接取 `R×R` 块）；
5. 方形源 = 恒等视图（zoom=1、偏移 0），近方形与量化后零位移范围均合法。

`equivalent_zoom` / `retention` 仅为描述量（zoom = √(画布面积/R²)，可为任意
≥1 的有限值；retention = R²/画布面积）。**没有** zoom 窗口、长宽比上限或
固定观测 bin；不存在"回退普通 bucket"的相机分支。

## 激活语义（单一条件）

`camera_active = 配置表存在 ∧ enabled ∧ probability > 0.0`。

- `probability` 为 [0,1] 有限数；`p=0` 合法即关；`enabled=false` 恒关（任意 p）。
- 有效关闭时：不建相机 planner、不消费相机 RNG、样本与 dev 普通路径
  bit-identical（含 spatial 启用的配置）。
- 相机与 spatial 的互斥按 **camera_active** 判定：`p=0` 或 `enabled=false`
  不阻挡 spatial。

## no-upscale 拒绝

选中但 post-EXIF 短边 < `R` 的样本被**明确拒绝**（rejection reason
`camera_no_upscale`）：不放大、不回退普通 bucket。JPEG draft 解码若某轴低于
计划画布，会对该样本从原始压缩数据全量重解码，不得以 draft 小图冒充；
draft 优化本身不关闭。

## 一个源样本 = 一个训练视图

`physical_views == logical_samples`：每样本恰好一个输出视图（普通路径或相机
视图，二者互斥）。相机视图的 audit/坐标全部来自实际相机裁剪；训练全画布坐标
走既有 `full_canvas_crop_coordinates`（latent 网格 = 画布尺寸/16），不二次套
仿射，不加新 token/位置分支。

## 配置

- `config/train_g1_camera_v2_p100.toml`：P100（低分辨率 shifted-square 目标配置，
  继承 train_g1 的 resolution=256 与全部参数）。
- `config/train_g1_camera_v2_p25.toml`：同几何，p=0.25。
- 无相机表 = 完全关闭；两者均未部署，需显式 GO 后从完整生产 checkpoint 经
  `scripts/training_stack.sh` 启动。

本轮不改：采样 CLI / prompt bank / tokenizer / 采样参数 / MBS 归档 /
`service.py` / `training_stack.sh`（队列确定性、warmup-then-bind、
`DATA_READY_TIMEOUT_SECONDS=900` 两项通用修复原样保留）。
