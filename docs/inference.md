# 独立推理

入口：`python -m sakuramoon.cli.infer`。它读取本地模型，不访问数据服务、W&B 或模型下载服务，也不读取 `.env`。GPU 进程在命令结束后退出。

从仓库运行时也可以用 `python scripts/infer.py`，这个薄包装会自动加入源码搜索路径；所有参数与模块入口相同。

## 准备文件

```text
runtime/
  model/
    qwen_3.5_2B/
      config.json
      tokenizer.json
      tokenizer_config.json       如原资产包含，保留
      model.safetensors
      ...                        保留原资产的其他 tokenizer 文件
    vae/
      config.json
      diffusion_pytorch_model.safetensors
  output_model/
    selected-model/
      resolved_config.toml       可选；缺少时用 --config
      train_state/
        growth_state.json        增长期需要，或指定 --growth-alpha
      model/
        config.json
        manifest.json
        model.safetensors.index.json
        model-00001-of-00002.safetensors
        model-00002-of-00002.safetensors
```

模型目录的文件集合、分片大小、tensor 名称/形状/dtype 按现有加载器严格校验；不要改文件名或把多个 checkpoint 的分片混在一起。只复制 `model/` 即可加载权重，不需要复制 `optimizer.pt`。需要继续训练时则必须保留完整 raw checkpoint，包括优化器、RNG 和所有训练 sidecar。

资产路径相对于 `--root`；`--checkpoint`、`--config` 和 `--output-dir` 相对当前工作目录解释。加载器不自动替换或下载冻结编码器。

## 一张图

在仓库根目录、已准备好的环境中：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m sakuramoon.cli.infer \
  --root /path/to/runtime \
  --checkpoint /path/to/runtime/output_model/selected-model \
  --prompt "A small white cat under cherry blossoms, anime illustration, daylight." \
  --width 512 --height 512 --seed 42 \
  --profile balanced --attention-backend sdpa \
  --output-dir outputs/cat
```

输出为 `seed-42-512x512.png` 与同名 JSON。PNG 中也嵌入相同 JSON，记录 checkpoint ID/update、提示词、条件词、尺寸、种子、CFG、noise scale、求解器、步数/NFE、growth alpha、后端及 torch/CUDA/HIP 版本。

命令拒绝覆盖已有输出。重复运行时更换输出目录或种子。

## 使用独立模型目录

```bash
python -m sakuramoon.cli.infer \
  --root /path/to/runtime \
  --checkpoint /path/to/exported/model \
  --config config/inference.toml \
  --prompt-file prompt.txt \
  --output-dir outputs/model-only
```

`prompt.txt` 是 UTF-8 纯文本。`--prompt` 与 `--prompt-file` 二选一。

当 checkpoint 有 `resolved_config.toml` 时优先使用它，尤其不能误改训练时的 `noise_scale` 或 `t_eps`。`--config` 可以指向完整的**已解析**训练配置，也可以指向独立推理配置；它不接受尚未展开的 `extends` 配置链。

## 参数

| 参数 | 默认 / 说明 |
|---|---|
| `--attention-backend` | `sdpa`；可显式选 `flash`，要求兼容的 FlashAttention-2 |
| `--device` | `cuda:0`；DCU 同样使用此命名 |
| `--width`、`--height` | 512；正整数且为 16 的倍数 |
| `--seed` | 42；范围 `[0, 2**63)` |
| `--num-images` | 1；串行生成，种子依次加 1，避免随图片数增长显存占用 |
| `--profile` | 来自配置中的 `sampling.profile` |
| `--steps` | 可选，覆盖 profile 的步数；Heun 的网络调用次数不是步数 |
| `--guidance-scale` | 可选，覆盖配置 CFG；有限非负数 |
| `--growth-alpha` | 默认读取 sidecar；无新增 slot 时可用 1.0，增长期不能猜测 |
| `--style` | 可选，逗号分隔的风格条件词 |
| `--character` | 可选，逗号分隔的角色身份条件词；不能和 `--style` 同用 |

独立推理用自然语言构造主 caption；风格/角色条件进入模型训练过的专门条件段，而不是拼接到隐藏的系统提示词中。超出训练 caption token 预算时会明确报错，不静默截掉提示词。当前 CLI 不提供 negative prompt；CFG 的无条件分支沿用训练时的空条件/null-token 约定。

## 后端和资源

`sdpa` 逐样本执行 PyTorch GQA，适合先在 NVIDIA 上跑通，也可供有空闲资源的 HIP 环境使用。它不会重新保存或转换 checkpoint。DCU 原生产路径使用 `flash`；没有 DAS/FlashAttention-2 时，该模式明确报错。

Qwen 的 FLA 加速是可选的推理性能优化。没有 FLA 时，文本塔使用 Transformers 的 PyTorch 路径，卷积兼容模块也使用 PyTorch。**生产训练**仍保留原有 fast-path preflight，不把慢速 fallback 当作已验证的训练环境。

该入口已在一张 RTX 5090 D 上用实际 20 层 checkpoint 完成 512×512 生成。未测量最小显存需求，不能把测试卡的 32 GB 当成最低要求。大分辨率会增加 token 数与激活内存；先单图测试，再增加分辨率。

相同 seed 只保证同一执行条件下的随机数输入约定；CUDA 与 DCU、不同 torch 或不同注意力实现不保证逐像素相同。图片质量需要独立的多提示词/多种子评估，单张示例不等价于 FID/IS 改善。
