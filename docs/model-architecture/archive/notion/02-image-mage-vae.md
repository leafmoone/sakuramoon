Here is the result of "view" for the Page with URL https://app.notion.com/p/3aaae967ecf2816096f0ea37634a2f7e as of 2026-07-27T08:45:56.966Z:
<page url="https://app.notion.com/p/3aaae967ecf2816096f0ea37634a2f7e" icon="🖼️">
<ancestor-path>
<parent-data-source url="collection://69ca66ff-43e7-4128-bb7b-9f3751506705" name="组件决策记录"/>
<ancestor-2-database url="https://app.notion.com/p/250be554eacc40219065073dfcf66fd7" title="组件决策记录"/>
<ancestor-3-page url="https://app.notion.com/p/3aaae967ecf281c8a10bf797b47982ff" title="新模型架构决策中心"/>
</ancestor-path>
<properties>
{"date:决定日期:is_datetime":0,"date:决定日期:start":"2026-07-27","url":"https://app.notion.com/p/3aaae967ecf2816096f0ea37634a2f7e","决策编号":"ARCH-2","序号":2,"影响":"高","标签":["架构","训练","数据","系统"],"状态":"已接受","组件决策":"02 图像表示与 Mage-VAE"}
</properties>
<content>
<callout icon="✅" color="green_bg">
	**决定：**直接使用并冻结 Microsoft 官方 Mage-VAE，以 posterior mean 在线编码动态 resize/crop 后的图像。使用原生 `128ch @ H/16` latent，不做额外 patchify，不预编码、不放大小图；二次元重建必须通过固定验证集门槛。
</callout>
# 背景
本组件承接 [01 约束、预算与验收标准](https://app.notion.com/p/3aaae967ecf281ba8f73fac2f9e4c4f3)。数据为约 11M 二次元 WebDataset，经网络存储读取；训练采用阶段式多分辨率，最低成品分辨率为 512。VAE 必须兼顾二次元重建质量、在线吞吐、动态裁剪和后续 JLT clean-latent x-pred 接口。
# 已接受决定
<table fit-page-width="true" header-row="true">
<tr>
<td>项目</td>
<td>决定</td>
</tr>
<tr>
<td>VAE</td>
<td>Microsoft 官方 Mage-VAE 权重与官方实现，训练期间冻结</td>
</tr>
<tr>
<td>VAE 属性</td>
<td>视为通用图像 VAE，不假定具备二次元专项优势</td>
</tr>
<tr>
<td>Latent</td>
<td>原生 `128 channels @ H/16 × W/16`</td>
</tr>
<tr>
<td>Posterior</td>
<td>训练和验证均取 posterior mean，`sample_posterior=False`</td>
</tr>
<tr>
<td>Patchify</td>
<td>DiT `patch_size=1`，不做额外 `2×2 patchify`</td>
</tr>
<tr>
<td>预编码</td>
<td>初始训练不预编码；动态裁剪后在线编码</td>
</tr>
<tr>
<td>放大</td>
<td>不放大小图；不能原生覆盖当前 bucket 的样本跳过该阶段</td>
</tr>
<tr>
<td>裁剪坐标</td>
<td>裁剪偏移不作为模型条件；每个 crop 被视为完整目标画面</td>
</tr>
</table>
# 图像到 latent 的接口
```plain text
WebDataset image
→ EXIF orientation
→ RGB
→ 当前阶段最近宽高比 bucket
→ 保持比例缩放，不放大
→ 可复现随机裁剪
→ [-1, 1]
→ frozen official Mage-VAE encoder
→ posterior mean
→ [B, 128, H/16, W/16]
→ flatten H/16 × W/16
→ 128 → hidden_size input projection
```
512 输入对应 `32×32=1024` 个 image token；1024 输入对应 `64×64=4096` 个 image token。
# Resize 与裁剪协议
- 修正 EXIF 后统一转为 RGB。
- 将样本分配到当前阶段最接近原图比例的 bucket；bucket 高宽必须为 16 的倍数。
- 不改变宽高比，不拉伸成正方形。
- 仅缩小，不放大。源图不能覆盖目标 bucket 时，从当前阶段排除。
- 大比例缩小时先使用分阶段 BOX 缩小，最后使用 bicubic 精确调整；不使用可能在线条边缘产生振铃的 Lanczos。
- 缩放后裁剪到 bucket，裁剪面积应尽量控制在原图约 15% 以内；超过时改分配 bucket 或跳过。
- 训练裁剪偏移由 `global_seed + epoch + sample_key` 派生，确保断点恢复可复现，同时允许同一图像跨 epoch 获得不同 crop。
- 验证集保存固定 bucket、缩放比例和裁剪坐标。
- 暂不做水平翻转；角色不对称、文字和左右方向 tags 的处理留到组件 11。
# Posterior 与 JLT 对齐
JLT 将 clean latent `x` 视为固定编码器给出的确定性 endpoint，参考实现通过 `latent_dist.mode()` 取得 posterior mean。因此本项目使用：
```python
vae = MageVAE(ckpt_path=vae_path, sample_posterior=False)
with torch.no_grad():
    clean_latent = vae.encode(processed_image)
```
训练随机性来自 crop、timestep、Gaussian noise、文本/tag dropout 和数据顺序，不额外加入 posterior sampling。x-pred 与采样的具体实现以后续组件 08 为准，并以固定提交的 JLT 参考项目为依据。
# 在线编码与预编码
初始路径为每卡加载冻结的 Mage-VAE encoder：
- 训练进程不加载 decoder，降低常驻显存。
- 同一 bucket、相同形状的图像合并编码。
- 允许对 encoder 的确定性部分使用 `torch.compile`。
- 吞吐测量必须包含网络读取、图像解码、resize/crop 和 VAE encode。
不默认预编码的原因：
- WebDataset 当前不含 latent。
- 多分辨率阶段和动态 crop 会要求多个分辨率或裁剪版本。
- 512 BF16 latent 约 256 KiB/样本，11M 样本约 2.88 TB；256、512、768 各一份合计约 10.1 TB，尚未包含冗余和其他增强版本。
- 网络存储可能把 VAE 计算瓶颈转换为更严重的 latent I/O 瓶颈。
只有完整路径 benchmark 证明有明确净收益时，才允许为特定阶段建立临时 latent cache。
# 二次元重建验证集
固定 2,000 张、冻结后不按结果重选：
- 1,600 张采用分层随机抽取，覆盖主要分辨率、宽高比和 tags 分布。
- 400 张采用风险覆盖抽取，重点覆盖人物近景、全身与手部、多人物、复杂服装、细线稿、纯色块、渐变、透明特效、文字和复杂背景。
- 固定随机种子，保存 `shard URL + sample key`。
- 去除损坏和重复样本，同时保存原图及所有 resize/crop 参数。
- 1,500 张按真实训练路径处理到 512 等效面积；500 张覆盖 768/1024 与极端宽高比。
# 通过门槛
<table fit-page-width="true" header-row="true">
<tr>
<td>指标</td>
<td>门槛</td>
</tr>
<tr>
<td>Median LPIPS</td>
<td>≤ 0.03</td>
</tr>
<tr>
<td>P95 LPIPS</td>
<td>≤ 0.08</td>
</tr>
<tr>
<td>Median SSIM</td>
<td>≥ 0.94</td>
</tr>
<tr>
<td>严重重建错误率</td>
<td>&lt; 1%</td>
</tr>
<tr>
<td>人工检查明显细节损失率</td>
<td>&lt; 5%</td>
</tr>
</table>
PSNR 仅记录，不作为硬门槛。人工检查重点为线宽变化、眼睛高光、手指粘连、发丝消失、色块溢出和纹理糊化。
# 位置接口约束
接受宽高比归一化 2D axial RoPE 的方向，但精确实现归组件 05：
```plain text
r_y = sqrt(H_latent / W_latent)
r_x = sqrt(W_latent / H_latent)
y ∈ [-r_y, r_y]
x ∈ [-r_x, r_x]
```
- position map 根据 crop 后的 latent H/W 生成；不编码原图 crop offset。
- 不使用把 crop 视为隐藏大画布窗口的 shifted-square offset map。
- 后续应显式提供 `log(W/H)` 和 `log(H×W)` 或等价的比例、尺度条件。
- 为兼容 GQA，RoPE 频率必须让同一 KV group 的 Q/K 保持一致；当前倾向所有 heads 使用固定共享频率。
- bucket 级 position map 与 `cos/sin` 可以缓存，预计不会显著影响训练吞吐。
- RoPE 旋转维度、频率、文本坐标和单流图文交互留到组件 05。
# 备选与否决
- **Posterior sample：**不采用；会让 JLT clean endpoint 随读取变化。
- **初始全量预编码：**不采用；存储、动态 crop 和网络 I/O 成本过高。
- **放大小图进入高分辨率阶段：**不采用；避免把插值伪细节作为高分辨率监督。
- **额外 latent patchify：**不采用；Mage-VAE 已直接输出 Transformer-ready H/16 latent。
- **保留原图 crop offset 的位置图：**不采用；会引入隐藏画布和“被裁出来”的构图先验。
- **固定正方形拉伸：**不采用；破坏几何比例。
# 后果
**正面：**
- 保留多分辨率阶段的动态 crop 数据增强。
- 与 JLT 的确定性 clean-latent 定义一致。
- 避免多 TB latent 缓存及网络存储放大。
- H/16 原生网格将 512 image token 控制在 1024。
- 不用伪高分辨率样本污染后续阶段。
**负面：**
- 每个训练 step 需要在线图像解码、预处理和 VAE encode。
- 高分辨率阶段可用样本数会因禁止放大而减少。
- Mage-VAE 的二次元域质量必须先通过实际数据验证。
- 动态随机 crop 可能造成 caption/tag 与可见内容不一致，需通过小裁剪比例和后续数据规则控制。
# 失败与回退
- 重建指标或人工错误率未通过：先核查颜色、resize/crop 与输入范围；确认是系统性 VAE 域偏差后，重新打开 VAE 决策。
- 512 完整训练吞吐低于 01 门槛：分段定位 VAE 与网络占比；先尝试 encoder-only、同形状批处理、compile 和本地 shard staging，再评估阶段性 latent cache。
- 高分辨率阶段可用原生样本不足：降低该阶段数据占比或最高训练分辨率，不允许通过无条件放大补足。
# 依据与参考
- [Mage-Flow / Mage-VAE 技术报告](https://arxiv.org/abs/2607.19064)
- [Microsoft Mage-Flow 官方模型](https://huggingface.co/microsoft/Mage-Flow)
- [Microsoft Mage-VAE 官方实现](https://github.com/microsoft/Mage/blob/main/mage_flow/models/modules/mage_vae.py)
- [JLT 固定参考提交 aca236e](https://github.com/akatsuki-neo/JLT/tree/aca236efa97aab3b7d865fd3d99a270431cf6ae5)
- [HDM 固定参考提交 5fef7c4](https://github.com/KohakuBlueleaf/HDM/tree/5fef7c4b71fe8386b497176021fe458810fdb7c0)
- [HDM 技术报告的位置编码与 Shifted Square Crop](https://github.com/KohakuBlueleaf/HDM/blob/5fef7c4b71fe8386b497176021fe458810fdb7c0/TechReport.md)
</content>
</page>
