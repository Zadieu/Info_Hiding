# `models/` — 网络结构实现（分工 2）

> 负责人：分工 2
> 内容：实现 **Encoder / Decoder / Adversary(Discriminator)**，严格按论文
> *HiDDeN: Hiding Data With Deep Networks*（Zhu, Kaplan, Johnson, Fei-Fei, ECCV 2018）
> 的 **Appendix A** 复现；正确处理 **消息空间复制（message replication）与通道拼接（channel concatenation）**。

本模块只负责“网络长什么样、前向传播怎么走”，**不负责训练**（分工 3 的 `train.py` / `losses.py`）、
**不负责噪声**（分工 4 的 `noise_layers/`）、**不负责评估画图**（分工 5）。

---

## 1. 文件清单

| 文件               | 作用                                                         |
| ------------------ | ------------------------------------------------------------ |
| `blocks.py`        | 基础积木 `ConvBNRelu`（Conv 3×3 / stride 1 / pad 1 → BatchNorm → ReLU），三个网络共用 |
| `encoder.py`       | 编码器 `Encoder`（E_θ）：cover + message → 编码图 I_en       |
| `decoder.py`       | 解码器 `Decoder`（D_φ）：(可能被加噪的) 图 → 预测消息 M_out  |
| `discriminator.py` | 对抗器 `Discriminator`（A_γ）：图 → P(该图是编码图) ∈ [0,1]  |
| `test_models.py`   | 自测脚本，证明上述实现正确（形状 / 概率 / 复制 / 梯度 / 尺寸无关 / 端到端联调） |
| `hidden.py`        | 把 Encoder→Noise→Decoder→Discriminator 组装成一个模型（与分工 3 协作的整合层） |
| `noise.py`         | 噪声层占位（正式噪声实现见顶层 `noise_layers/`，分工 4）     |

---

## 2. 架构对照表（论文 Appendix A → 代码）

所有卷积块默认 **3×3 卷积、stride 1、padding 1**，因此 H、W 在网络内部保持不变。
中间通道数均为 **64**。

### Encoder（`encoder.py`）

论文原文：4 个 Conv-BN-ReLU 块（64 通道）→ 复制消息成 message volume → 与图像特征、原图按通道拼接成 `(64+L+C)` → 1 个 Conv-BN-ReLU 块（64 通道）→ 1×1 卷积（C 通道，无激活）。

```
cover (B,C,H,W) ──► [ConvBNReLU × 4]  ──► features (B,64,H,W)
message (B,L)   ──► view(B,L,1,1).expand(B,L,H,W) ──► message_volume (B,L,H,W)
concat([features, message_volume, cover], dim=1)  ──► (B, 64+L+C, H, W)
        ──► ConvBNReLU(64) ──► Conv2d 1×1 → C 通道（无激活） ──► encoded I_en (B,C,H,W)
```

> **消息空间复制 + 通道拼接（本分工的核心难点）**：
> 消息是一维的 `(B, L)`，图像是三维的 `(B, C, H, W)`，维度对不上无法直接拼。
> 我们把消息 reshape 成 `(B, L, 1, 1)` 再用 `expand` 广播到 `(B, L, H, W)`——
> 即在**每一个空间位置都放一份完整的 L-bit 消息**。这样下一层卷积无论滑到哪个位置，
> 都能读到完整消息，编码器才能自由地决定把消息藏在图像的哪个区域。
> `expand` 不复制内存，真正的拼接由后续 `torch.cat` 完成。

> **关于拼接顺序**：本实现的拼接顺序是 `[features, message_volume, cover]`，
> 而 ando 复现是 `[message, features, cover]`。**顺序不同不影响正确性**——紧随其后的
> `after_concat_layer` 卷积会按实际拼接顺序学到对应权重，二者完全等价。

> **关于输入尺寸**：前向传播直接读取输入张量真实的 H、W 来做消息广播（而非依赖构造时
> 传入的 H、W），因此对非训练尺寸的输入同样适用。

### Decoder（`decoder.py`）

论文原文：7 个 Conv-BN-ReLU 块（64 通道）+ 1 个 Conv-BN-ReLU 块（L 通道）→ 对所有空间维度做平均池化 → L×L 线性层。

```
image (B,C,H',W') ──► [ConvBNReLU × 7] ──► ConvBNReLU(→L) ──►
        AdaptiveAvgPool2d(1×1) ──► (B,L) ──► Linear(L,L) ──► M_out (B,L)
```

> **为什么用全局平均池化**：噪声层可能裁剪图像（Crop 只保留 3.5% 面积），
> 解码器收到的尺寸 H'、W' 不固定。全局平均池化把任意 H'×W' 压成长度 L 的向量，
> 使解码器**与输入尺寸无关**，从而能抗裁剪。M_out 为实数值，只在评估时四舍五入到 {0,1}。

### Discriminator / Adversary（`discriminator.py`）

论文原文：3 个 Conv-BN-ReLU 块（64 通道）→ 空间平均 → 线性层做二分类。

```
image (B,C,H,W) ──► [ConvBNReLU × 3] ──► AdaptiveAvgPool2d(1×1) ──►
        (B,64) ──► Linear(64,1) ──► Sigmoid ──► A(I) ∈ [0,1]  (B,1)
```

> **关于输出形式（与论文、与 ando 复现的差异说明）**：论文 Appendix A 写的是
> “2 个输出单元 + softmax 的二分类”；ando 复现则输出**单个原始 logit（不接 sigmoid）**，
> 训练时用 `BCEWithLogitsLoss`。本项目选择第三种等价写法：直接输出
> **单个概率 `A(I)=P(I 是编码图) ∈ [0,1]`**（末尾接 Sigmoid，返回 `(B,1)`）。
> 这三种写法对二分类问题**数学上等价**。我们之所以采用概率头，是为了直接对接本组
> `losses.py` 中论文原式的概率形式：`L_G = log(1 − A(I_en))`、
> `L_A = log(1 − A(I_co)) + log(A(I_en))`，这要求 `A(·)` 是 [0,1] 的概率。

> **⚠️ 给分工 3 的数值稳定性提醒**：本模块输出的是“硬” sigmoid 概率，一旦判别器
> 极度自信、`A` 饱和到 0 或 1，`losses.py` 里的 `log(A)` / `log(1 − A)` 会变成
> `-inf` 导致训练出现 NaN。ando 复现用 `BCEWithLogitsLoss`（内部 logsumexp）规避了这一点。
> 因此 **`losses.py` 中务必对 `A` 做 `clamp(eps, 1 − eps)`（或改用 `logsigmoid`）**。
> 这是损失实现侧的职责，本模块按接口契约只负责返回合法概率。

---

## 3. 关键超参数（与论文 / 权威复现保持一致）

参数取自论文，并与社区广泛使用的 PyTorch 复现
[`ando-khachatryan/HiDDeN`](https://github.com/ando-khachatryan/HiDDeN) 对齐
（注：该复现在其 README 中**引用了原作者的 Lua+Torch 实现**
[`jirenz/HiDDeN`](https://github.com/jirenz/HiDDeN)；原论文本身没有给出 PyTorch 代码，
故以原作者 Lua+Torch 实现为根、ando 的 PyTorch 复现为对齐参照）：

| 超参数                    | 取值                                        | 出处                                                         |
| ------------------------- | ------------------------------------------- | ------------------------------------------------------------ |
| Encoder 特征块数          | 4                                           | Appendix A / ando `encoder_blocks=4`                         |
| Encoder 拼接后块数        | 1                                           | Appendix A                                                   |
| Decoder 块数              | 7 (+1 个输出 L 通道)                        | Appendix A / ando `decoder_blocks=7`                         |
| Discriminator 块数        | 3                                           | Appendix A / ando `discriminator_blocks=3`                   |
| 中间通道数                | 64                                          | Appendix A / ando `encoder_channels=decoder_channels=discriminator_channels=64` |
| 卷积核 / stride / padding | 3×3 / 1 / 1                                 | Appendix A                                                   |
| 隐写实验                  | C=1, 16×16, L=52                            | Sec. 4.1                                                     |
| 水印实验                  | C=3(YUV), 128×128, L=30，λ_I=0.7，λ_G=0.001 | Sec. 4.2                                                     |

实现细节说明：

- `ConvBNRelu` 中卷积设 `bias=False`，因为紧随其后的 BatchNorm 自带可学习偏置（β），卷积偏置冗余、会被 BN 抵消。
  **这是常见做法，但需说明：ando 复现保留了卷积的默认 `bias=True`**；两者训练行为等价（BN 会吸收偏置），
  仅参数量略有差异，不影响复现结果与论文一致性。
- Encoder 最后的 1×1 输出卷积**保留**偏置（其后无 BN），这与 ando 复现一致。
- 权重初始化使用 PyTorch 默认（论文与 ando 复现均未指定自定义初始化），不额外修改以保证可复现一致。
- Encoder 前向用输入张量的真实 H、W 做消息广播，因此对非训练尺寸的输入同样适用。

---

## 4. 对外接口契约（与分工 3 / 4 约定，请勿改动签名）

```python
Encoder(C, H, W, L)          # forward(cover (B,C,H,W), message (B,L)) -> encoded (B,C,H,W)
Decoder(C, L)                # forward(image (B,C,H',W'))             -> message (B,L)
Discriminator(C)             # forward(image (B,C,H,W))               -> prob   (B,1) in [0,1]
```

这些签名被 `models/hidden.py` 与 `train.py` 直接调用，任何修改都会影响分工 3 的训练代码。

---

## 5. 如何验证我的实现是否正确

在 `DataLoader/` 目录下运行自测脚本（**无需数据、无需 GPU**）：

```bash
python models/test_models.py
```

脚本会检查：

1. 两种论文配置（隐写 C1/16×16/L52、水印 C3/128×128/L30）下的输出形状；
2. Discriminator 输出确实是 [0,1] 概率；
3. message volume 在每个空间位置都正确复制了完整消息；
4. 梯度能正常回传到三个网络（backward 可用）；
5. Decoder 对裁剪后/非方形的输入尺寸无关；
6. 与分工 3 的 `losses.py` 端到端联调，能完整跑一次优化步。

> **依赖说明**：第 1–5 项为**本分工自包含测试**，不依赖任何其他分工的文件即可运行；
> 第 6 项（端到端联调）依赖分工 3 的 `models/hidden.py` 与 `losses.py`，需这两个文件就位后才能通过。

全部通过会打印 `ALL TESTS PASSED`。

也可以用分工 3 的训练脚本做一次合成数据冒烟测试：

```bash
python train.py --use_synthetic --num_epochs 2 --train_size 60 --val_size 24 \
    --batch_size 12 --num_workers 0 --C 1 --H 16 --W 16 --L 52 --noise identity
```

应观察到 `L_M` 随训练下降、checkpoint 正常保存，证明网络结构与整个流水线打通。