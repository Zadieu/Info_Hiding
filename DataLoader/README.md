# HiDDeN — PyTorch 复现

基于论文 [HiDDeN: Hiding Data With Deep Networks](https://arxiv.org/abs/1807.09937) (ECCV 2018) 的 PyTorch 实现，包含 Encoder / Decoder / Discriminator、COCO DataLoader、论文 Sec.3 损失、Adam 训练循环与 Checkpoint 保存。

## 项目结构

```
HiDDeN/
├── config.py              # 训练超参与默认实验设定
├── dataset.py             # COCO DataLoader、消息采样
├── losses.py              # L_M, L_I, L_G, L_A 与 bit accuracy
├── train.py               # 训练 / 验证主入口
├── requirements.txt
├── models/
│   ├── encoder.py         # 附录 A：消息复制 + 通道拼接
│   ├── decoder.py
│   ├── discriminator.py
│   ├── noise.py           # Identity / Gaussian / Crop / Dropout / Cropout
│   ├── hidden.py          # HiDDeNModel 封装
│   └── blocks.py
├── utils/
│   └── checkpoint.py      # 保存 / 加载权重
├── data/
│   └── coco/train2017/   # COCO 训练图像（*.jpg，至少 11000 张）
├── checkpoints/
│   └── hidden_stego/      # 训练产出（epoch_010.pt … epoch_200.pt）
└── COCO2017_Cache/        # 可选：ModelScope 原始下载缓存
```

## 环境安装

```bash
pip install -r requirements.txt
```

依赖：`torch>=2.0`、`torchvision>=0.15`、`Pillow>=9.0`。推荐使用 CUDA GPU 训练。

## 数据准备

### 论文要求

| 项目 | 说明 |
|------|------|
| 数据集 | MS COCO `train2017` |
| 训练集 | 10,000 张载体图 |
| 验证集 | 1,000 张（与训练不重叠的子集） |
| 消息 | 每位独立 `Uniform{0,1}`，训练时在线随机生成，**无需单独消息文件** |

### 目录布局

将 COCO 图片解压到：

```
data/coco/train2017/
├── 000000000009.jpg
├── 000000000025.jpg
└── ...
```

代码使用 `dataset.py` 中的 `_FlatImageDataset` 直接读取该目录下的 `.jpg` 文件。

### 数据划分方式

`build_loaders` 对数据集按**文件名排序后的前 11000 张**划分：

- 索引 `0 … 9999` → 训练集（10,000）
- 索引 `10000 … 10999` → 验证集（1,000）

全组实验请保持相同 `data_root` 与 `train_size` / `val_size`，以保证划分一致。

### 从 ModelScope 缓存链接（可选）

若本地已有 `COCO2017_Cache`，可用目录联接避免复制大文件（Windows 示例）：

```powershell
New-Item -ItemType Directory -Force -Path data\coco
cmd /c mklink /J "data\coco\train2017" "<你的路径>\COCO2017train\train2017"
```

或直接指定 `--data_root` 指向缓存内的 `train2017` 文件夹。

## 快速开始

### 无数据冒烟测试（检查代码 / GPU）

```bash
python train.py --use_synthetic --train_size 120 --val_size 24 --num_epochs 2
```

### 论文隐写实验（Sec. 4.1 默认设定）

灰度 16×16，`L=52`，Identity 噪声，Adam `lr=1e-3`，`batch=12`，200 epoch：

```bash
python train.py ^
  --data_root data/coco/train2017 ^
  --train_size 10000 ^
  --val_size 1000 ^
  --C 1 --H 16 --W 16 --L 52 ^
  --noise identity ^
  --batch_size 12 ^
  --lr 1e-3 ^
  --num_epochs 200 ^
  --device cuda
```

PowerShell 可将 `^` 换为行末反引号 `` ` ``，或写成一行。

### 水印 / 鲁棒实验（Sec. 4.2 示例）

```bash
python train.py ^
  --data_root data/coco/train2017 ^
  --C 3 --H 128 --W 128 --L 30 ^
  --noise dropout ^
  --num_epochs 400
```

### 断点续训

```bash
python train.py --data_root data/coco/train2017 --resume checkpoints/hidden_stego/epoch_200.pt
```

将从 checkpoint 中记录的 `epoch + 1` 继续训练。

## 训练结果（`hidden_stego`，200 epoch，Identity）

在 COCO 10k/1k 划分、16×16 灰度、`L=52` 设定下，最终 epoch 典型指标：

| 指标 | 数值 |
|------|------|
| train bit accuracy | ≈ 0.963 |
| val bit accuracy | ≈ 0.962 |
| val L_M | ≈ 0.057 |

最终权重：`checkpoints/hidden_stego/epoch_200.pt`

## 命令行参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--data_root` | `data/coco/train2017` | COCO 图像目录 |
| `--use_synthetic` | off | 随机张量代替 COCO（仅冒烟） |
| `--train_size` / `--val_size` | 10000 / 1000 | 训练 / 验证样本数 |
| `--C` / `--H` / `--W` | 1 / 16 / 16 | 通道与高宽 |
| `--L` | 52 | 消息比特长度 |
| `--noise` | `identity` | `identity` \| `gaussian` \| `crop` \| `dropout` \| `cropout` |
| `--lambda_I` / `--lambda_G` | 1.0 / 0.1 | 式 (1) 中图像损失与对抗损失权重 |
| `--lr` | 1e-3 | Adam 学习率 |
| `--batch_size` | 12 | 批大小 |
| `--num_epochs` | 200 | 训练轮数 |
| `--save_every` | 10 | 每 N epoch 保存 checkpoint |
| `--experiment_name` | `hidden_stego` | 子目录名 |
| `--resume` | — | 从指定 `.pt` 恢复 |
| `--device` | auto | `cuda` / `cpu` |

## 训练流程（与论文对应）

每个 batch：

1. 从 DataLoader 取载体图 `I_co`（cover）
2. `sample_messages` 生成随机比特消息 `M_in`
3. **更新判别器 A**（式 2）：最小化 `L_A = log(1-A(I_co)) + log(A(I_en))`
4. **更新 Encoder + Decoder**（式 1）：
   - `L_ED = L_M + λ_I·L_I + λ_G·L_G`
   - `L_M`：消息重建；`L_I`：含密图接近载体；`L_G`：对抗项使 `A(I_en)→0`
5. Decoder 输入为 **噪声层输出** `I_no`；判别器对抗分支使用 **编码输出** `I_en`

优化器：两个 Adam（`opt_ed` 用于 E+D，`opt_d` 用于 A），学习率默认 `1e-3`。

## 损失函数（`losses.py`）

| 符号 | 实现 | 说明 |
|------|------|------|
| L_I | `loss_I` | `‖I_co - I_en‖² / (C·H·W)`，再对 batch 求均值 |
| L_M | `loss_M` | `‖M_in - M_out‖² / L` |
| L_G | `loss_G` | `mean(log(1 - A(I_en)))` |
| L_A | `loss_A` | `mean(log(1-A(I_co)) + log(A(I_en)))` |
| — | `bit_accuracy` | 将 `M_out` 四舍五入到 {0,1} 后与 `M_in` 比较 |

## Checkpoint 格式

路径：`checkpoints/<experiment_name>/epoch_XXX.pt`

| 字段 | 内容 |
|------|------|
| `epoch` | 当前训练轮次 |
| `encoder` / `decoder` / `noise` | 网络 `state_dict` |
| `discriminator` | 判别器权重 |
| `opt_ed` / `opt_d` | 优化器状态 |
| `config` | 训练时完整配置字典 |

加载示例（供评估 / 可视化模块使用）：

```python
from pathlib import Path
import torch
from models.hidden import HiDDeNModel
from utils.checkpoint import load_checkpoint

ckpt_path = Path("checkpoints/hidden_stego/epoch_200.pt")
cfg = torch.load(ckpt_path, map_location="cpu", weights_only=False)["config"]

model = HiDDeNModel(cfg["image_channels"], cfg["image_height"], cfg["image_width"],
                    cfg["message_length"], cfg["noise_type"], {...}).cuda()
# 加载权重后：encoded, noised, decoded = model(cover, message)
```

## 模块 API（组内对接）

```python
from dataset import build_loaders, sample_messages
from losses import loss_M, loss_I, loss_G, loss_A, bit_accuracy
from utils.checkpoint import save_checkpoint, load_checkpoint
```

- **`build_loaders(data_root, C, H, W, train_size, val_size, batch_size, num_workers, use_synthetic)`**  
  返回 `(train_loader, val_loader)`，batch 元素为 `(image_tensor, label)`，`label` 未使用。

- **`sample_messages(batch_size, L, device)`**  
  返回 `(B, L)` 的 float 张量，取值为 0 或 1。

## 网络概要（附录 A）

- **Encoder**：4× Conv3×3-BN-ReLU（64 通道）→ 将 `M_in` 复制为 `L×H×W` → 与特征图、原图拼接 → 1× Conv-BN-ReLU → 1×1 Conv → `I_en`
- **Decoder**：7×64 卷积 + 1×L 卷积 → GAP → Linear → `M_out`
- **Discriminator**：3×64 卷积 → GAP → Linear → Sigmoid → `A(·)∈[0,1]`
- **Noise**：无参数层，由 `--noise` 选择类型；`forward(cover, encoded) → noised`

## 噪声层（`models/noise.py`）

| `--noise` | 行为 |
|-----------|------|
| `identity` | `I_no = I_en` |
| `gaussian` | 高斯模糊（`sigma` 默认 2.0） |
| `crop` | 随机裁剪小块 |
| `dropout` | 随机用载体像素替换含密像素 |
| `cropout` | 随机区域保留含密、其余用载体 |

当前训练脚本每次运行固定一种噪声；**训练时随机切换多种噪声**需在此基础上扩展 `NoiseLayer` 或 `train.py`。

## 已知说明

- 验证集为 COCO train 的子集，非官方 `val2017`；与论文「训练未见过的 1000 张」表述一致即可。
- 日志中 `L_G`、`L_A` 在对抗训练饱和后可能稳定在约 **-18.42**（`log` 下溢），此时应以 **bit accuracy** 与 **L_M** 为主指标。
- 本项目**未包含** TensorBoard、PSNR 统计脚本、Cover/Stego 对比图生成；可由评估模块单独加载 checkpoint 实现。

## 参考文献

Zhu, J., Kaplan, R., Johnson, J., & Fei-Fei, L. (2018). *HiDDeN: Hiding Data With Deep Networks*. ECCV 2018. [arXiv:1807.09937](https://arxiv.org/abs/1807.09937)
