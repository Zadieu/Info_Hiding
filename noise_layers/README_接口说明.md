# HiDDeN 噪声层模块接口说明

本文档用于小组联调时快速说明 `hidden_repro` 中噪声层模块的职责、目录组成、接口标准和接入方式。

适用对象：

- 负责 `Encoder / Decoder / Adversary` 的同学
- 负责训练循环、损失函数、checkpoint 的同学
- 后续需要把噪声层合并进总项目的人

## 1. 模块作用

这个模块实现的是 HiDDeN 论文里的 `Noise Layer`，位置在 `Encoder` 和 `Decoder` 之间。

它的作用不是做普通数据增强，而是在训练阶段模拟真实传输过程中的破坏，让 `Decoder` 学会从被破坏后的图像中恢复消息。

在论文里，噪声层主要用于两类任务：

- 隐写模式：使用 `IdentityNoise`
- 鲁棒水印模式：使用 `Gaussian / Dropout / Crop / Cropout / JPEG` 等噪声

标准数据流如下：

```python
cover -> encoder -> encoded -> noise_layer -> noised -> decoder -> decoded_message
```

如果项目里有对抗判别器，一般接法是：

```python
cover -> encoder -> encoded -> adversary
cover -> encoder -> encoded -> noise_layer -> noised -> decoder
```

## 2. 目录组成

当前目录如下：

```text
hidden_repro/
├─ noise_layers/
│  ├─ base.py
│  ├─ identity.py
│  ├─ gaussian.py
│  ├─ dropout.py
│  ├─ crop.py
│  ├─ cropout.py
│  ├─ jpeg.py
│  ├─ manager.py
│  └─ __init__.py
```

各文件职责如下：

- `noise_layers/base.py`
  - 定义统一基类 `BaseNoiseLayer`
  - 约束所有噪声层的共同接口
- `noise_layers/identity.py`
  - 实现 `IdentityNoise`
  - 不修改 `encoded`
- `noise_layers/gaussian.py`
  - 实现 `GaussianBlurNoise`
  - 对 `encoded` 做高斯模糊
- `noise_layers/dropout.py`
  - 实现 `DropoutNoise`
  - 随机保留 `encoded` 的像素，其余位置替换成 `cover`
- `noise_layers/crop.py`
  - 实现 `CropNoise`
  - 对 `encoded` 做随机正方形裁剪
- `noise_layers/cropout.py`
  - 实现 `CropoutNoise`
  - 只保留一块 `encoded` 区域，其余区域使用 `cover`
- `noise_layers/jpeg.py`
  - 实现 `JpegMaskNoise`
  - 实现 `JpegDropNoise`
  - 两者都是训练时用于近似 JPEG 压缩的可微模块
- `noise_layers/manager.py`
  - 实现 `NoiseManager`
  - 实现 `build_noise_layer()`
  - 实现 `build_paper_noise_layers()`
- `test_noise_layers.py`
  - 噪声层冒烟测试
- `demo_train_step.py`
  - 一个最小训练步 demo，用来验证噪声层能否接入训练流程

## 3. 推荐放置方式

合并到总项目时，推荐把这个模块放在和 `models` 平级的一层，而不是塞进某个网络文件里。

推荐结构：

```text
project_root/
├─ models/
│  ├─ encoder.py
│  ├─ decoder.py
│  └─ adversary.py
├─ noise_layers/
│  ├─ base.py
│  ├─ identity.py
│  ├─ gaussian.py
│  ├─ dropout.py
│  ├─ crop.py
│  ├─ cropout.py
│  ├─ jpeg.py
│  ├─ manager.py
│  └─ __init__.py
├─ losses/
├─ datasets/
├─ train.py
└─ config.py
```

如果项目采用包结构，也推荐保持 `models` 和 `noise_layers` 平级：

```text
project_root/
├─ hidden/
│  ├─ models/
│  ├─ noise_layers/
│  ├─ losses/
│  ├─ datasets/
│  └─ __init__.py
├─ train.py
└─ config.py
```

原因很简单：

- `noise_layers` 不是网络主干
- `noise_layers` 也不是 dataset transform
- 单独成目录后，训练主流程更清楚
- 后续扩展和排错更方便

## 4. 接口标准

所有噪声层继承自 `BaseNoiseLayer`，统一接口如下：

```python
noised = noise_layer(cover, encoded)
```

### 输入

- `cover`
  - 原始封面图像
  - shape: `(B, C, H, W)`
- `encoded`
  - `Encoder` 输出的含密图像
  - shape: `(B, C, H, W)`

### 输入约束

- `cover.shape == encoded.shape`
- 两者必须都是 4 维张量
- 两者必须在同一设备上
- 建议像素范围统一为 `[0, 1]`

### 输出

- `noised`
  - 送入 `Decoder` 的图像

输出形状规则：

- 除 `CropNoise` 外，其他层输出 shape 与输入相同
- `CropNoise` 可能输出 `(B, C, H', W')`

## 5. 已实现的噪声层

当前导出接口如下：

```python
from hidden_repro.noise_layers import (
    IdentityNoise,
    GaussianBlurNoise,
    DropoutNoise,
    CropNoise,
    CropoutNoise,
    JpegMaskNoise,
    JpegDropNoise,
    NoiseManager,
    build_noise_layer,
    build_paper_noise_layers,
)
```

各层含义如下：

- `IdentityNoise()`
  - 不做任何修改
  - 对应论文里的 Identity
- `GaussianBlurNoise(sigma=2.0)`
  - 对 `encoded` 做高斯模糊
  - 论文里的 Gaussian 指的是模糊，不是加性高斯噪声
- `DropoutNoise(p=0.3)`
  - 像素级随机保留 `encoded`
  - 未保留位置用 `cover` 替换
- `CropNoise(p=0.035)`
  - 对 `encoded` 做随机正方形裁剪
  - 输出尺寸可能变小
- `CropoutNoise(p=0.3)`
  - 只保留一块 `encoded` 区域
  - 其他区域回退为 `cover`
- `JpegMaskNoise()`
  - 保留低频 DCT 系数，直接屏蔽高频系数
- `JpegDropNoise()`
  - 对高频 DCT 系数进行渐进式 dropout

## 6. 推荐接入方式

### 6.1 训练主流程中的位置

标准接法：

```python
encoded = encoder(cover, message)
noised = noise_manager(cover, encoded)
decoded = decoder(noised)
adv_pred = adversary(encoded)
```

这里需要注意：

- `noise_manager` 接在 `Encoder` 后、`Decoder` 前
- `Adversary` 一般接 `encoded`，不是接 `noised`

### 6.2 单层使用

如果只想单独测试某一层：

```python
from hidden_repro.noise_layers import DropoutNoise

noise = DropoutNoise(p=0.3)
noised = noise(cover, encoded)
```

```python
from hidden_repro.noise_layers import CropNoise

noise = CropNoise(p=0.035)
noised = noise(cover, encoded)
```

### 6.3 使用随机噪声管理器

如果要按论文组合模型的方式训练：

```python
from hidden_repro.noise_layers import NoiseManager, build_paper_noise_layers

noise_manager = NoiseManager(build_paper_noise_layers())
noised = noise_manager(cover, encoded)
```

这表示：

- 先构造论文配置中的 11 个噪声层
- 每个 forward 随机选一个噪声层执行

### 6.4 最小联调版本

如果总项目还没完全跑通，建议先只接 `IdentityNoise()`：

```python
from hidden_repro.noise_layers import IdentityNoise, NoiseManager

noise_manager = NoiseManager([IdentityNoise()])
encoded = encoder(cover, message)
noised = noise_manager(cover, encoded)
decoded = decoder(noised)
```

等主流程确认稳定后，再切换到：

```python
noise_manager = NoiseManager(build_paper_noise_layers())
```

## 7. 联调注意事项

### 7.1 Decoder 必须支持变尺寸输入

`CropNoise` 会改变输入尺寸，因此 `Decoder` 不能把输入大小写死。

推荐做法：

- 卷积提取特征
- 使用全局平均池化或 `AdaptiveAvgPool2d`
- 再映射到消息长度

### 7.2 Encoder 输出必须是图像张量

这个模块默认处理图像张量，而不是中间特征图。

也就是说：

- 可以接 `Encoder` 输出的图像
- 不能直接接隐藏层 feature map，除非整个项目统一约定它就是图像格式

### 7.3 像素范围建议统一为 `[0, 1]`

如果不同同学写的模块使用不同数值范围，最容易在联调时出问题。

推荐统一约定：

- `cover` 在 `[0, 1]`
- `encoded` 在 `[0, 1]`
- `noise_layers` 不负责做额外归一化

### 7.4 `Adversary` 的输入不要接错

推荐：

```python
adv_pred = adversary(encoded)
```

不推荐：

```python
adv_pred = adversary(noised)
```

因为论文里对抗判别器主要区分的是 `cover` 和 `encoded`。

### 7.5 JPEG 近似层的定位

`JpegMaskNoise` 和 `JpegDropNoise` 是训练阶段的可微近似层。

要区分：

- 训练时：用近似 JPEG，便于反向传播
- 测试鲁棒性时：建议额外跑真实 JPEG 压缩实验

## 8. 队友可直接参考的模板

### 8.1 完整接入模板

```python
from hidden_repro.noise_layers import NoiseManager, build_paper_noise_layers

noise_manager = NoiseManager(build_paper_noise_layers())

cover = batch["image"].to(device)
message = batch["message"].to(device)

encoded = encoder(cover, message)
noised = noise_manager(cover, encoded)
decoded = decoder(noised)
adv_pred = adversary(encoded)
```

### 8.2 最小可跑模板

```python
from hidden_repro.noise_layers import IdentityNoise, NoiseManager

noise_manager = NoiseManager([IdentityNoise()])

cover = batch["image"].to(device)
message = batch["message"].to(device)

encoded = encoder(cover, message)
noised = noise_manager(cover, encoded)
decoded = decoder(noised)
```

## 9. 测试文件

### 9.1 冒烟测试

运行方式：

```bash
python hidden_repro/test_noise_layers.py
```

测试内容：

- 各噪声层是否能成功前向
- 输出 shape 是否符合预期
- `DropoutNoise` 边界行为是否正确
- `NoiseManager` 是否能返回合法的 4 维张量

### 9.2 最小训练步演示

运行方式：

```bash
python hidden_repro/demo_train_step.py
```

这个脚本会完成：

- 构造一个假的 `Encoder`
- 构造一个支持变尺寸输入的假的 `Decoder`
- 调用 `NoiseManager(build_paper_noise_layers())`
- 完成一次前向
- 计算 loss
- 执行一次反向传播和参数更新

如果这个 demo 能跑通，说明噪声层模块已经具备接入训练主流程的基本条件。

## 10. 推荐联调顺序

建议按下面顺序合并：

1. 先运行 `test_noise_layers.py`
2. 再运行 `demo_train_step.py`
3. 把 `NoiseManager([IdentityNoise()])` 接入真实训练循环
4. 确认 `Encoder -> Noise -> Decoder` 主链路跑通
5. 逐个启用 `Dropout / Crop / Gaussian`
6. 最后启用 `build_paper_noise_layers()` 做组合噪声训练
