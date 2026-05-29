# ==================== metrics.py ====================
"""
任务5: PSNR、Bit Accuracy计算
参考: tester.lua 中的 PSNR 和 computeDiff 函数

修复记录:
  [Fix-4] compute_psnr_per_channel: psnr 计算后忘记写入 psnrs[name]，导致返回空字典
  [Fix-5] compute_bit_accuracy: is_logit 默认值改为 False
          原因: losses.py 的 bit_accuracy() 直接 .round().clamp(0,1)，说明 decoder
          输出已是 [0,1]，不是 logit；默认 True 会多做一次 sigmoid，精度会偏低
"""

import torch
import numpy as np


class MetricsCalculator:
    """指标计算器 - 对应 tester.lua 中的功能"""

    def __init__(self, image_range: float = 1.0):
        """
        Args:
            image_range: 图像像素范围，默认1.0（归一化后 [0,1]）
        """
        self.image_range = image_range
        # 与 losses.py 的 bit_accuracy 保持一致，使用 .round() 而非 0.499 阈值
        # 保留此属性供外部直接访问，但内部 compute_bit_accuracy 已不再使用
        self.msg_threshold = 0.5

    # ------------------------------------------------------------------
    # PSNR
    # ------------------------------------------------------------------
    def compute_psnr(self, cover: torch.Tensor, stego: torch.Tensor) -> float:
        """
        计算整体 PSNR。
        参考 tester.lua 的 PSNR()，统一使用 torch 操作。

        Args:
            cover: 原始图像 [B, C, H, W]，值域 [0, 1]
            stego: 编码图像 [B, C, H, W]，值域 [0, 1]
        Returns:
            psnr: float，单位 dB
        """
        if isinstance(cover, np.ndarray):
            cover = torch.from_numpy(cover)
        if isinstance(stego, np.ndarray):
            stego = torch.from_numpy(stego)

        mse = torch.mean((cover.float() - stego.float()) ** 2)
        if mse == 0:
            return float('inf')

        # [Fix-1] 统一使用 torch.log10，避免 np/torch 混用
        range_t = torch.tensor(self.image_range, dtype=torch.float32, device=mse.device)
        psnr = 20.0 * torch.log10(range_t) - 10.0 * torch.log10(mse)
        return psnr.item()

    def compute_psnr_per_channel(self, cover: torch.Tensor, stego: torch.Tensor) -> dict:
        """
        计算每个通道的 PSNR（用于 YUV 图像，对应论文表格 PSNR_Y/U/V）。
        参考 tester.lua 中 Y/U/V 不同 range 的计算。

        论文 YUV range:
            Y 通道: 1.0
            U 通道: 0.436 × 2 = 0.872
            V 通道: 0.615 × 2 = 1.230

        Returns:
            dict: {'y': float, 'u': float, 'v': float}
        """
        if cover.shape[1] == 3:
            cover_yuv = self.rgb_to_yuv(cover)
            stego_yuv = self.rgb_to_yuv(stego)
        else:
            cover_yuv = cover
            stego_yuv = stego

        channel_ranges = {'y': 1.0, 'u': 0.872, 'v': 1.23}
        psnrs = {}

        for i, (name, r) in enumerate(channel_ranges.items()):
            if i >= cover_yuv.shape[1]:
                break
            mse = torch.mean(
                (cover_yuv[:, i:i+1].float() - stego_yuv[:, i:i+1].float()) ** 2
            )
            if mse > 0:
                r_t = torch.tensor(r, dtype=torch.float32, device=mse.device)
                psnr_val = 20.0 * torch.log10(r_t) - 10.0 * torch.log10(mse)
                # [Fix-4] 原代码计算了 psnr_val 但忘记赋值，此处补全
                psnrs[name] = psnr_val.item()
            else:
                psnrs[name] = float('inf')

        return psnrs

    # ------------------------------------------------------------------
    # Bit Accuracy / BER
    # ------------------------------------------------------------------
    def compute_bit_accuracy(
        self,
        msg_gt: torch.Tensor,
        msg_pred: torch.Tensor,
        is_logit: bool = False,   # [Fix-5] 默认改为 False
    ):
        """
        计算 Bit Accuracy 和 BER。

        与 losses.py 的 bit_accuracy() 保持一致：
            pred = msg_out.detach().round().clamp(0, 1)
        decoder 输出已是 [0,1]，不需要 sigmoid。

        Args:
            msg_gt:   原始消息 [B, L]，值为 0 或 1
            msg_pred: 预测消息 [B, L]，值域 [0,1]（decoder 直接输出）
            is_logit: 若 decoder 最后一层未过激活函数，设为 True 会先做 sigmoid。
                      默认 False，与现有 losses.py 保持一致。
        Returns:
            accuracy: float ∈ [0, 1]，0.5 = 随机猜测
            ber:      float ∈ [0, 1]，bit error rate
        """
        if msg_pred.dim() == 1:
            msg_pred = msg_pred.unsqueeze(0)
            msg_gt = msg_gt.unsqueeze(0)

        if is_logit:
            msg_pred = torch.sigmoid(msg_pred)

        # 与 losses.py 完全对齐：round().clamp(0,1)
        msg_pred_binary = msg_pred.detach().round().clamp(0, 1)

        accuracy = (msg_pred_binary == msg_gt).float().mean().item()
        ber = 1.0 - accuracy
        return accuracy, ber

    # ------------------------------------------------------------------
    # 辅助：多阈值统计（对应 confusion.lua）
    # ------------------------------------------------------------------
    def compute_message_metrics(self, msg_gt: torch.Tensor, msg_pred: torch.Tensor,
                                thresholds=None) -> dict:
        """
        消息在不同阈值下的误差统计，对应 confusion.lua 的 msg_threasholds。
        """
        if thresholds is None:
            thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        results = {}
        for thresh in thresholds:
            diff = torch.abs(msg_gt.float() - msg_pred.float())
            results[thresh] = (diff > thresh).float().mean().item()
        return results

    def compute_image_metrics(self, cover: torch.Tensor, stego: torch.Tensor,
                              thresholds=None) -> dict:
        """
        图像在不同阈值下的像素差异统计，对应 confusion.lua 的 img_threasholds。
        """
        if thresholds is None:
            thresholds = [1/256, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4]
        diff = torch.abs(cover.float() - stego.float())
        results = {}
        for thresh in thresholds:
            results[thresh] = (diff > thresh).float().mean().item()
        return results

    # ------------------------------------------------------------------
    # 色彩空间转换
    # ------------------------------------------------------------------
    @staticmethod
    def rgb_to_yuv(rgb: torch.Tensor) -> torch.Tensor:
        """BT.601 RGB → YUV，与 jpeg.py 中的转换矩阵一致。"""
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        y =  0.299   * r + 0.587   * g + 0.114   * b
        u = -0.14713 * r - 0.28886 * g + 0.436   * b
        v =  0.615   * r - 0.51499 * g - 0.10001 * b
        return torch.cat([y, u, v], dim=1)
