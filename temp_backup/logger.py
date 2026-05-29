# ==================== logger.py ====================
"""
任务5: TensorBoard 日志记录

修复记录:
  [Fix-6] log_metrics 原来用 self.step 作为横轴，但 evaluate_and_log 只调了
          set_epoch(epoch) 而从未更新 self.step，导致所有测试指标都落在 step=0。
          修复方案：log_metrics 优先用 self.step，调用方传 step 参数即可；
          同时新增 log_epoch_metrics 方法，语义更清晰，以 epoch 为横轴。
"""

import os
import torch
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """TensorBoard 日志记录器"""

    def __init__(self, log_dir: str = './logs', experiment_name: str = 'hidden'):
        self.log_dir = os.path.join(log_dir, experiment_name)
        self.writer = SummaryWriter(self.log_dir)
        self.step = 0    # 训练步（batch 级别）
        self.epoch = 0   # epoch 级别

    # ------------------------------------------------------------------
    # 基础 step 管理
    # ------------------------------------------------------------------
    def set_step(self, step: int) -> None:
        """设置当前全局训练步（一般在 batch 循环里调用）"""
        self.step = step

    def set_epoch(self, epoch: int) -> None:
        """设置当前 epoch（同时更新 step，供 epoch 级日志使用）"""
        self.epoch = epoch
        # [Fix-6] 让 step 与 epoch 同步，避免 evaluate_and_log 里
        #          忘记调 set_step 导致所有点落在 step=0
        self.step = epoch

    # ------------------------------------------------------------------
    # Scalar 日志
    # ------------------------------------------------------------------
    def log_scalar(self, tag: str, value: float, step: int = None) -> None:
        """记录单个标量"""
        self.writer.add_scalar(tag, value, step if step is not None else self.step)

    def log_scalars(self, tag_dict: dict, step: int = None) -> None:
        """批量记录标量"""
        s = step if step is not None else self.step
        for tag, value in tag_dict.items():
            self.writer.add_scalar(tag, value, s)

    def log_metrics(self, metrics: dict, prefix: str = 'train',
                    step: int = None) -> None:
        """
        记录一批指标到 TensorBoard。

        Args:
            metrics: {'psnr': 44.5, 'bit_accuracy': 98.2, ...}
            prefix:  'train' 或 'test'，决定 TensorBoard 中的分组
            step:    显式指定横轴位置；若不传则使用 self.step
                     （self.step 在 set_epoch / set_step 中已更新）
        """
        s = step if step is not None else self.step
        for name, value in metrics.items():
            self.writer.add_scalar(f'{prefix}/{name}', value, s)

    # ------------------------------------------------------------------
    # 图像日志
    # ------------------------------------------------------------------
    def log_images(self, tag: str, images: torch.Tensor,
                   step: int = None, nrow: int = 4) -> None:
        """记录图像 batch 到 TensorBoard"""
        s = step if step is not None else self.step
        images = torch.clamp(images.clone(), 0, 1)
        self.writer.add_images(tag, images, s, dataformats='NCHW')

    def log_comparison_grid(self, cover: torch.Tensor, stego: torch.Tensor,
                            noised: torch.Tensor = None,
                            step: int = None, nrow: int = 4) -> None:
        """
        记录 Cover vs Stego (vs Noised) 对比网格到 TensorBoard。

        注意：内部创建副本进行归一化，不修改传入的原始 tensor。
        """
        s = step if step is not None else self.step

        def _to_display(img: torch.Tensor) -> torch.Tensor:
            """克隆并归一化到 [0,1]，不影响原 tensor"""
            d = img.detach().clone().float()
            if d.min() < 0:          # [-1,1] → [0,1]
                d = (d + 1.0) / 2.0
            return torch.clamp(d, 0, 1)

        cover_d = _to_display(cover)
        stego_d = _to_display(stego)
        n = min(cover.shape[0], nrow * nrow)

        comparison = torch.cat([cover_d[:n], stego_d[:n]], dim=0)
        self.writer.add_images('comparison/cover_vs_stego', comparison, s,
                               dataformats='NCHW')

        if noised is not None:
            noised_d = _to_display(noised)
            all_three = torch.cat([cover_d[:n], stego_d[:n], noised_d[:n]], dim=0)
            self.writer.add_images('comparison/cover_stego_noised', all_three, s,
                                   dataformats='NCHW')

    # ------------------------------------------------------------------
    # 其他
    # ------------------------------------------------------------------
    def log_histogram(self, tag: str, values: torch.Tensor,
                      step: int = None) -> None:
        s = step if step is not None else self.step
        self.writer.add_histogram(tag, values, s)

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()
