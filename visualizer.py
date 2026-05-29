# ==================== visualizer.py ====================
"""
任务5: 生成 Cover/Stego 对比图、鲁棒性曲线、训练曲线

鲁棒性曲线有两种风格：
  plot_robustness_curve()            ← 单模型 × 多噪声（训练中实时评估用）
  plot_robustness_curve_paper_style() ← 多模型 × 单噪声（对应论文图10，训练结束后用）
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import save_image, make_grid


class Visualizer:
    """可视化工具，对应 checkpoint.lua 的采样与报告功能"""

    def __init__(self, save_dir: str = './visualizations'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'samples'), exist_ok=True)

    # ------------------------------------------------------------------
    # Cover / Stego 对比图
    # ------------------------------------------------------------------
    def save_comparison_image(self, cover: torch.Tensor, stego: torch.Tensor,
                               epoch: int, idx: int = 0,
                               denorm_func=None) -> str:
        """
        保存单张 Cover（左） vs Stego（右）并排对比图，
        同时保存差异放大图（×10）。
        对应 checkpoint.lua 的 sample() 功能。
        """
        if cover.dim() == 4:
            cover = cover[idx]
            stego = stego[idx]

        if denorm_func is not None:
            cover = denorm_func(cover)
            stego = denorm_func(stego)

        cover = torch.clamp(cover.detach().float(), 0, 1)
        stego = torch.clamp(stego.detach().float(), 0, 1)

        comparison = torch.cat([cover, stego], dim=2)   # 水平拼接
        save_path = os.path.join(
            self.save_dir, 'samples', f'epoch_{epoch:03d}_sample_{idx}.png'
        )
        save_image(comparison, save_path)

        diff = torch.clamp(torch.abs(cover - stego) * 10, 0, 1)
        diff_path = os.path.join(
            self.save_dir, 'samples', f'epoch_{epoch:03d}_diff_{idx}.png'
        )
        save_image(diff, diff_path)
        return save_path

    def save_batch_grid(self, cover: torch.Tensor, stego: torch.Tensor,
                        epoch: int, nrow: int = 4) -> str:
        """保存批量对比网格（上半 Cover，下半 Stego）"""
        cover = torch.clamp(cover.detach().float(), 0, 1)
        stego = torch.clamp(stego.detach().float(), 0, 1)
        n = min(cover.shape[0], nrow * nrow)
        cover_grid = make_grid(cover[:n], nrow=nrow)
        stego_grid = make_grid(stego[:n], nrow=nrow)
        comparison = torch.cat([cover_grid, stego_grid], dim=1)  # 垂直拼接
        save_path = os.path.join(
            self.save_dir, 'samples', f'epoch_{epoch:03d}_grid.png'
        )
        save_image(comparison, save_path)
        return save_path

    # ------------------------------------------------------------------
    # 训练曲线
    # ------------------------------------------------------------------
    def plot_training_curves(self, history: dict,
                              save_name: str = 'training_curves.png') -> str:
        """
        绘制 Loss / PSNR / Bit Accuracy / BER 训练曲线（2×2 子图）。

        Args:
            history: {
                'epochs':  [1, 2, ...],
                'loss':    [...],   # 可选
                'psnr':    [...],   # 可选
                'bit_acc': [...],   # 可选，百分比
                'ber':     [...],   # 可选
            }
        """
        epochs = history.get('epochs', [])
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        _plot_if_exists(axes[0, 0], epochs, history.get('loss'),
                        'Loss', 'Training Loss', 'b-')
        _plot_if_exists(axes[0, 1], epochs, history.get('psnr'),
                        'PSNR (dB)', 'Image Quality (PSNR)', 'g-')

        if 'bit_acc' in history:
            axes[1, 0].plot(epochs, history['bit_acc'], 'r-', linewidth=2)
            axes[1, 0].axhline(y=95, color='green', linestyle='--',
                               alpha=0.6, label='95% 目标线')
            axes[1, 0].set_ylim(0, 105)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Bit Accuracy (%)')
            axes[1, 0].set_title('Message Recovery Accuracy')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()

        if 'ber' in history and any(v > 0 for v in history['ber']):
            axes[1, 1].plot(epochs, history['ber'], color='orange', linewidth=2)
            axes[1, 1].set_yscale('log')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Bit Error Rate')
            axes[1, 1].set_title('Bit Error Rate (log scale)')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Training curves saved to {save_path}")
        return save_path

    # ------------------------------------------------------------------
    # 鲁棒性曲线 ① —— 单模型 × 多噪声（训练中实时调用）
    # ------------------------------------------------------------------
    def plot_robustness_curve(self, results: dict,
                               save_name: str = 'robustness_curve.png') -> str:
        """
        单模型在多种噪声下的鲁棒性曲线。
        适合在训练过程中每 N epoch 调用一次。

        Args:
            results: {
                'dropout': {'intensities': [0.9,0.7,0.5,0.3,0.1],
                            'accuracies':  [99, 97, 92, 80, 65]},
                'gaussian': {...},
            }
        """
        styles = {
            'identity': {'color': 'gray',   'marker': 'x'},
            'dropout':  {'color': '#1f77b4','marker': 'o'},
            'cropout':  {'color': '#2ca02c','marker': 's'},
            'crop':     {'color': '#d62728','marker': '^'},
            'gaussian': {'color': '#ff7f0e','marker': 'd'},
            'jpeg':     {'color': '#9467bd','marker': 'p'},
        }

        fig, ax = plt.subplots(figsize=(10, 6))
        for noise_type, data in results.items():
            s = styles.get(noise_type, {'color': 'black', 'marker': 'o'})
            ax.plot(data['intensities'], data['accuracies'],
                    color=s['color'], marker=s['marker'],
                    linewidth=2, markersize=8,
                    label=noise_type.capitalize())

        ax.axhline(y=95, color='green', linestyle='--', alpha=0.6, label='95% 基准')
        ax.axhline(y=50, color='red',   linestyle='--', alpha=0.6, label='随机猜测 50%')
        ax.set_xlabel('Noise Intensity', fontsize=13)
        ax.set_ylabel('Bit Accuracy (%)', fontsize=13)
        ax.set_title('Robustness: Bit Accuracy vs Noise Intensity', fontsize=14)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)

        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Robustness curve saved to {save_path}")
        return save_path

    # ------------------------------------------------------------------
    # 鲁棒性曲线 ② —— 多模型 × 单噪声（对应论文图 10，训练结束后使用）
    # ------------------------------------------------------------------
    def plot_robustness_curve_paper_style(
        self, results: dict,
        save_name: str = 'robustness_paper_style.png'
    ) -> str:
        """
        论文图 10 / 图 11 风格：固定噪声类型，对比多个模型的鲁棒性。

        Args:
            results: {
                'Dropout': {                          # 噪声类型（子图标题）
                    'intensity_label': 'Keep Ratio p',
                    'intensities': [0.9, 0.7, 0.5, 0.3, 0.1],
                    'models': {
                        'Identity':   {'accuracies': [86, 86, 86, 86, 86]},
                        'Specialized':{'accuracies': [100,100, 97, 94, 83]},
                        'Combined':   {'accuracies': [100,100, 94, 93, 88]},
                    }
                },
                'Gaussian': { ... },
                'JPEG':     { ... },
            }

        使用示例（训练结束后从多个 checkpoint 加载结果）：
            results = build_robustness_results(
                models={'Identity': ckpt_identity,
                        'Specialized': ckpt_specialized,
                        'Combined': ckpt_combined},
                val_loader=val_loader, cfg=cfg, device=device
            )
            vis.plot_robustness_curve_paper_style(results)
        """
        model_styles = {
            'Identity':    {'color': '#1f77b4', 'marker': 'o', 'ls': '-'},
            'Specialized': {'color': '#ff7f0e', 'marker': '*', 'ls': '-'},
            'Combined':    {'color': '#2ca02c', 'marker': 's', 'ls': '-'},
            'Digimarc':    {'color': '#d62728', 'marker': 'D', 'ls': '--'},
        }

        noise_types = list(results.keys())
        n = len(noise_types)
        fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), sharey=False)
        if n == 1:
            axes = [axes]

        for ax, noise_type in zip(axes, noise_types):
            data = results[noise_type]
            x = data['intensities']
            for model_name, model_data in data['models'].items():
                ms = model_styles.get(model_name,
                                      {'color': 'gray', 'marker': 'o', 'ls': '-'})
                ax.plot(x, model_data['accuracies'],
                        color=ms['color'], marker=ms['marker'],
                        linestyle=ms['ls'], linewidth=2, markersize=8,
                        label=model_name)
            ax.axhline(y=95, color='gray', ls=':', alpha=0.6, lw=1)
            ax.axhline(y=50, color='red',  ls=':', alpha=0.6, lw=1)
            ax.set_xlabel(data.get('intensity_label', 'Intensity'), fontsize=11)
            ax.set_ylabel('Bit Accuracy (%)', fontsize=11)
            ax.set_title(f'{noise_type}', fontsize=13, fontweight='bold')
            ax.set_ylim(48, 102)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

        plt.suptitle('Robustness: Bit Accuracy vs Noise Intensity (Paper Fig.10)',
                     fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Paper-style robustness curve saved to {save_path}")
        return save_path

    # ------------------------------------------------------------------
    # PSNR 对比柱状图（对应论文图8上方表格）
    # ------------------------------------------------------------------
    def plot_psnr_comparison(self, models_results: dict,
                              save_name: str = 'psnr_comparison.png') -> str:
        """
        Args:
            models_results: {'Identity': 44.63, 'Dropout': 42.52, ...}
        """
        models = list(models_results.keys())
        psnrs  = list(models_results.values())
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(models, psnrs, color='steelblue', edgecolor='black')
        for bar, psnr in zip(bars, psnrs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f'{psnr:.1f}', ha='center', va='bottom', fontsize=10)
        ax.set_xlabel('Model / Noise Type', fontsize=13)
        ax.set_ylabel('PSNR (dB)', fontsize=13)
        ax.set_title('PSNR Comparison (Paper Fig.8)', fontsize=14)
        ax.grid(True, axis='y', alpha=0.3)
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return save_path

    # ------------------------------------------------------------------
    # 文字报告
    # ------------------------------------------------------------------
    def save_detailed_report(self, metrics_dict: dict, epoch: int,
                              save_name: str = 'metrics_report.txt') -> str:
        """
        追加写入指标报告，对应 checkpoint.lua 的 write_final_report。
        """
        report_path = os.path.join(self.save_dir, save_name)
        with open(report_path, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Epoch: {epoch}\n")
            f.write(f"{'='*60}\n")
            for key, value in metrics_dict.items():
                if isinstance(value, dict):
                    f.write(f"\n{key}:\n")
                    for sk, sv in value.items():
                        f.write(f"  {sk}: {sv:.6f}\n")
                elif isinstance(value, float):
                    f.write(f"{key}: {value:.6f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        return report_path


# ------------------------------------------------------------------
# 内部工具函数
# ------------------------------------------------------------------
def _plot_if_exists(ax, epochs, data, ylabel, title, style):
    if data is None:
        return
    ax.plot(epochs, data, style, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
