# ==================== main_task5.py ====================
"""
任务5主程序：整合 metrics / logger / visualizer

修复记录:
  [Fix-1] evaluate_and_log: 模型调用接口从 model.encode/decode() 改为
          model(cover, message) → (encoded, noised, decoded)，与 train.py 一致
  [Fix-2] evaluate_and_log: dataloader 返回 (image, label)，message 需要用
          sample_messages() 生成，不能从 dataloader 中取
  [Fix-3] test_robustness: model.set_noise_layer() / model.apply_noise() 不存在，
          改用 hidden_repro.noise_layers 中的 build_noise_layer() 直接构造噪声层，
          用 model.encoder / model.decoder 分步推断
  [Fix-6] 统一调 set_epoch(epoch) 而不是 set_step，已在 logger.py 中修复同步逻辑
"""

import torch
from config import TrainConfig
from dataset import sample_messages

# 任务5模块
from metrics import MetricsCalculator
from logger import TensorBoardLogger
from visualizer import Visualizer

# 噪声层（任务4同学的模块）
from noise_layers import build_noise_layer


class Task5Manager:
    """任务5管理器：PSNR / Bit Accuracy / TensorBoard / 对比图 / 鲁棒性曲线"""

    def __init__(self, log_dir: str = './logs', vis_dir: str = './visualizations',
                 experiment_name: str = 'hidden'):
        self.metrics = MetricsCalculator(image_range=1.0)
        self.logger = TensorBoardLogger(log_dir, experiment_name)
        self.visualizer = Visualizer(vis_dir)

        self.history = {
            'epochs': [], 'loss': [], 'psnr': [], 'bit_acc': [], 'ber': []
        }
        self.best_bit_acc = 0.0
        self.best_epoch = 0

    # ------------------------------------------------------------------
    # 主评估入口（每 epoch 末调用）
    # ------------------------------------------------------------------
    def evaluate_and_log(self, model, val_loader, epoch: int, cfg: TrainConfig,
                         device: torch.device) -> dict:
        """
        在验证集上评估模型，记录所有指标，保存对比图。

        Args:
            model:      HiDDeNModel 实例（来自 train.py）
            val_loader: 验证集 DataLoader（来自 build_loaders）
            epoch:      当前 epoch 编号
            cfg:        TrainConfig，用于读取 cfg.L（消息长度）
            device:     torch.device

        Returns:
            dict: {'psnr', 'bit_accuracy', 'ber'}
        """
        model.eval()

        total_psnr = 0.0
        total_bit_acc = 0.0
        total_ber = 0.0
        num_batches = 0
        sample_cover = sample_stego = None  # 用于可视化的第一个 batch

        with torch.no_grad():
            # [Fix-2] dataloader 只返回 (image, label)，message 要单独生成
            for batch_idx, (images, _) in enumerate(val_loader):
                images = images.to(device)
                B = images.shape[0]

                # 生成随机消息（与 train.py 中的做法完全一致）
                messages = sample_messages(B, cfg.L, device)

                # [Fix-1] 正确的模型调用方式：返回 (encoded, noised, decoded)
                encoded, noised, decoded = model(images, messages)

                # 计算指标
                psnr = self.metrics.compute_psnr(images, encoded)
                # [Fix-5] is_logit=False，decoder 输出已是 [0,1]，与 losses.py 一致
                bit_acc, ber = self.metrics.compute_bit_accuracy(
                    messages, decoded, is_logit=False
                )

                total_psnr += psnr
                total_bit_acc += bit_acc
                total_ber += ber
                num_batches += 1

                if batch_idx == 0:
                    sample_cover = images.detach().cpu()
                    sample_stego = encoded.detach().cpu()

        # 平均
        avg_psnr    = total_psnr    / num_batches
        avg_bit_acc = total_bit_acc / num_batches
        avg_ber     = total_ber     / num_batches

        # [Fix-6] set_epoch 会同步更新 self.step，log_metrics 横轴正常
        self.logger.set_epoch(epoch)
        self.logger.log_metrics({
            'psnr':         avg_psnr,
            'bit_accuracy': avg_bit_acc * 100,   # TensorBoard 显示百分比
            'ber':          avg_ber,
        }, prefix='test')

        # 保存对比图
        if sample_cover is not None:
            self.visualizer.save_comparison_image(sample_cover, sample_stego, epoch)
            self.visualizer.save_batch_grid(sample_cover, sample_stego, epoch)
            self.logger.log_comparison_grid(sample_cover, sample_stego, step=epoch)

        # 更新历史
        self.history['epochs'].append(epoch)
        self.history['psnr'].append(avg_psnr)
        self.history['bit_acc'].append(avg_bit_acc * 100)
        self.history['ber'].append(avg_ber)

        if avg_bit_acc > self.best_bit_acc:
            self.best_bit_acc = avg_bit_acc
            self.best_epoch = epoch
            print(f"  [Task5] 新最佳 Bit Accuracy: {self.best_bit_acc*100:.2f}% @ epoch {epoch}")

        print(f"  [Task5] epoch={epoch}  PSNR={avg_psnr:.2f}dB  "
              f"BitAcc={avg_bit_acc*100:.2f}%  BER={avg_ber:.4f}")

        return {'psnr': avg_psnr, 'bit_accuracy': avg_bit_acc, 'ber': avg_ber}

    # ------------------------------------------------------------------
    # 记录训练 batch 指标（在 train.py 的 batch 循环里调用）
    # ------------------------------------------------------------------
    def log_train_batch(self, step: int, loss_dict: dict) -> None:
        """
        记录单个 batch 的训练损失。

        Args:
            step:      全局训练步数（epoch * steps_per_epoch + batch_idx）
            loss_dict: {'L_M': ..., 'L_I': ..., 'L_G': ..., 'L_A': ..., 'bit_acc': ...}

        在 train.py 的 batch 循环末尾调用：
            task5.log_train_batch(global_step, {
                'L_M': lm.item(), 'L_I': li.item(),
                'L_G': lg.item(), 'L_A': la.item(),
                'bit_acc': bit_acc_val * 100,
            })
        """
        self.logger.set_step(step)
        self.logger.log_metrics(loss_dict, prefix='train')

    # ------------------------------------------------------------------
    # 鲁棒性测试（建议在训练结束后调用，或每隔 N epoch 调一次）
    # ------------------------------------------------------------------
    def test_robustness(self, model, val_loader, epoch: int, cfg: TrainConfig,
                        device: torch.device) -> dict:
        """
        测试当前模型在不同噪声强度下的鲁棒性，生成对应图10风格的曲线。

        [Fix-3] 直接用 build_noise_layer() 构造噪声层，通过
                model.encoder / model.decoder 分步推断，
                不再调用不存在的 model.set_noise_layer() / model.apply_noise()

        Returns:
            results: 可直接传给 visualizer.plot_robustness_curve()
        """
        # 论文图10的噪声类型与强度配置
        NOISE_CONFIGS = {
            'dropout':  {'param': 'p',     'values': [0.9, 0.7, 0.5, 0.3, 0.1],
                         'kwargs_key': 'p',  'label': 'Keep Ratio p'},
            'cropout':  {'param': 'p',     'values': [0.9, 0.7, 0.5, 0.3, 0.1],
                         'kwargs_key': 'p',  'label': 'Keep Ratio p'},
            'gaussian': {'param': 'sigma', 'values': [0.5, 1.0, 1.5, 2.0, 3.0],
                         'kwargs_key': 'sigma', 'label': 'Blur σ'},
            'jpeg':     {'param': 'q',     'values': [90, 70, 50, 30, 10],
                         'kwargs_key': None, 'label': 'Quality Q (simulated)'},
        }

        model.eval()
        results = {}

        for noise_name, config in NOISE_CONFIGS.items():
            intensities = config['values']
            accuracies = []

            for intensity in intensities:
                # 构造对应强度的噪声层
                # jpeg 系列没有 quality 参数，用 JpegMaskNoise 固定近似
                if noise_name == 'jpeg':
                    noise_layer = build_noise_layer('jpeg_mask')
                else:
                    kwargs = {config['kwargs_key']: intensity}
                    noise_layer = build_noise_layer(noise_name, **kwargs)
                noise_layer = noise_layer.to(device).eval()

                total_acc = 0.0
                n_batches = 0

                with torch.no_grad():
                    for images, _ in val_loader:
                        images = images.to(device)
                        B = images.shape[0]
                        messages = sample_messages(B, cfg.L, device)

                        # 分步推断：encoder → 外部噪声层 → decoder
                        encoded = model.encoder(images, messages)
                        noised  = noise_layer(images, encoded)  # cover, encoded

                        # CropNoise 可能改变空间尺寸，decoder 用全局平均池化处理
                        decoded = model.decoder(noised)

                        acc, _ = self.metrics.compute_bit_accuracy(
                            messages, decoded, is_logit=False
                        )
                        total_acc += acc
                        n_batches += 1

                avg_acc = (total_acc / n_batches) * 100  # 转为百分比
                accuracies.append(avg_acc)
                print(f"  [Robustness] {noise_name} intensity={intensity:.2f}: "
                      f"BitAcc={avg_acc:.1f}%")

            results[noise_name] = {
                'intensities': intensities,
                'accuracies': accuracies,
            }

        # 绘制单模型鲁棒性曲线
        self.visualizer.plot_robustness_curve(
            results, save_name=f'robustness_epoch_{epoch:03d}.png'
        )
        return results

    # ------------------------------------------------------------------
    # 训练结束后的汇总
    # ------------------------------------------------------------------
    def finalize(self) -> None:
        """训练结束后调用：生成训练曲线 + 最终报告 + 关闭 logger"""
        self.visualizer.plot_training_curves(self.history)
        if self.history['epochs']:
            final = {
                'best_bit_accuracy': self.best_bit_acc,
                'best_epoch':        float(self.best_epoch),
                'final_psnr':        self.history['psnr'][-1],
                'final_bit_acc_%':   self.history['bit_acc'][-1],
                'final_ber':         self.history['ber'][-1],
            }
            self.visualizer.save_detailed_report(
                final, self.history['epochs'][-1], 'final_report.txt'
            )
        self.logger.close()
        print(f"\n[Task5] 完成！最佳 BitAcc = {self.best_bit_acc*100:.2f}% "
              f"@ epoch {self.best_epoch}")


# ------------------------------------------------------------------
# 论文图10：多模型对比（训练结束后，从多个 checkpoint 加载结果使用）
# ------------------------------------------------------------------
def build_robustness_results_multi_model(
    models: dict,          # {'Identity': model_obj, 'Specialized': model_obj, ...}
    val_loader,
    cfg: TrainConfig,
    device: torch.device,
    calc: MetricsCalculator = None,
) -> dict:
    """
    构造 plot_robustness_curve_paper_style 所需的 results 字典。

    使用场景（训练全部结束后）：
        from main_task5 import build_robustness_results_multi_model
        from visualizer import Visualizer

        models = {
            'Identity':    load_model('checkpoints/identity/epoch_200.pt', cfg, device),
            'Specialized': load_model('checkpoints/dropout/epoch_200.pt', cfg, device),
            'Combined':    load_model('checkpoints/combined/epoch_400.pt', cfg, device),
        }
        results = build_robustness_results_multi_model(models, val_loader, cfg, device)
        vis = Visualizer('./visualizations')
        vis.plot_robustness_curve_paper_style(results)
    """
    if calc is None:
        calc = MetricsCalculator()

    NOISE_CONFIGS = {
        'Dropout':  {'noise_name': 'dropout', 'kwargs_key': 'p',
                     'values': [0.9, 0.7, 0.5, 0.3, 0.1],
                     'intensity_label': 'Keep Ratio p'},
        'Cropout':  {'noise_name': 'cropout', 'kwargs_key': 'p',
                     'values': [0.9, 0.7, 0.5, 0.3, 0.1],
                     'intensity_label': 'Keep Ratio p'},
        'Gaussian': {'noise_name': 'gaussian', 'kwargs_key': 'sigma',
                     'values': [0.5, 1.0, 1.5, 2.0, 3.0],
                     'intensity_label': 'Blur σ'},
        'JPEG':     {'noise_name': 'jpeg_mask', 'kwargs_key': None,
                     'values': [1],   # jpeg_mask 无强度参数，只测一个点
                     'intensity_label': 'Simulated JPEG'},
    }

    results = {}
    for display_name, cfg_item in NOISE_CONFIGS.items():
        intensities = cfg_item['values']
        results[display_name] = {
            'intensity_label': cfg_item['intensity_label'],
            'intensities': intensities,
            'models': {},
        }
        for model_name, model in models.items():
            model.eval()
            accs = []
            for intensity in intensities:
                if cfg_item['kwargs_key'] is None:
                    noise_layer = build_noise_layer(cfg_item['noise_name'])
                else:
                    noise_layer = build_noise_layer(
                        cfg_item['noise_name'],
                        **{cfg_item['kwargs_key']: intensity}
                    )
                noise_layer = noise_layer.to(device).eval()

                total_acc, n = 0.0, 0
                with torch.no_grad():
                    for images, _ in val_loader:
                        images   = images.to(device)
                        messages = sample_messages(images.shape[0], cfg.L, device)
                        encoded  = model.encoder(images, messages)
                        noised   = noise_layer(images, encoded)
                        decoded  = model.decoder(noised)
                        acc, _   = calc.compute_bit_accuracy(messages, decoded)
                        total_acc += acc
                        n += 1
                accs.append((total_acc / n) * 100)
            results[display_name]['models'][model_name] = {'accuracies': accs}

    return results


# ------------------------------------------------------------------
# 接入 train.py 的最简模板
# ------------------------------------------------------------------
"""
在 train.py 的 main() 中接入 Task5（只需改 5 处）：

  from main_task5 import Task5Manager    # ← 新增 1

  task5 = Task5Manager(                  # ← 新增 2（在 for epoch 循环前）
      log_dir='logs', vis_dir='visualizations',
      experiment_name=cfg.experiment_name
  )
  global_step = 0

  for epoch in range(start_epoch, cfg.num_epochs + 1):
      # 训练阶段（在 batch 循环末尾加一行）
      for step, (cover, _) in enumerate(train_loader, start=1):
          ...（原有训练代码）...
          task5.log_train_batch(global_step, {   # ← 新增 3
              'L_M': lm.item(), 'L_I': li.item(),
              'L_G': lg.item(), 'L_A': la.item(),
              'bit_acc': bit_accuracy(message, decoded) * 100,
          })
          global_step += 1

      # 验证阶段（在 logging.info 之后加）
      task5.evaluate_and_log(                    # ← 新增 4
          model, val_loader, epoch, cfg, device
      )

  task5.finalize()                               # ← 新增 5（训练循环结束后）
"""
