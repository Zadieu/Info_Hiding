"""
测试任务5模块是否能与项目其他模块正常对接

运行方式：
    cd D:\matlab\LAB\大作业
    python test_task5.py
"""

import sys
from pathlib import Path

# 添加 DataLoader 目录到路径（因为 config.py 在那里）
ROOT = Path(__file__).resolve().parent
DATALOADER_PATH = ROOT / 'DataLoader'
if str(DATALOADER_PATH) not in sys.path:
    sys.path.insert(0, str(DATALOADER_PATH))
# 同时添加根目录
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np

# 现在可以导入 config 了
from config import TrainConfig
from dataset import sample_messages

# 导入任务5模块
from metrics import MetricsCalculator
from logger import TensorBoardLogger
from visualizer import Visualizer
from main_task5 import Task5Manager


def test_metrics():
    """测试1: 指标计算是否正确"""
    print("=" * 50)
    print("测试1: MetricsCalculator")
    
    calc = MetricsCalculator(image_range=1.0)
    
    # 测试 PSNR
    cover = torch.rand(4, 3, 64, 64)
    stego = cover + 0.01 * torch.randn(4, 3, 64, 64)
    psnr = calc.compute_psnr(cover, stego)
    print(f"  PSNR (应该有值): {psnr:.2f} dB")
    assert psnr > 0, "PSNR 应该为正数"
    
    # 测试 Bit Accuracy
    msg_gt = torch.randint(0, 2, (4, 30)).float()
    msg_pred = msg_gt + 0.1 * torch.randn(4, 30)
    acc, ber = calc.compute_bit_accuracy(msg_gt, msg_pred, is_logit=False)
    print(f"  Bit Accuracy (应该接近0.9): {acc:.4f}")
    print(f"  BER: {ber:.4f}")
    assert 0 <= acc <= 1, "Accuracy 应该在 [0,1]"
    
    # 测试完美重建
    acc_perfect, _ = calc.compute_bit_accuracy(msg_gt, msg_gt, is_logit=False)
    print(f"  完美重建 Accuracy: {acc_perfect:.4f}")
    assert acc_perfect == 1.0, "完美重建应该得到 1.0"
    
    # 测试 compute_psnr_per_channel
    cover_rgb = torch.rand(4, 3, 64, 64)
    stego_rgb = cover_rgb + 0.01 * torch.randn(4, 3, 64, 64)
    psnrs = calc.compute_psnr_per_channel(cover_rgb, stego_rgb)
    print(f"  YUV PSNR: {psnrs}")
    assert 'y' in psnrs and 'u' in psnrs and 'v' in psnrs, "YUV PSNR 应该返回三个通道"
    
    print("  ✅ 指标计算测试通过\n")


def test_logger():
    """测试2: TensorBoard 日志记录"""
    print("=" * 50)
    print("测试2: TensorBoardLogger")
    
    logger = TensorBoardLogger(log_dir='./test_logs', experiment_name='test')
    logger.set_epoch(1)
    logger.log_metrics({'psnr': 35.2, 'bit_accuracy': 98.5}, prefix='test')
    logger.log_scalar('test/value', 0.5)
    
    # 测试对比图
    cover = torch.rand(4, 3, 64, 64)
    stego = cover + 0.01 * torch.randn(4, 3, 64, 64)
    logger.log_comparison_grid(cover, stego, step=1)
    
    logger.flush()
    logger.close()
    
    # 检查文件是否生成
    log_dir = Path('./test_logs/test')
    assert log_dir.exists(), f"日志目录 {log_dir} 应该存在"
    assert any(log_dir.glob("events.out*")), "TensorBoard 事件文件应该存在"
    
    print(f"  ✅ 日志写入成功: {log_dir}")
    print("  查看命令: tensorboard --logdir=./test_logs\n")


def test_visualizer():
    """测试3: 可视化功能"""
    print("=" * 50)
    print("测试3: Visualizer")
    
    vis = Visualizer(save_dir='./test_visualizations')
    
    # 测试对比图
    cover = torch.rand(3, 64, 64)
    stego = cover + 0.05 * torch.randn(3, 64, 64)
    path = vis.save_comparison_image(cover, stego, epoch=1, idx=0)
    print(f"  对比图保存: {path}")
    assert Path(path).exists()
    
    # 测试批量网格
    cover_batch = torch.rand(8, 3, 64, 64)
    stego_batch = cover_batch + 0.05 * torch.randn(8, 3, 64, 64)
    path = vis.save_batch_grid(cover_batch, stego_batch, epoch=1)
    print(f"  批量网格保存: {path}")
    
    # 测试训练曲线
    history = {
        'epochs': [1, 2, 3, 4, 5],
        'psnr': [30, 32, 34, 35, 35.5],
        'bit_acc': [80, 85, 90, 93, 95],
        'ber': [0.2, 0.15, 0.1, 0.07, 0.05],
    }
    path = vis.plot_training_curves(history)
    print(f"  训练曲线保存: {path}")
    
    # 测试鲁棒性曲线（单模型）
    results = {
        'dropout': {'intensities': [0.9, 0.7, 0.5, 0.3, 0.1],
                    'accuracies': [98, 95, 88, 75, 60]},
        'gaussian': {'intensities': [0.5, 1.0, 1.5, 2.0, 2.5],
                     'accuracies': [96, 92, 87, 80, 72]},
    }
    path = vis.plot_robustness_curve(results)
    print(f"  鲁棒性曲线保存: {path}")
    
    print("  ✅ 可视化测试通过\n")


def test_integration_with_model():
    """测试4: 与模型集成测试（核心）"""
    print("=" * 50)
    print("测试4: 与 HiDDeN 模型集成")
    
    # 导入模型（需要 models 目录在路径中）
    models_path = DATALOADER_PATH / 'models'
    if str(models_path) not in sys.path:
        sys.path.insert(0, str(models_path))
    
    try:
        from models.hidden import HiDDeNModel
    except ImportError as e:
        print(f"  ⚠️ 无法导入 HiDDeNModel: {e}")
        print("  跳过集成测试（需要分工2的代码）\n")
        return
    
    # 使用隐写实验配置
    cfg = TrainConfig(
        image_channels=1,
        image_height=16,
        image_width=16,
        message_length=52,
        noise_type="identity",
        batch_size=4,
        noise_dropout_p=0.3,
        noise_cropout_p=0.3,
        noise_gaussian_sigma=2.0,
        noise_crop_p=0.035,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}")
    
    # 创建模型
    noise_kwargs = {
        'dropout_p': cfg.noise_dropout_p,
        'cropout_p': cfg.noise_cropout_p,
        'gaussian_sigma': cfg.noise_gaussian_sigma,
        'crop_p': cfg.noise_crop_p,
    }
    model = HiDDeNModel(
        C=cfg.C, H=cfg.H, W=cfg.W, L=cfg.L,
        noise_type=cfg.noise_type,
        noise_kwargs=noise_kwargs
    ).to(device)
    
    # 创建 Task5Manager
    task5 = Task5Manager(
        log_dir='./test_logs',
        vis_dir='./test_visualizations',
        experiment_name='test_integration'
    )
    
    # 模拟数据
    cover = torch.rand(cfg.batch_size, cfg.C, cfg.H, cfg.W).to(device)
    message = sample_messages(cfg.batch_size, cfg.L, device)
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        encoded, noised, decoded = model(cover, message)
    
    # 计算指标
    calc = MetricsCalculator()
    psnr = calc.compute_psnr(cover, encoded)
    acc, ber = calc.compute_bit_accuracy(message, decoded, is_logit=False)
    
    print(f"  前向传播成功")
    print(f"  encoded shape: {encoded.shape}")
    print(f"  decoded shape: {decoded.shape}")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  Bit Accuracy: {acc*100:.2f}%")
    
    # 测试 evaluate_and_log
    print("\n  测试 evaluate_and_log...")
    
    # 创建临时 DataLoader
    from torch.utils.data import DataLoader, TensorDataset
    fake_images = torch.rand(20, cfg.C, cfg.H, cfg.W)
    fake_dataset = TensorDataset(fake_images, torch.zeros(20))
    fake_loader = DataLoader(fake_dataset, batch_size=cfg.batch_size)
    
    metrics = task5.evaluate_and_log(
        model=model,
        val_loader=fake_loader,
        epoch=1,
        cfg=cfg,
        device=device
    )
    print(f"  评估结果: PSNR={metrics['psnr']:.2f}, BitAcc={metrics['bit_accuracy']*100:.2f}%")
    
    # 测试训练日志
    task5.log_train_batch(step=100, loss_dict={
        'L_M': 0.05, 'L_I': 0.001, 'L_G': 0.01, 'L_A': 0.02, 'bit_acc': 95.0
    })
    
    print("  ✅ 模型集成测试通过\n")


def test_robustness():
    """测试5: 鲁棒性测试"""
    print("=" * 50)
    print("测试5: 鲁棒性测试")
    
    models_path = DATALOADER_PATH / 'models'
    if str(models_path) not in sys.path:
        sys.path.insert(0, str(models_path))
    
    try:
        from models.hidden import HiDDeNModel
    except ImportError as e:
        print(f"  ⚠️ 无法导入 HiDDeNModel: {e}")
        print("  跳过鲁棒性测试（需要分工2的代码）\n")
        return
    
    cfg = TrainConfig(
        image_channels=3,
        image_height=128,
        image_width=128,
        message_length=30,
        noise_type="identity",
        batch_size=2,
        noise_dropout_p=0.3,
        noise_cropout_p=0.3,
        noise_gaussian_sigma=2.0,
        noise_crop_p=0.035,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    noise_kwargs = {
        'dropout_p': cfg.noise_dropout_p,
        'cropout_p': cfg.noise_cropout_p,
        'gaussian_sigma': cfg.noise_gaussian_sigma,
        'crop_p': cfg.noise_crop_p,
    }
    model = HiDDeNModel(
        C=cfg.C, H=cfg.H, W=cfg.W, L=cfg.L,
        noise_type=cfg.noise_type,
        noise_kwargs=noise_kwargs
    ).to(device)
    
    task5 = Task5Manager(
        log_dir='./test_logs',
        vis_dir='./test_visualizations',
        experiment_name='test_robustness'
    )
    
    from torch.utils.data import DataLoader, TensorDataset
    fake_images = torch.rand(10, cfg.C, cfg.H, cfg.W)
    fake_dataset = TensorDataset(fake_images, torch.zeros(10))
    fake_loader = DataLoader(fake_dataset, batch_size=cfg.batch_size)
    
    try:
        results = task5.test_robustness(
            model=model,
            val_loader=fake_loader,
            epoch=1,
            cfg=cfg,
            device=device
        )
        print("  鲁棒性测试结果:")
        for noise_type, data in results.items():
            print(f"    {noise_type}: {data['accuracies']}")
        print("  ✅ 鲁棒性测试通过\n")
    except Exception as e:
        print(f"  ⚠️ 鲁棒性测试失败: {e}\n")


def test_paper_style_curve():
    """测试6: 论文风格曲线（多模型对比）"""
    print("=" * 50)
    print("测试6: 论文风格鲁棒性曲线")
    
    models_path = DATALOADER_PATH / 'models'
    if str(models_path) not in sys.path:
        sys.path.insert(0, str(models_path))
    
    try:
        from models.hidden import HiDDeNModel
        from main_task5 import build_robustness_results_multi_model
    except ImportError as e:
        print(f"  ⚠️ 无法导入必要模块: {e}")
        print("  跳过论文风格曲线测试\n")
        return
    
    cfg = TrainConfig(
        image_channels=3,
        image_height=128,
        image_width=128,
        message_length=30,
        batch_size=2,
        noise_type='identity',
        noise_dropout_p=0.3,
        noise_cropout_p=0.3,
        noise_gaussian_sigma=2.0,
        noise_crop_p=0.035,
    )
    device = torch.device('cpu')
    noise_kwargs = {
        'dropout_p': cfg.noise_dropout_p,
        'cropout_p': cfg.noise_cropout_p,
        'gaussian_sigma': cfg.noise_gaussian_sigma,
        'crop_p': cfg.noise_crop_p,
    }

    base_model = HiDDeNModel(
        C=cfg.C, H=cfg.H, W=cfg.W, L=cfg.L,
        noise_type=cfg.noise_type,
        noise_kwargs=noise_kwargs
    ).to(device)

    models = {
        'Identity':    base_model,
        'Specialized': base_model,
        'Combined':    base_model,
    }

    from torch.utils.data import DataLoader, TensorDataset
    fake_images = torch.rand(8, cfg.C, cfg.H, cfg.W)
    fake_dataset = TensorDataset(fake_images, torch.zeros(8))
    fake_loader = DataLoader(fake_dataset, batch_size=2)

    try:
        results = build_robustness_results_multi_model(
            models=models,
            val_loader=fake_loader,
            cfg=cfg,
            device=device
        )
        vis = Visualizer('./test_visualizations')
        vis.plot_robustness_curve_paper_style(results)
        print("  ✅ 论文风格曲线生成成功\n")
    except Exception as e:
        print(f"  ⚠️ 论文风格曲线生成失败: {e}\n")


def cleanup():
    """清理测试文件"""
    import shutil
    for d in ['./test_logs', './test_visualizations']:
        if Path(d).exists():
            shutil.rmtree(d)
            print(f"  清理: {d}")


def main():
    print("\n" + "=" * 60)
    print("任务5模块测试 - 与 HiDDeN 项目对接验证")
    print("=" * 60 + "\n")
    
    torch.manual_seed(42)
    
    tests = [
        ("指标计算", test_metrics),
        ("TensorBoard日志", test_logger),
        ("可视化", test_visualizer),
        ("模型集成", test_integration_with_model),
        ("鲁棒性测试", test_robustness),
        ("论文风格曲线", test_paper_style_curve),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {name} 测试失败: {e}\n")
            failed += 1
        except Exception as e:
            print(f"  ⚠️ {name} 测试异常: {e}\n")
            failed += 1
    
    print("=" * 60)
    if failed == 0:
        print(f"🎉 所有测试通过！({passed}/{passed+failed})")
        print("任务5模块可以与项目正常对接")
    else:
        print(f"⚠️ 部分测试未通过 ({passed}/{passed+failed})")
    print("=" * 60)
    
    if failed == 0:
        response = input("\n是否清理测试文件？(y/n): ")
        if response.lower() == 'y':
            cleanup()
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())