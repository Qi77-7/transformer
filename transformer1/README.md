# Transformer与消融实验

本项目手工实现了 Transformer 模型，并在小数据集iwslt2017上进行消融实验，包括关闭位置编码、关闭残差连接等设置。

## 硬件要求

- CPU 或 GPU（建议 CUDA）
- Python 3.8+
- PyTorch >= 2.0
- matplotlib, tqdm

安装依赖：
```bash
pip install -r requirements.txt
```
## 运行命令与随机种子（Exact Commands）
方式 1：使用脚本自动运行
```
bash scripts/run.sh
```
方式 2：手动执行（Windows CMD 版本）
```
# Baseline（正常 Transformer）
python main_ablation.py --epochs 5 --batch_size 8 --lr 5e-4 --d_model 64 --num_heads 4 --d_ff 256 --num_layers 2 --device cpu --exp_name baseline --seed 42

# NoPos（去掉位置编码）
python main_ablation.py --epochs 5 --batch_size 8 --lr 5e-4 --d_model 64 --num_heads 4 --d_ff 256 --num_layers 2 --device cpu --disable_pos_encoding --exp_name no_pos --seed 42

# NoResidual（去掉残差连接）
python main_ablation.py --epochs 5 --batch_size 8 --lr 5e-4 --d_model 64 --num_heads 4 --d_ff 256mkdir results logs
```
## 结果说明

每个实验的日志文件保存在：
```
logs/baseline.log
logs/no_pos.log
logs/no_residual.log
```
每个实验的训练曲线图保存在：
```
results/baseline_curve.png
results/no_pos_curve.png
results/no_residual_curve.png
```
模型参数（最优权重）保存在：
```
checkpoints_ablation_baseline/best_model.pt
checkpoints_ablation_no_pos/best_model.pt
checkpoints_ablation_no_residual/best_model.pt
```