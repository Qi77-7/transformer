#!/bin/bash

# 创建结果与日志文件夹
mkdir -p results logs

# 基线实验
python main_ablation.py --epochs 5 --batch_size 8 --lr 5e-4 --d_model 64 --num_heads 4 --d_ff 256 --num_layers 2 --device cpu --exp_name baseline --seed 42 > logs/baseline.log 2>&1

# 无位置编码实验
python main_ablation.py --epochs 5 --batch_size 8 --lr 5e-4 --d_model 64 --num_heads 4 --d_ff 256 --num_layers 2 --device cpu --disable_pos_encoding --exp_name no_pos --seed 42 > logs/no_pos.log 2>&1

# 无残差连接实验
python main_ablation.py --epochs 5 --batch_size 8 --lr 5e-4 --d_model 64 --num_heads 4 --d_ff 256 --num_layers 2 --device cpu --disable_residual --exp_name no_residual --seed 42 > logs/no_residual.log 2>&1

echo "✅ All experiments finished! Logs saved in logs/ and results saved in results/"
