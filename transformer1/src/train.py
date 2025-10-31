"""
训练模块 - 作业要求：实现训练稳定性技巧（学习率调度、梯度裁剪、AdamW）、
参数统计、模型保存/加载、训练曲线可视化
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import time
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F


class Trainer:
    """训练器类 - 实现作业要求的训练功能"""

    def __init__(self, model, train_loader, val_loader, vocab_size, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.vocab_size = vocab_size

        # 训练统计 - 用于绘制训练曲线
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

    def train_epoch(self, optimizer, scheduler, clip_grad=1.0):
        """训练一个epoch - 包含梯度裁剪"""
        self.model.train()
        total_loss = 0
        total_tokens = 0

        pbar = tqdm(self.train_loader, desc="训练")
        for batch_idx, (input_ids, target_ids) in enumerate(pbar):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            optimizer.zero_grad()

            # 前向传播 - 语言建模任务
            logits = self.model(input_ids, input_ids)

            # 计算损失 - 忽略padding token
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                target_ids.view(-1),
                ignore_index=0  # 忽略padding
            )

            # 反向传播
            loss.backward()

            # 梯度裁剪 - 作业要求的稳定性技巧
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)

            optimizer.step()
            scheduler.step()

            # 统计信息
            total_loss += loss.item() * input_ids.size(0) * input_ids.size(1)
            total_tokens += (target_ids != 0).sum().item()

            current_loss = total_loss / total_tokens if total_tokens > 0 else 0
            pbar.set_postfix({'loss': f'{current_loss:.4f}'})

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        return avg_loss

    def validate(self):
        """验证函数"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for input_ids, target_ids in self.val_loader:
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)

                logits = self.model(input_ids, input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, self.vocab_size),
                    target_ids.view(-1),
                    ignore_index=0
                )

                total_loss += loss.item() * input_ids.size(0) * input_ids.size(1)
                total_tokens += (target_ids != 0).sum().item()

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        return avg_loss

    def train(self, epochs, lr=3e-4, warmup_steps=4000, clip_grad=1.0, save_dir='checkpoints'):
        """完整训练流程 - 实现作业所有训练要求"""
        os.makedirs(save_dir, exist_ok=True)

        # 优化器 - 使用AdamW（作业要求）
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)

        # 学习率调度 - Transformer专用（warmup + 衰减）
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return max(0.0, float(warmup_steps) ** 0.5 * float(step) ** -0.5)

        scheduler = LambdaLR(optimizer, lr_lambda)

        # 参数统计 - 作业要求
        print(f"开始训练，参数数量: {self.model.get_num_params():,}")
        print(f"训练设备: {self.device}")

        best_val_loss = float('inf')

        for epoch in range(epochs):
            start_time = time.time()

            # 训练一个epoch
            train_loss = self.train_epoch(optimizer, scheduler, clip_grad)

            # 验证
            val_loss = self.validate()

            # 计算困惑度
            train_ppl = math.exp(train_loss)
            val_ppl = math.exp(val_loss)

            # 记录统计信息
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.learning_rates.append(optimizer.param_groups[0]['lr'])

            epoch_time = time.time() - start_time

            print(f'Epoch {epoch + 1:03d} | Time: {epoch_time:.2f}s | '
                  f'LR: {self.learning_rates[-1]:.6f} | '
                  f'Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f} | '
                  f'Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}')

            # 模型保存/加载 - 作业要求
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'vocab_size': self.vocab_size
                }, os.path.join(save_dir, 'best_model.pt'))

            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pt'))

    def plot_training_curves(self, save_path='training_curves.png'):
        """绘制训练曲线 - 作业要求的结果可视化"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # 损失曲线
        ax1.plot(self.train_losses, label='训练损失')
        ax1.plot(self.val_losses, label='验证损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('训练和验证损失')
        ax1.legend()
        ax1.grid(True)

        # 学习率曲线
        ax2.plot(self.learning_rates, color='red')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('学习率变化')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class Seq2SeqTrainer:
    """序列到序列任务训练器"""

    def __init__(self, model, train_loader, val_loader, tgt_vocab_size, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.tgt_vocab_size = tgt_vocab_size

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, optimizer, scheduler, clip_grad=1.0):
        """序列到序列训练epoch"""
        self.model.train()
        total_loss = 0
        total_tokens = 0

        pbar = tqdm(self.train_loader, desc="训练")
        for batch_idx, (src, tgt) in enumerate(pbar):
            src, tgt = src.to(self.device), tgt.to(self.device)

            # 序列到序列：输入是src和tgt[:-1]，目标是tgt[1:]
            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            optimizer.zero_grad()

            logits = self.model(src, tgt_input)

            # 改成 reshape，避免 view 出错
            loss = F.cross_entropy(
                logits.reshape(-1, self.tgt_vocab_size),
                tgt_target.reshape(-1),
                ignore_index=0
            )

            loss.backward()

            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * tgt_input.size(0) * tgt_input.size(1)
            total_tokens += (tgt_target != 0).sum().item()

            current_loss = total_loss / total_tokens if total_tokens > 0 else 0
            pbar.set_postfix({'loss': f'{current_loss:.4f}'})

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        return avg_loss

    def validate(self):
        """序列到序列验证"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for src, tgt in self.val_loader:
                src, tgt = src.to(self.device), tgt.to(self.device)
                tgt_input = tgt[:, :-1]
                tgt_target = tgt[:, 1:]

                logits = self.model(src, tgt_input)
                loss = F.cross_entropy(
                    logits.reshape(-1, self.tgt_vocab_size),
                    tgt_target.reshape(-1),
                    ignore_index=0
                )

                total_loss += loss.item() * tgt_input.size(0) * tgt_input.size(1)
                total_tokens += (tgt_target != 0).sum().item()

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        return avg_loss

    def train(self, epochs, lr=3e-4, warmup_steps=4000, clip_grad=1.0, save_dir='checkpoints'):
        """序列到序列训练"""
        os.makedirs(save_dir, exist_ok=True)

        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return max(0.0, float(warmup_steps) ** 0.5 * float(step) ** -0.5)

        scheduler = LambdaLR(optimizer, lr_lambda)

        print(f"开始Seq2Seq训练，参数数量: {self.model.get_num_params():,}")

        best_val_loss = float('inf')

        for epoch in range(epochs):
            start_time = time.time()

            train_loss = self.train_epoch(optimizer, scheduler, clip_grad)
            val_loss = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            epoch_time = time.time() - start_time

            print(f'Epoch {epoch + 1:03d} | Time: {epoch_time:.2f}s | '
                  f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, os.path.join(save_dir, 'best_model.pt'))

    def plot_training_curves(self, save_path='training_curves_seq2seq.png'):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(self.train_losses, label='Train Loss', marker='o')
        plt.plot(self.val_losses, label='Val Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss (Seq2Seq)')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

