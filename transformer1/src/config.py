"""
配置模块 - 作业要求：提供清晰的超参数设置
用于实验设置和消融实验
"""

# 基础模型配置 - 作业要求的超参数表
base_model_config = {
    'vocab_size': 10000,      # 词汇表大小
    'd_model': 128,           # 模型维度 - 作业示例值
    'num_heads': 4,           # 注意力头数 - 作业示例值
    'd_ff': 512,              # 前馈网络维度 - 作业示例值
    'num_layers': 2,          # 层数 - 作业示例值
    'max_len': 512,           # 最大序列长度
    'dropout': 0.1,           # dropout率
}

# CPU优化配置 - 减小模型规模以适应CPU训练
cpu_model_config = {
    'vocab_size': 5000,       # 减小词汇表
    'd_model': 64,            # 减小模型维度
    'num_heads': 2,           # 减少注意力头
    'd_ff': 256,              # 减小前馈网络
    'num_layers': 2,          # 保持2层
    'max_len': 128,           # 减小序列长度
    'dropout': 0.1,
}

# 编码器-解码器配置 - 用于完整Transformer
seq2seq_model_config = {
    'src_vocab_size': 8000,   # 源语言词汇表
    'tgt_vocab_size': 8000,   # 目标语言词汇表
    'd_model': 128,
    'num_heads': 4,
    'd_ff': 512,
    'num_encoder_layers': 3,  # 编码器层数
    'num_decoder_layers': 3,  # 解码器层数
    'max_len': 512,
    'dropout': 0.1,
}

# 训练配置 - 作业要求的训练参数
train_config = {
    'epochs': 50,             # 训练轮数
    'batch_size': 32,         # 批次大小 - 作业示例值
    'learning_rate': 3e-4,    # 学习率 - 作业示例值
    'warmup_steps': 4000,     # 学习率warmup步数
    'clip_grad': 1.0,         # 梯度裁剪
    'seq_len': 128,           # 序列长度
}

# CPU优化训练配置
cpu_train_config = {
    'epochs': 20,             # 减少训练轮数
    'batch_size': 8,          # 减小批次大小
    'learning_rate': 5e-4,    # 稍大学习率
    'warmup_steps': 1000,     # 减少warmup
    'clip_grad': 1.0,
    'seq_len': 64,            # 减小序列长度
}

# 数据集配置
dataset_config = {
    'tiny_shakespeare': {
        'name': 'tiny_shakespeare',
        'task': 'lm',  # 语言建模
    },
    'iwslt2017': {
        'name': 'iwslt2017',
        'task': 'translation',  # 机器翻译
    }
}