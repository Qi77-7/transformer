"""
主训练脚本 - 作业要求：提供运行命令和硬件要求
支持语言建模和序列到序列任务
"""
"""
import torch
import argparse
from src.model import Transformer
from src.data_loader import load_tiny_shakespeare_local, load_iwslt2017_local
from src.train import Trainer, Seq2SeqTrainer
from src.config import base_model_config, train_config, seq2seq_model_config


def main():
    # 命令行参数解析 - 作业要求：提供重现实验的exact命令行
    parser = argparse.ArgumentParser(description='训练Transformer模型')
    parser.add_argument('--task', type=str, default='lm', choices=['lm', 'translation'],
                        help='任务类型: lm(语言建模) 或 translation(机器翻译)')
    parser.add_argument('--dataset', type=str, default='tiny_shakespeare',
                        choices=['tiny_shakespeare', 'iwslt2017'],
                        help='选择数据集')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='训练设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    # 设置随机种子 - 作业要求：重现实验
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print(f"任务: {args.task}, 数据集: {args.dataset}")
    print(f"设备: {args.device}, 随机种子: {args.seed}")

    # 根据任务选择配置
    if args.task == 'lm':
        # 语言建模任务
        train_loader, val_loader, vocab = load_tiny_shakespeare_local(
            file_path=r"D:\pycode\transformer\data\tiny_shakespeare.txt",  # 指定本地文件路径
            seq_len=train_config['seq_len'],
            batch_size=args.batch_size
        )

        #model_config = base_model_config.copy()
        model_config = {
            'd_model': 512,
            'num_heads': 8,
            'd_ff': 2048,
            'num_encoder_layers': 6,
            'num_decoder_layers': 6,  # LM任务也可保持6层decoder，或设置为0
            'dropout': 0.1
        }
        model_config['vocab_size'] = len(vocab)
        # 语言建模使用相同的src和tgt词汇表
        model = Transformer(
            src_vocab_size=len(vocab),
            tgt_vocab_size=len(vocab),
            **{k: v for k, v in model_config.items() if k != 'vocab_size'}
        )
        trainer = Trainer(model, train_loader, val_loader, len(vocab), args.device)

    else:
        # 机器翻译任务
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab = load_iwslt2017_local(
            seq_len=train_config['seq_len'],
            batch_size=args.batch_size
        )

        model_config = seq2seq_model_config.copy()
        model_config['src_vocab_size'] = len(src_vocab)
        model_config['tgt_vocab_size'] = len(tgt_vocab)

        model = Transformer(**model_config)
        trainer = Seq2SeqTrainer(model, train_loader, val_loader, len(tgt_vocab), args.device)

    # 参数统计 - 作业要求
    print(f"模型参数数量: {model.get_num_params():,}")

    # 开始训练
    trainer.train(
        epochs=args.epochs,
        lr=args.lr,
        warmup_steps=train_config['warmup_steps'],
        clip_grad=train_config['clip_grad'],
        save_dir=f'checkpoints_{args.task}_{args.dataset}'
    )

    # 绘制训练曲线 - 作业要求：结果图表
    trainer.plot_training_curves(f'training_curves_{args.task}_{args.dataset}.png')


if __name__ == '__main__':
    main()
"""
"""
主训练脚本 - 作业要求：提供运行命令和硬件要求
专注于IWSLT2017机器翻译任务，展示完整Encoder-Decoder能力
"""

import torch
import argparse
from src.model import Transformer
from src.data_loader import load_iwslt2017_local
from src.train import Seq2SeqTrainer
from src.config import train_config


def main():
    # 命令行参数解析 - 作业要求：提供重现实验的exact命令行
    parser = argparse.ArgumentParser(description='训练Transformer机器翻译模型')
    parser.add_argument('--epochs', type=int, default=2, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='训练设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--d_model', type=int, default=64, help='模型维度')
    parser.add_argument('--num_heads', type=int, default=2, help='注意力头数')
    parser.add_argument('--d_ff', type=int, default=128, help='前馈网络维度')
    parser.add_argument('--num_layers', type=int, default=1, help='编码器和解码器层数')

    args = parser.parse_args()

    # 设置随机种子 - 作业要求：重现实验
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print("=" * 60)
    print("Transformer机器翻译训练 - IWSLT2017英德翻译")
    print("=" * 60)
    print(f"设备: {args.device}")
    print(f"随机种子: {args.seed}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"训练轮数: {args.epochs}")
    print(f"模型配置: d_model={args.d_model}, heads={args.num_heads}")
    print(f"          d_ff={args.d_ff}, layers={args.num_layers}")

    # 加载IWSLT2017机器翻译数据集
    print("\n加载IWSLT2017英德翻译数据集...")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = load_iwslt2017_local(
        seq_len=train_config['seq_len'],
        batch_size=args.batch_size
    )
    print(f"源语言词汇表大小: {len(src_vocab)}")
    print(f"目标语言词汇表大小: {len(tgt_vocab)}")
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"验证样本数: {len(val_loader.dataset)}")

    # 模型配置 - 使用完整的Encoder-Decoder架构
    model_config = {
        'src_vocab_size': len(src_vocab),
        'tgt_vocab_size': len(tgt_vocab),
        'd_model': args.d_model,
        'num_heads': args.num_heads,
        'd_ff': args.d_ff,
        'num_encoder_layers': args.num_layers,
        'num_decoder_layers': args.num_layers,
        'max_len': 512,
        'dropout': 0.1,
    }

    # 创建完整的Encoder-Decoder Transformer模型
    print("\n创建Encoder-Decoder Transformer模型...")
    model = Transformer(**model_config)

    # 参数统计 - 作业要求
    print(f"模型参数数量: {model.get_num_params():,}")
    print(f"编码器层数: {args.num_layers}, 解码器层数: {args.num_layers}")
    print(f"注意力头数: {args.num_heads}, 模型维度: {args.d_model}")

    # 创建序列到序列训练器
    trainer = Seq2SeqTrainer(model, train_loader, val_loader, len(tgt_vocab), args.device)

    # 开始训练
    print("\n" + "=" * 50)
    print("开始Encoder-Decoder Transformer训练...")
    print("=" * 50)

    trainer.train(
        epochs=args.epochs,
        lr=args.lr,
        warmup_steps=train_config['warmup_steps'],
        clip_grad=train_config['clip_grad'],
        save_dir='checkpoints_translation_iwslt2017'
    )

    # 绘制训练曲线 - 作业要求：结果图表
    print("\n生成训练曲线...")
    trainer.plot_training_curves('training_curves_translation_iwslt2017.png')

    print("\n" + "=" * 60)
    print("训练完成！")
    print("结果文件:")
    print(f"- checkpoints_translation_iwslt2017/ (模型文件)")
    print(f"- training_curves_translation_iwslt2017.png (训练曲线)")
    print("=" * 60)


if __name__ == '__main__':
    main()