import torch
import argparse
from src.model_ablation import Transformer
from src.data_loader import load_tiny_shakespeare_local, load_iwslt2017_local
from src.train import Seq2SeqTrainer
from src.config import train_config


def main():
    parser = argparse.ArgumentParser(description='Transformer消融实验')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--d_ff', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--disable_pos_encoding', action='store_true', help='禁用位置编码')
    parser.add_argument('--disable_residual', action='store_true', help='禁用残差连接')
    parser.add_argument('--exp_name', type=str, default='baseline', help='实验名称')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    print(f"\n🚀 消融实验：{args.exp_name}")
    print(f"位置编码: {'关闭' if args.disable_pos_encoding else '开启'}")
    print(f"残差连接: {'关闭' if args.disable_residual else '开启'}")

    #train_loader, val_loader, vocab = load_tiny_shakespeare_local(
        #file_path=r"D:\pycode\transformer\data\tiny_shakespeare.txt",
        #seq_len=train_config['seq_len'],
        #batch_size=args.batch_size
    #)
    train_loader, val_loader, test_loader, vocab, vocab = load_iwslt2017_local(
        seq_len=train_config['seq_len'],
        batch_size=args.batch_size
    )

    model_config = {
        'src_vocab_size': len(vocab),
        'tgt_vocab_size': len(vocab),
        'd_model': args.d_model,
        'num_heads': args.num_heads,
        'd_ff': args.d_ff,
        'num_encoder_layers': args.num_layers,
        'num_decoder_layers': args.num_layers,
        'max_len': 512,
        'dropout': 0.1,
        'disable_pos_encoding': args.disable_pos_encoding,
        'disable_residual': args.disable_residual,
    }

    model = Transformer(**model_config)
    print(f"模型参数: {model.get_num_params():,}")

    trainer = Seq2SeqTrainer(model, train_loader, val_loader, len(vocab), args.device)
    trainer.train(
        epochs=args.epochs,
        lr=args.lr,
        warmup_steps=train_config['warmup_steps'],
        clip_grad=train_config['clip_grad'],
        save_dir=f"checkpoints_ablation_{args.exp_name}"
    )

    trainer.plot_training_curves(f"results/{args.exp_name}_curve.png")
    print(f"✅ 实验 {args.exp_name} 结束，图表保存在 results/{args.exp_name}_curve.png")


if __name__ == '__main__':
    main()
