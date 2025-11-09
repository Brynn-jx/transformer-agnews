import argparse

parser = argparse.ArgumentParser()

# --- Model Parameters ---
parser.add_argument('--d_model', type=int, default=256, help='模型的维度')
parser.add_argument('--num_heads', type=int, default=8, help='注意力头的数量')
parser.add_argument('--num_layers', type=int, default=4, help='Transformer 层数')
parser.add_argument('--dff', type=int, default=1024, help='前馈网络的维度')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout 比率')
parser.add_argument('--max_len', type=int, default=50, help='目标序列长度')
parser.add_argument("--yml_path", help="yml path", type=str)
parser.add_argument("--save_path", help="save path", type=str)

# --- Training Parameters ---
parser.add_argument('--train_batch_size', type=int, default=64, help='训练时的批次大小')
parser.add_argument('--dataset', type=str, default='c-fashion', help='使用的数据集名称')
parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
parser.add_argument('--num_epochs', type=int, default=10, help='训练的总轮数')
parser.add_argument('--optimizer', type=str, default='AdamW', help='优化器类型')
parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
parser.add_argument('--val_metric', type=str, default='best_acc', help='验证指标，用于选择最佳模型')
parser.add_argument('--save_final_model', action='store_true', help='是否在训练结束后保存最终模型')

# --- Testing Parameters ---
parser.add_argument('--eval_batch_size', type=int, default=64, help='评估时的批次大小')
parser.add_argument('--open_world', action='store_true', help='是否开启开放世界评估')
# parser.add_argument('--load_model', type=str, default=None, help='加载的模型路径') # 注释掉的参数
parser.add_argument('--topk', type=int, default=1, help='Top-K 准确率')
parser.add_argument('--text_encoder_batch_size', type=int, default=1024, help='文本编码器的批次大小')
# parser.add_argument('--threshold', type=float, default=0.4, help='分类阈值') # 注释掉的参数
parser.add_argument('--threshold_trials', type=int, default=50, help='阈值搜索的试验次数')
parser.add_argument('--bias', type=float, default=0.001, help='偏置项')
parser.add_argument('--text_first', action='store_true', help='是否先处理文本')
