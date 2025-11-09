import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model.transformer_model import TransformerModel
from parameters import parser
from utils import *
import pickle
from torch.cuda.amp import GradScaler, autocast

# 你自己的数据加载器
from dataset.agnews_dataset import AGNewsDataset  # 用你原来的那个类
from torch.utils.data import DataLoader

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

def evaluate(model, val_loader, criterion):
    model.eval()
    loss_list = []
    acc_list= []
    with torch.no_grad():
        with tqdm(total=len(val_loader)) as pbar:
            for src, labels in val_loader:
                src, labels = src.to(device), labels.to(device)
                preds = model(src, src)
                loss = criterion(preds, labels)
                loss_list.append(loss.item())
                pred_labels = preds.argmax(1)
                correct = (pred_labels == labels).sum().item()
                correct = correct/len(labels)
                acc_list.append(correct)
                pbar.set_postfix(loss=f"{loss:.4f}", correct=f"{correct:.2%}")
                pbar.update(1)

    acc = np.mean(acc_list)
    loss = np.mean(loss_list)
    print(f"Validation Accuracy: {acc * 100:.2f}%")
    print(f"Validation Loss: {loss:.2f}")
    return loss, acc


def train_model(model, train_loader, val_loader, optimizer, criterion, config):
    best_acc = 0.0
    best_loss = 1e5
    record_dict = {
        "train": [],
        "val": []
    }
    global_step = -1
    scaler = GradScaler(device)  # 初始化梯度缩放器
    # ================== 训练 ==================
    for epoch in range(config.num_epochs):
        with tqdm(total=len(train_loader), desc="epoch % 3d" % (epoch + 1)) as pbar:
            model.train()
            total_loss = 0
            total_acc = 0
            global_step += 1
            for src, labels in train_loader:
                src, labels = src.to(device), labels.to(device)
                tgt = src  # 分类任务中可视作目标句（自回归输入）
                optimizer.zero_grad()
                # 使用 autocast 上下文管理器
                # with autocast():
                #     output = model(src, tgt)
                #     loss = criterion(output, labels)
                # # 缩放损失并反向传播
                # scaler.scale(loss).backward()
                # # 缩放梯度并更新权重
                # scaler.step(optimizer)
                # # 更新缩放器
                # scaler.update()
                output = model(src, tgt)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                pred_labels = output.argmax(1)
                correct = (pred_labels == labels).sum().item()
                correct = correct / len(labels)
                optimizer.step()
                total_loss += loss.item()
                total_acc += correct
                record_dict["train"].append({
                    "loss": loss.item(), "acc": correct, "step": global_step, "epoch": epoch + 1
                })
                pbar.update(1)
                pbar.set_postfix({"epoch": epoch + 1, "loss": loss})

        print(f"Epoch {epoch + 1} | Train Loss: {total_loss / len(train_loader):.4f}")
        print(f"Epoch {epoch + 1} | Train Acc: {total_acc / len(train_loader):.4f}%")
        # 验证
        loss, acc = evaluate(model, val_loader, criterion)
        if config.val_metric == 'best_acc':
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), config.save_path + "/best_transformer_agnews.pth")
                print(f" Model saved with acc={best_acc * 100:.2f}%")
        else:
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), config.save_path + "/best_transformer_agnews.pth")
                print(f" Model saved with loss={loss}")
        record_dict["val"].append({
            "loss": loss, "acc": acc, "step": global_step, "epoch": epoch + 1
        })


    if config.save_final_model:
        torch.save(model.state_dict(), config.save_path + "/final_transformer_agnews.pth")

    #  保存训练损失和准确率数据
    with open(config.save_path + "/record_dict.pkl", "wb") as f:
        pickle.dump(record_dict, f)


if __name__ == "__main__":
    config = parser.parse_args()
    print(config)
    if config.yml_path:
        load_args(config.yml_path, config)

    # ================== 初始化 ==================
    train_dataset = AGNewsDataset("./dataset/ag", phase='train')
    val_dataset = AGNewsDataset("./dataset/ag", phase='val')
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.train_batch_size, shuffle=False)

    vocab_size = len(train_dataset.get_vocab())
    model = TransformerModel(vocab_size=vocab_size, d_model=config.d_model, num_heads=config.num_heads,
                             num_layers=config.num_layers, d_ff=config.dff).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    train_model(model, train_loader, val_loader, optimizer, criterion, config)
