import torch
from torch.utils.data import DataLoader
from model.transformer_model import TransformerModel
from dataset.agnews_dataset import AGNewsDataset
from parameters import parser
from utils import *
from tqdm import tqdm
import torch.nn as nn


device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

def evaluate(model, val_loader, criterion):
    model.eval()
    loss_list = []
    acc_list = []
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
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"Test Loss: {loss:.2f}")
    return loss, acc


if __name__ == "__main__":
    config = parser.parse_args()
    if config.yml_path:
        load_args(config.yml_path, config)
    test_dataset = AGNewsDataset("./dataset/ag", phase='test')
    test_loader = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False)

    vocab_size = len(test_dataset.get_vocab())
    model = TransformerModel(vocab_size=vocab_size, d_model=config.d_model, num_heads=config.num_heads,
                             num_layers=config.num_layers, d_ff=config.dff).to(device)
    model.load_state_dict(torch.load(config.save_path + "/best_transformer_agnews.pth", weights_only=True))
    criterion = nn.CrossEntropyLoss()
    evaluate(model, test_loader, criterion)
