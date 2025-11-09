import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer


# **3. 处理数据**
class AGNewsDataset(Dataset):
    def __init__(self,root, phase, model_name = "bert-base-uncased", max_length=50):
        super(AGNewsDataset, self).__init__()
        self.root = root
        self.phase = phase
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = self.read_data()
        if phase == "train":
            self.texts = train_texts
            self.labels = train_labels
        elif phase == "val":
            self.texts = val_texts
            self.labels = val_labels
        else:
            self.texts = test_texts
            self.labels = test_labels

        # 加载 BERT Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def read_data(self):
        df = pd.read_csv("./dataset/ag/train.csv")  # 读取训练集数据
        df.columns = ["label", "title", "description"]  # CSV 有3列: 标签, 标题, 描述
        df["text"] = df["title"] + " " + df["description"]  # 合并标题和描述作为输入文本
        df["label"] = df["label"] - 1  # AG NEWS 的标签是 1-4，我们转换成 0-3
        train_texts, train_labels = df["text"].tolist(), df["label"].tolist()
        number = int(len(train_texts))
        num_val = int(number * 0.2)
        val_texts, val_labels = train_texts[: num_val], train_labels[: num_val]
        train_texts, train_labels = train_texts[num_val: number], train_labels[num_val: number]

        df = pd.read_csv("./dataset/ag/test.csv")  # 读取测试集数据
        df.columns = ["label", "title", "description"]  # CSV 有3列: 标签, 标题, 描述
        df["text"] = df["title"] + " " + df["description"]  # 合并标题和描述作为输入文本
        df["label"] = df["label"] - 1  # AG NEWS 的标签是 1-4，我们转换成 0-3
        test_texts, test_labels = df["text"].tolist(), df["label"].tolist()
        # test_num = int(len(test_texts)*0.1)
        # test_texts, test_labels = test_texts[: test_num], test_labels[: test_num]
        return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        return input_ids, torch.tensor(label, dtype=torch.long)


    def get_vocab(self):
        return self.tokenizer.vocab

    def get_pad_token_id(self):
        return self.tokenizer.pad_token_id

    def get_unk_token_id(self):
        return self.tokenizer.unk_token_id
