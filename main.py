from typing import List, Set, Any
import torch
import torch.nn as nn
import torch.optim as optim
import polars as pl

# 准备数据
data = [
    ("你好", "反面"),
    ("吃了吗", "正面"),
    ("没事呢", "正面"),
    ("哈哈", "正面"),
    ("你今天过得怎么样", "正面"),
    ("真讨厌", "负面"),
    ("好伤心", "负面"),
    ("好烦", "负面"),
    ("不开心", "负面"),
    ("好失望", "负面"),
]


class SimpleTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SimpleTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        hidden = torch.mean(embedded, dim=1)
        hidden = self.fc(hidden)
        output = self.output(hidden)
        return output


def load_data(file_names: List[str], batch_size: int):
    merged_data: pl.DataFrame = None
    for file_name in file_names:
        if file_name == "":
            continue
        reader = pl.read_csv_batched(file_name, separator="\t", batch_size=batch_size)
        for batch in reader:
            if merged_data is None:
                merged_data = batch
            else:
                merged_data = merged_data.append(batch)
    return merged_data


def build_vocab():
    pass


def main():
    # 构建词汇表
    vocab: Set | Any = set()
    for sentence, _ in data:
        print(f"sentence: {sentence}")
        vocab.update(sentence)
    vocab = sorted(vocab)
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    # 创建模型实例
    embedding_dim = 10
    hidden_dim = 8
    output_dim = 2  # 正面和负面情感
    model = SimpleTextClassifier(len(vocab), embedding_dim, hidden_dim, output_dim)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器

    # 将文本数据转换为张量
    def text_to_tensor(text: str) -> torch.Tensor:
        return torch.tensor([word_to_idx[word] for word in text], dtype=torch.long)

    # 训练模型
    for epoch in range(100):
        total_loss = 0
        for sentence, label in data:
            model.zero_grad()
            inputs = text_to_tensor(sentence)
            outputs = model(inputs.unsqueeze(0))
            loss = criterion(
                outputs, torch.tensor([0 if label == "正面" else 1], dtype=torch.long))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/100], Loss: {total_loss:.4f}")

    # 使用训练好的模型进行预测
    test_sentence = input("请输入你的文字进行情感预测:")
    inputs = text_to_tensor(test_sentence)
    outputs = model(inputs.unsqueeze(0))
    predicted_label = "正面" if torch.argmax(outputs).item() == 0 else "负面"
    print(f"句子'{test_sentence}'的情感预测为：{predicted_label}")


if __name__ == "__main__":
    load_data()
