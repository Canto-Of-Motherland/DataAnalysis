import jieba
import torch
import torch.nn as nn
import numpy as np
import pickle

MAX_SEQ_LEN = 100

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, embed_matrix, num_classes, bidirectional=False):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embed_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False  # Freeze the embeddings

        self.lstm = nn.LSTM(embed_dim, 128, batch_first=True, bidirectional=bidirectional)
        self.dropout1 = nn.Dropout(0.5)
        lstm_output_dim = 128 * 2 if bidirectional else 128

        self.dense1 = nn.Linear(lstm_output_dim, 64)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.dropout1(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        x = self.softmax(x)
        return x


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, embed_matrix, num_classes, bidirectional=False):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embed_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False  # Freeze the embeddings

        self.rnn = nn.RNN(embed_dim, 128, batch_first=True, bidirectional=bidirectional)
        self.dropout1 = nn.Dropout(0.5)
        rnn_output_dim = 128 * 2 if bidirectional else 128

        self.dense1 = nn.Linear(rnn_output_dim, 64)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.dropout1(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        x = self.softmax(x)
        return x


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, embed_matrix, num_classes, bidirectional=False):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embed_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False  # Freeze the embeddings

        self.gru = nn.GRU(embed_dim, 128, batch_first=True, bidirectional=bidirectional)
        self.dropout1 = nn.Dropout(0.5)
        gru_output_dim = 128 * 2 if bidirectional else 128

        self.dense1 = nn.Linear(gru_output_dim, 64)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.gru(x)
        x = self.dropout1(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        x = self.softmax(x)
        return x


def text_to_sequence(texts, word_index, max_seq_len=MAX_SEQ_LEN):
    sequences = [[word_index.get(word, 0) for word in jieba.lcut(sentence)] for sentence in texts]
    padded_sequences = np.zeros((len(sequences), max_seq_len), dtype=int)

    for i, seq in enumerate(sequences):
        if len(seq) > max_seq_len:
            padded_sequences[i, :max_seq_len] = seq[:max_seq_len]
        else:
            padded_sequences[i, :len(seq)] = seq

    return padded_sequences


# 示例句子
sample_sentence = "再靠近一点就会爆炸。"
segments = jieba.lcut(sample_sentence)  # 分词结果
embedding_matrix = np.load("../../../../data/embeddings_matrix.npy")
EMBED_DIM = 300  # 词向量维度
VOCAB_SIZE = len(embedding_matrix)
NUM_CLASSES = 36

# 选择模型类型
# model = LSTMClassifier(VOCAB_SIZE, EMBED_DIM, embedding_matrix, NUM_CLASSES, bidirectional=False)
# model.load_state_dict(torch.load('./models/Bi-LSTM-model.pth'))

# model = GRUClassifier(VOCAB_SIZE, EMBED_DIM, embedding_matrix, NUM_CLASSES, bidirectional=False)
# model.load_state_dict(torch.load('./models/Bi-GRU-model.pth'))

model = RNNClassifier(VOCAB_SIZE, EMBED_DIM, embedding_matrix, NUM_CLASSES, bidirectional=False)
model.load_state_dict(torch.load('../../../../models/Bi-RNN-model.pth'))

model.eval()

with open('../../../../data/class2num.pkl', 'rb') as f:
    class_index = pickle.load(f)

with open('../../../../data/word2idx.pkl', 'rb') as f:
    word_index = pickle.load(f)

input_seq = text_to_sequence([sample_sentence], word_index)
input_tensor = torch.tensor(input_seq)
output = model(input_tensor).squeeze(0)
predicted_classes = torch.argmax(output, dim=1).numpy()

index_to_class = {index: label for label, index in class_index.items()}
predicted_labels = [index_to_class[idx] for idx in predicted_classes]
result = list(zip(segments, predicted_labels))

print(result)
