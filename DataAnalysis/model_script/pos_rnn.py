import jieba
import torch
import torch.nn as nn
import numpy as np
import pickle

# 设置最大序列长度
MAX_SEQUENCE_LENGTH = 100

# 加载预训练的词向量矩阵
embeddings_matrix = np.load('data/embeddings_matrix.npy')

# 模型参数
EMBEDDING_DIM = 300  # 词向量维度，即每个词向量的维度大小
VOCAB_SIZE = len(embeddings_matrix)  # 词汇表的大小，即词向量矩阵的行数
NUM_CLASSES = 36  # 分类的类别数

# 定义词性标注器模型类（使用LSTM）
class WordTagging(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embeddings_matrix, num_classes, bidirectional=False):
        super(WordTagging, self).__init__()
        
        # 定义嵌入层，用于将输入的词索引转化为词向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embeddings_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False  # 冻结词向量权重，使其在测试过程中不更新
        
        # 定义LSTM层，用于处理序列数据
        self.lstm = nn.LSTM(embedding_dim, 128, batch_first=True, bidirectional=bidirectional)
        self.dropout1 = nn.Dropout(0.5)  # Dropout层，用于防止过拟合
        lstm_output_dim = 128 * 2 if bidirectional else 128  # 根据是否为双向LSTM，设置输出维度

        # 定义全连接层和激活函数
        self.dense1 = nn.Linear(lstm_output_dim, 64)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)  # 第二个Dropout层

        # 定义输出层
        self.dense2 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=2)  # Softmax层，用于多分类任务

    def forward(self, x):
        # 嵌入层：将输入的词索引转化为词向量
        x = self.embedding(x)
        
        # LSTM层：处理词向量序列
        x, _ = self.lstm(x)
        x = self.dropout1(x)  # 应用Dropout杀死部分节点（训练的时候防止过拟合的）
        
        # 全连接层和激活函数：对LSTM的输出进行进一步处理
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout2(x)  # 再次应用Dropout
        
        # 输出层：得到分类结果
        x = self.dense2(x)
        x = self.softmax(x)  # 应用Softmax

        return x

# 定义词性标注器模型类（使用RNN）
class WordTaggingRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embeddings_matrix, num_classes, bidirectional=False):
        super(WordTaggingRNN, self).__init__()
        
        # 定义嵌入层，用于将输入的词索引转化为词向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embeddings_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False  # 冻结词向量权重，使其在训练过程中不更新
        
        # 定义RNN层，用于处理序列数据
        self.rnn = nn.RNN(embedding_dim, 128, batch_first=True, bidirectional=bidirectional)
        self.dropout1 = nn.Dropout(0.5)  # Dropout层，用于防止过拟合
        rnn_output_dim = 128 * 2 if bidirectional else 128  # 根据是否为双向RNN，设置输出维度

        # 定义全连接层和激活函数
        self.dense1 = nn.Linear(rnn_output_dim, 64)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)  # 第二个Dropout层

        # 定义输出层
        self.dense2 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=2)  # Softmax层，用于多分类任务

    def forward(self, x):
        # 嵌入层：将输入的词索引转化为词向量
        x = self.embedding(x)
        
        # RNN层：处理词向量序列
        x, _ = self.rnn(x)
        x = self.dropout1(x)  # 应用Dropout
        
        # 全连接层和激活函数：对RNN的输出进行进一步处理
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout2(x)  # 再次应用Dropout
        
        # 输出层：得到分类结果
        x = self.dense2(x)
        x = self.softmax(x)  # 应用Softmax

        return x

# 定义词性标注器模型类（使用GRU）
class WordTaggingGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embeddings_matrix, num_classes, bidirectional=False):
        super(WordTaggingGRU, self).__init__()
        
        # 定义嵌入层，用于将输入的词索引转化为词向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embeddings_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False  # 冻结词向量权重，使其在训练过程中不更新
        
        # 定义GRU层，用于处理序列数据
        self.gru = nn.GRU(embedding_dim, 128, batch_first=True, bidirectional=bidirectional)
        self.dropout1 = nn.Dropout(0.5)  # Dropout层，用于防止过拟合
        gru_output_dim = 128 * 2 if bidirectional else 128  # 根据是否为双向GRU，设置输出维度

        # 定义全连接层和激活函数
        self.dense1 = nn.Linear(gru_output_dim, 64)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)  # 第二个Dropout层

        # 定义输出层
        self.dense2 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=2)  # Softmax层，用于多分类任务

    def forward(self, x):
        # 嵌入层：将输入的词索引转化为词向量
        x = self.embedding(x)
        
        # GRU层：处理词向量序列
        x, _ = self.gru(x)
        x = self.dropout1(x)  # 应用Dropout
        
        # 全连接层和激活函数：对GRU的输出进行进一步处理
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout2(x)  # 再次应用Dropout
        
        # 输出层：得到分类结果
        x = self.dense2(x)
        x = self.softmax(x)  # 应用Softmax

        return x

# 文本分词和编码函数，将文本转换为模型输入的格式
def tokenizer(texts, word_index):
    data = []
    for sentence in texts:
        # 将每个词转换为对应的索引，如果词不在词汇表中，则使用索引0
        new_sentence = [word_index.get(word, 0) for word in sentence]
        data.append(new_sentence)

    # 对齐句子长度，填充或截断到最大序列长度
    padded_data = np.zeros((len(data), MAX_SEQUENCE_LENGTH), dtype=int)

    for i, sentence in enumerate(data):
        if len(sentence) > MAX_SEQUENCE_LENGTH:
            padded_data[i, :MAX_SEQUENCE_LENGTH] = sentence[:MAX_SEQUENCE_LENGTH]
        else:
            padded_data[i, :len(sentence)] = sentence

    return padded_data

# 使用双向RNN进行词性标注
def Bi_RNN(sentence):
    # 加载预训练的模型
    model = WordTaggingRNN(VOCAB_SIZE, EMBEDDING_DIM, embeddings_matrix, NUM_CLASSES, bidirectional=False)
    model.load_state_dict(torch.load('models/Bi-RNN-model.pth'))
    model.eval()

    # 加载标签和词汇索引
    with open('data/class2num.pkl', 'rb') as f:
        labels_index = pickle.load(f)

    with open('data/word2idx.pkl', 'rb') as f:
        word_index = pickle.load(f)

    # 分词，将句子分解为单词列表
    segments = jieba.lcut(sentence)
    input_ids = tokenizer([segments], word_index)  # 将单词列表转换为索引列表
    input_ids = torch.tensor(input_ids)  # 转换为PyTorch张量

    # 模型预测
    with torch.no_grad():
        output = model(input_ids)  # 预测结果
        output = output.squeeze(0)  # 去掉批次维度

    result = torch.argmax(output, dim=1).numpy().tolist()  # 获取每个词的预测标签索引

    # 标签映射，将索引转换为实际标签
    idx2cls = {index: char for char, index in labels_index.items()}
    cls_list = [idx2cls[i] for i in result]

    # 将单词和标签结合起来
    final_result = [{'word': word, 'tag': tag} for word, tag in zip(segments, cls_list)]

    return final_result

# 使用双向GRU进行词性标注
def Bi_GRU(sentence):
    # 加载预训练的模型
    model = WordTaggingGRU(VOCAB_SIZE, EMBEDDING_DIM, embeddings_matrix, NUM_CLASSES, bidirectional=False)
    model.load_state_dict(torch.load('models/Bi-GRU-model.pth'))
    model.eval()

    # 加载标签和词汇索引
    with open('data/class2num.pkl', 'rb') as f:
        labels_index = pickle.load(f)

    with open('data/word2idx.pkl', 'rb') as f:
        word_index = pickle.load(f)

    # 分词，将句子分解为单词列表
    segments = jieba.lcut(sentence)
    input_ids = tokenizer([segments], word_index)  # 将单词列表转换为索引列表
    input_ids = torch.tensor(input_ids)  # 转换为PyTorch张量

    # 模型预测
    with torch.no_grad():
        output = model(input_ids)  # 预测结果
        output = output.squeeze(0)  # 去掉批次维度

    result = torch.argmax(output, dim=1).numpy().tolist()  # 获取每个词的预测标签索引

    # 标签映射，将索引转换为实际标签
    idx2cls = {index: char for char, index in labels_index.items()}
    cls_list = [idx2cls[i] for i in result]

    # 将单词和标签结合起来
    final_result = [{'word': word, 'tag': tag} for word, tag in zip(segments, cls_list)]

    return final_result

# 使用双向LSTM进行词性标注
def Bi_LSTM(sentence):
    # 加载预训练的模型
    model = WordTagging(VOCAB_SIZE, EMBEDDING_DIM, embeddings_matrix, NUM_CLASSES, bidirectional=False)
    model.load_state_dict(torch.load('models/Bi-LSTM-model.pth'))
    model.eval()

    # 加载标签和词汇索引
    with open('data/class2num.pkl', 'rb') as f:
        labels_index = pickle.load(f)

    with open('data/word2idx.pkl', 'rb') as f:
        word_index = pickle.load(f)

    # 分词，将句子分解为单词列表
    segments = jieba.lcut(sentence)
    input_ids = tokenizer([segments], word_index)  # 将单词列表转换为索引列表
    input_ids = torch.tensor(input_ids)  # 转换为PyTorch张量

    # 模型预测
    with torch.no_grad():
        output = model(input_ids)  # 预测结果
        output = output.squeeze(0)  # 去掉批次维度

    result = torch.argmax(output, dim=1).numpy().tolist()  # 获取每个词的预测标签索引

    # 标签映射，将索引转换为实际标签
    idx2cls = {index: char for char, index in labels_index.items()}
    cls_list = [idx2cls[i] for i in result]

    # 将单词和标签结合起来
    final_result = [{'word': word, 'tag': tag} for word, tag in zip(segments, cls_list)]

    return final_result

# 主函数，用于测试模型
if __name__ == '__main__':
    print(Bi_RNN("全民制作人们你们好，我是练习时长两年半的个人练习生蔡徐坤，喜欢唱、跳、rap、篮球，music"))
