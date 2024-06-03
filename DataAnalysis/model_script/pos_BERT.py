from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

def BERT_POS(sentence):
    tokenizer = AutoTokenizer.from_pretrained('models/BERT_POS')
    model = AutoModelForTokenClassification.from_pretrained('models/BERT_POS')
    inputs = tokenizer(sentence, return_tensors="pt")
    # 获取模型输出
    outputs = model(**inputs).logits
    # 获取每个 token 的预测标签
    predictions = torch.argmax(outputs, dim=2)
    # 将 token ID 转换为 token
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    # 获取标签映射
    label_list = model.config.id2label

    # 创建一个包含元组的列表，每个元组的第一个元素是 token，第二个元素是其对应的标注
    result = [{"word": token, "tag": label_list[prediction.item()]} for token, prediction in zip(tokens, predictions[0])
              if token not in tokenizer.all_special_tokens]

    return result




if __name__ == "__main__":

    # # 加载预训练模型和分词器
    # tokenizer = AutoTokenizer.from_pretrained("../../../../models/BERT_POS")
    # model = AutoModelForTokenClassification.from_pretrained("../../../../models/BERT_POS")
    # # tokenizer = AutoTokenizer.from_pretrained("KoichiYasuoka/chinese-bert-wwm-ext-upos")
    # # model = AutoModelForTokenClassification.from_pretrained("KoichiYasuoka/chinese-bert-wwm-ext-upos")
    #
    # # 输入句子
    # sentence = "你好，我是麦克雷"
    # sentence = input("请输入你想要进行词性标注的句子：")
    # # 对输入句子进行分词
    # inputs = tokenizer(sentence, return_tensors="pt")
    #
    # # 获取模型输出
    # outputs = model(**inputs).logits
    #
    # # 获取每个 token 的预测标签
    # predictions = torch.argmax(outputs, dim=2)
    #
    # # 将 token ID 转换为 token
    # tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    #
    #
    # # 获取标签映射
    # label_list = model.config.id2label
    #
    # # 创建一个包含元组的列表，每个元组的第一个元素是 token，第二个元素是其对应的标注
    # result = [{"word":token, "tag":label_list[prediction.item()]} for token, prediction in zip(tokens, predictions[0]) if token not in tokenizer.all_special_tokens]
    #
    # print("词性标注结果如下：")
    # # 打印结果
    # print(result)
    print(BERT_POS("你好"))

