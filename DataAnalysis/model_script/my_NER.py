# import os
# import torch
# from transformers import AutoTokenizer, AutoModelForTokenClassification
# from seqeval.metrics.sequence_labeling import get_entities
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# # Load model from HuggingFace Hub
# tokenizer = AutoTokenizer.from_pretrained("shibing624/bert4ner-base-chinese")
# model = AutoModelForTokenClassification.from_pretrained("shibing624/bert4ner-base-chinese")
# label_list = ['I-ORG', 'B-LOC', 'O', 'B-ORG', 'I-LOC', 'I-PER', 'B-TIME', 'I-TIME', 'B-PER']
#
# sentence = "我是张几，我来自中国。"
#
#
# def get_entity(sentence):
#     tokens = tokenizer.tokenize(sentence)
#     inputs = tokenizer.encode(sentence, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(inputs).logits
#     predictions = torch.argmax(outputs, dim=2)
#     char_tags = [(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].numpy())][1:-1]
#
#
#     print(sentence)
#     print(char_tags)
#
#     pred_labels = [i[1] for i in char_tags]
#     entities = []
#     line_entities = get_entities(pred_labels)
#     for i in line_entities:
#         word = sentence[i[1]: i[2] + 1]
#         entity_type = i[0]
#         entities.append((word, entity_type))
#
#
#
#     print("Sentence entity:")
#     print(entities)
#
#
# get_entity(sentence)
# import os
# import torch
# from transformers import AutoTokenizer, AutoModelForTokenClassification
# from seqeval.metrics.sequence_labeling import get_entities
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# # Load model from HuggingFace Hub
# tokenizer = AutoTokenizer.from_pretrained("shibing624/bert4ner-base-chinese")
# model = AutoModelForTokenClassification.from_pretrained("shibing624/bert4ner-base-chinese")
# label_list = ['I-ORG', 'B-LOC', 'O', 'B-ORG', 'I-LOC', 'I-PER', 'B-TIME', 'I-TIME', 'B-PER']
#
# sentence = "我来自中国。"
#
#
# def get_entity(sentence):
#     tokens = tokenizer.tokenize(sentence)
#     inputs = tokenizer.encode(sentence, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(inputs).logits
#     predictions = torch.argmax(outputs, dim=2)
#     char_tags = [(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].numpy())][1:-1]
#     print(sentence)
#     print(char_tags)
#
#     pred_labels = [i[1] for i in char_tags]
#     entities = []
#     line_entities = get_entities(pred_labels)
#     for i in line_entities:
#         word = sentence[i[1]: i[2] + 1]
#         entity_type = i[0]
#         entities.append((word, entity_type))
#
#     print("Sentence entity:")
#     print(entities)
#
#
# get_entity(sentence)
from nerpy import NERModel

def BERT_NER(sentence):
    model = NERModel("bert", "shibing624/bert4ner-base-chinese")
    _, _, entities = model.predict([sentence], split_on_space=False)
    return [{"entity":i, "tag":j} for i, j in entities[0]]


if __name__ == "__main__":
    # model = NERModel("bert", "shibing624/bert4ner-base-chinese")
    # predictions, raw_outputs, entities = model.predict(["我的名字是张几，我来自中国"], split_on_space=False)
    # print(predictions)
    # print(raw_outputs)
    # print(entities)
    print(BERT_NER("我是蔡徐坤，喜欢唱、跳、rap、篮球，我来自美国"))
