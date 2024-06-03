import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 定义去除多余空白字符的处理函数
WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

# 加载模型和分词器

def summarize_text(text, max_length=84, num_beams=4):
    """
    给定一段文本，生成其摘要。

    参数：
    text (str): 需要摘要的文本
    max_length (int): 生成摘要的最大长度
    num_beams (int): beam search 的宽度

    返回：
    summary_model (str): 生成的摘要
    """
    model_name = r"C:\Users\29032\.cache\huggingface\hub\models--csebuetnlp--mT5_m2o_chinese_simplified_crossSum\snapshots\484d6b4a469251fb8432f9da7eb2da761932668f"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # 处理输入文本
    input_text = WHITESPACE_HANDLER(text)

    # 将文本编码为模型输入
    input_ids = tokenizer(
        [input_text],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )["input_ids"]

    # 生成摘要
    output_ids = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        no_repeat_ngram_size=2,
        num_beams=num_beams
    )[0]

    # 解码生成的摘要
    summary = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return summary


# 示例用法
if __name__ == '__main__':
    article_text = """"
    Videos that say approved vaccines are dangerous and cause autism, cancer or infertility are among those that will be taken down, the company said.  The policy includes the termination of accounts of anti-vaccine influencers.  Tech giants have been criticised for not doing more to counter false health information on their sites.  In July, US President Joe Biden said social media platforms were largely responsible for people's scepticism in getting vaccinated by spreading misinformation, and appealed for them to address the issue.  YouTube, which is owned by Google, said 130,000 videos were removed from its platform since last year, when it implemented a ban on content spreading misinformation about Covid vaccines.  In a blog post, the company said it had seen false claims about Covid jabs "spill over into misinformation about vaccines in general". The new policy covers long-approved vaccines, such as those against measles or hepatitis B.  "We're expanding our medical misinformation policies on YouTube with new guidelines on currently administered vaccines that are approved and confirmed to be safe and effective by local health authorities and the WHO," the post said, referring to the World Health Organization.
    """
    summary = summarize_text(article_text)
    print(summary)
