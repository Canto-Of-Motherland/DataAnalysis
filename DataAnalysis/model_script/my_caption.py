from transformers import pipeline
from PIL import Image

def Captioner(file_path):
    captioner = pipeline("image-to-text", model="D:\桌面\武大本科期间文件\大三下文件\智能信息系统\大实验\代码\models\Taiyi-BLIP-750M-Chinese")
    image = Image.open(file_path)
    text = captioner(image)
    return text[0]["generated_text"]


if __name__ == "__main__":
    print(Captioner("D:\桌面\武大本科期间文件\大三下文件\智能信息系统\大实验\代码\ZJ\WebProjectZJ\media\只因.jpg"))