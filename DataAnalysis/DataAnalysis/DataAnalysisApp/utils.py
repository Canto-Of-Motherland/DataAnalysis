import random
import smtplib
import string

from DataAnalysisApp.models import User
from email.mime.text import MIMEText
from email.utils import formataddr


def checkSignIn(request) -> dict:
    try:
        if request.session['status_sign'] == '1':
            user_name = request.session['username']
            return {'signal': True, 'username': user_name}
        else:
            return {'signal': False, 'username': '登录'}
    except KeyError:
        return {'signal': False, 'username': '登录'}
    

def verificationGenerator() -> str:
    verification_code = ""

    for i in range(6):
        item = random.choice(string.digits)
        verification_code += item
    
    return verification_code



def sendMail(verification_code: str, address: str, function: str) -> None:
    try:
        info = '【舆情群体极化分析系统】您的验证码为%s，有效期5分钟，该验证码仅用于%s，请勿泄露。' % (verification_code, function)
        sender = '3095631599@qq.com'
        receiver = address
        send_code = 'dzppwxmgsgsodcdg'

        message = MIMEText(info, _charset='utf-8')
        message['From'] = formataddr(("", sender))
        message['To'] = formataddr(('', receiver))
        message['Subject'] = '验证码'

        server = smtplib.SMTP_SSL('smtp.qq.com', 465)
        server.login(sender, send_code)
        server.sendmail(sender, [receiver, ], message.as_string())
        server.quit()
        return True
    except Exception as e:
        print(e)
        return False