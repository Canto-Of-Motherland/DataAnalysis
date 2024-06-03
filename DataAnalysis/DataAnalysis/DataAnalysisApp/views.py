import time
import os
import json
from pathlib import Path

from django.http import JsonResponse, HttpResponseRedirect
from django.shortcuts import render, redirect
from DataAnalysisApp.models import User
from DataAnalysisApp import utils
# from DataAnalysisApp.model_script.pos_rnn import Bi_RNN, Bi_GRU, Bi_LSTM
# from DataAnalysisApp.model_script.pos_BERT import BERT_POS
# from DataAnalysisApp.model_script.my_NER import BERT_NER

base_dir = Path(__file__).resolve().parent.parent


def logout(request):
    if request.method == 'GET':
        return redirect('')
    else:
        request.session['status_sign'] = '0'
        return JsonResponse({})


def index(request):
    if request.method == 'GET':
        if utils.checkSignIn(request)['signal']:
            username = utils.checkSignIn(request)['username']
            url = 'index'
        else:
            username = '登录'
            url = 'signIn'
        return render(request, 'index.html', {'data': {'username': username, 'url': url}})


def signIn(request):
    if request.method == 'GET':
        if utils.checkSignIn(request)['signal']:
            return HttpResponseRedirect('/')
        else:
            return render(request, 'signIn.html')
    else:
        func_code = request.POST.get('func_code')
        status_code = 0
        if func_code == '1':
            user_email = request.POST.get('email')
            if User.objects.filter(user_email=user_email):
                verification_code = utils.verificationGenerator()
                timestamp_start = int(time.time())
                request.session['code'] = verification_code
                request.session['code_used'] = '0'
                request.session['timestamp'] = timestamp_start
                if not utils.sendMail(verification_code, user_email, '登录'):
                    status_code = 1
            else:
                status_code = 2
        elif func_code == '2':
            user_email = request.POST.get('email')
            user_password = request.POST.get('password')
            result = User.objects.filter(user_email=user_email)
            if result:
                if result.first().user_password == user_password:
                    request.session['status_sign'] = '1'
                    request.session['username'] = result.first().user_name
                else:
                    status_code = 1
            else:
                status_code = 2
        elif func_code == '3':
            try:
                user_email = request.POST.get('email')
                user_code = request.POST.get('code')
                result = User.objects.filter(user_email=user_email)
                if result:
                    if int(request.session['timestamp']) + 300 > int(time.time()):
                        if request.session['code'] == user_code:
                            if request.session['code_used'] == '0':
                                request.session['status_sign'] = '1'
                                request.session['code_used'] = '1'
                                request.session['username'] = result.first().user_name
                            else:
                                status_code = 1
                        else:
                            status_code = 2
                    else:
                        status_code = 3
                else:
                    status_code = 4
            except Exception as e:
                status_code = 5
        return JsonResponse({'status_code': status_code})

def signUp(request):
    if request.method == 'GET':
        if utils.checkSignIn(request)['signal']:
            return HttpResponseRedirect('/')
        else:
            return render(request, 'signUp.html')
    else:
        func_code = request.POST.get('func_code')
        status_code = 0
        if func_code == '1':
            user_email = request.POST.get('email')
            if User.objects.filter(user_email=user_email):
                status_code = 1
            else:
                verification_code = utils.verificationGenerator()
                timestamp_start = int(time.time())
                request.session['user_email'] = user_email
                request.session['code'] = verification_code
                request.session['code_used'] = '0'
                request.session['timestamp'] = timestamp_start
                if not utils.sendMail(verification_code, user_email, '注册'):
                    status_code = 2
        elif func_code == '2':
            try:
                user_code = request.POST.get('code')
                if int(request.session['timestamp']) + 300 >= int(time.time()):
                    if request.session['code'] == user_code:
                        if request.session['code_used'] == '1':
                            status_code = 1
                        else:
                            request.session['code_used'] = '1'
                    else:
                        status_code = 2
                else:
                    status_code = 3
            except Exception as e:
                print(e)
                status_code = 4
        elif func_code == '3':
            user_id = 'u' + str(int(time.time()))
            user_name = request.POST.get('username')
            user_email = request.session['user_email']
            user_password = request.POST.get('password')
            print(user_id, user_name, user_password)
            user = User(user_id=user_id, user_name=user_name, user_email=user_email, user_password=user_password)
            user.save()
        return JsonResponse({'status_code': status_code})


def sentence(request):
    if request.method == 'GET':
        if utils.checkSignIn(request)['signal']:
            username = utils.checkSignIn(request)['username']
            url = 'sentence'
        else:
            return HttpResponseRedirect('/sign-in/')
            
        return render(request, 'sentence.html', {'data': {'username': username, 'url': url}})
    else:
        func_code = request.POST.get('func_code')
        # if func_code == '1':
        #     sentence = request.POST.get('sentence')
        #     model = request.POST.get('model')
        #     data = [{'word': '你', 'tag': 'AA'}, {'word': '好', 'tag': 'BB'}, {'word': '吗', 'tag': 'CC'}]
        #     return JsonResponse({'data': data})
        # elif func_code == '2':
        #     sentence = request.POST.get('sentence')
        #     data = [{'entity': '你', 'tag': 'AA'}, {'entity': '武汉大学', 'tag': 'CC'}]
        #     return JsonResponse({'data': data})
        if func_code == '1':
            sentence = request.POST.get('sentence')
            model = request.POST.get('model')
            # if model == '1':
            #     data = Bi_RNN(sentence)
            # if model == '2':
            #     data = Bi_GRU(sentence)
            # if model == '3':
            #     data = Bi_LSTM(sentence)
            # if model == '4':
            #     data = BERT_POS(sentence)
            data = [{'word': '你', 'tag': 'AA'}, {'word': '好', 'tag': 'BB'}, {'word': '吗', 'tag': 'CC'}]
            return JsonResponse({'data': data})
        elif func_code == '2':
            sentence = request.POST.get('sentence')
            data = [{'entity': '你', 'tag': 'AA'}, {'entity': '武汉大学', 'tag': 'CC'}]
            # data = BERT_NER(sentence)
            return JsonResponse({'data': data})


def opinionClassification(request):
    if request.method == 'GET':
        if utils.checkSignIn(request)['signal']:
            username = utils.checkSignIn(request)['username']
            url = 'opinionClassification'
        else:
            return HttpResponseRedirect('/sign-in/')
        return render(request, 'opinionClassification.html', {'data': {'username': username, 'url': url}})
    else:
        with open(os.path.join(base_dir, 'static/img/1-1.json'), 'r', encoding='utf-8') as file_1_1:
            data_1_1 = json.load(file_1_1) 
        with open(os.path.join(base_dir, 'static/img/1-2.json'), 'r', encoding='utf-8') as file_1_2:
            data_1_2 = json.load(file_1_2) 
        time.sleep(10)
        return JsonResponse({
            'data': {
                'data_1_1': data_1_1,
                'data_1_2': data_1_2,
            }
        })


def opinionAnalysis(request):
    if request.method == 'GET':
        if utils.checkSignIn(request)['signal']:
            username = utils.checkSignIn(request)['username']
            url = 'opinionAnalysis'
        else:
            return HttpResponseRedirect('/sign-in/')
        return render(request, 'opinionAnalysis.html', {'data': {'username': username, 'url': url}})
    else:
        with open(os.path.join(base_dir, 'static/img/2-1.json'), 'r', encoding='utf-8') as file_2_1:
            data_2_1 = json.load(file_2_1)
        with open(os.path.join(base_dir, 'static/img/2-2.json'), 'r', encoding='utf-8') as file_2_2:
            data_2_2 = json.load(file_2_2)
        with open(os.path.join(base_dir, 'static/img/2-3.json'), 'r', encoding='utf-8') as file_2_3:
            data_2_3 = json.load(file_2_3)
        time.sleep(10)
        return JsonResponse({
            'data': {
                'data_2_1': data_2_1,
                'data_2_2': data_2_2,
                'data_2_3': data_2_3,
            }
        })