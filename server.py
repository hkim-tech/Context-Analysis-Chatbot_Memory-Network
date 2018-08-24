# BottlePy web programming micro-framework
import bottle
from bottle import request, route, template, get, post, delete
# import urllib.request
# from urllib.parse import urlencode
import os
import os.path
import traceback
import json
import socket

hostname = socket.gethostname()
IP = socket.gethostbyname(hostname)

# # import answers
# with open("answers.json", 'rt', encoding='UTF8') as f:
#     answers = json.load(f)

# import apps from subfolders
for dir in os.listdir():
    appFilename = os.path.join(dir, dir + '.py')
    if  os.path.isfile(appFilename):
        print("Importing " + dir + "...")
        try:
            __import__(dir + '.' + dir)
        except:
            print("Failed to import " + dir + ":")
            msg = traceback.format_exc()
            print(msg)
            bottle.route('/' + dir, 'GET', 
                lambda msg=msg, dir=dir: 
                    reportImportError(dir, msg))

def reportImportError(dir, msg):
    return """<html><body>
        <h1>There was an error importing application {0}</h1>
        <pre>{1}</pre>
        </body></html>""".format(dir, msg)

@route('/<filename:path>')
def send_static(filename):
    """Helper handler to serve up static game assets.

    (Borrowed from BottlePy documentation examples.)"""
    if str(filename).find('.py') == -1:
        return bottle.static_file(filename, root='.')
    else:
        return """You do not have sufficient permissions to access this page."""

@route('/', method='Get')
def index():
    if os.path.isfile("index.html"):
        return bottle.static_file("index.html", root='.')
    else:
        return """<html><body>Nothing to see here... move along now.
        Or, create an index.html file.</body></html>"""


    

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from scipy.spatial import distance
import codecs
import re
import numpy as np


#모델 로딩
model = load_model('aurabot.h5')

story_maxlen = 5
query_maxlen = 5
vocab = ['?', 'UNK', '건', '계십니까', '너', '너에', '너희', '넌', '널', '누구냐구', '누구야', '누군지', '니가', '다른', '대해', '만든', '뭐든', '뭐야', '뭘', '반갑다', '반갑워', '봇', '봇소개1', '봇소개그만', '사람', '사람이', '소개', '소개1', '소개해줘', '심심해', '아는게', '아무거나', '아우라', '아우라소개1', '아우라소개그만', '아우라에', '안녕', '알려줘', '없어', '오랜만이야', '이름이', '인사1', '인사그만', '있어', '재밌는거', '주인', '주인이', '지루해', '팀', '팀에', '팀을', '하세요', '하십니까', '하이', '할수', '할줄', '해봐', '해줘', '헬로우']
word_idx = {'?': 1, 'UNK': 2, '건': 3, '계십니까': 4, '너': 5, '너에': 6, '너희': 7, '넌': 8, '널': 9, '누구냐구': 10, '누구야': 11, '누군지': 12, '니가': 13, '다른': 14, '대해': 15, '만든': 16, '뭐든': 17, '뭐야': 18, '뭘': 19, '반갑다': 20, '반갑워': 21, '봇': 22, '봇소개1': 23, '봇소개그만': 24, '사람': 25, '사람이': 26, '소개': 27, '소개1': 28, '소개해줘': 29, '심심해': 30, '아는게': 31, '아무거나': 32, '아우라': 33, '아우라소개1': 34, '아우라소개그만': 35, '아우라에': 36, '안녕': 37, '알려줘': 38, '없어': 39, '오랜만이야': 40, '이름이': 41, '인사1': 42, '인사그만': 43, '있어': 44, '재밌는거': 45, '주인': 46, '주인이': 47, '지루해': 48, '팀': 49, '팀에': 50, '팀을': 51, '하세요': 52, '하십니까': 53, '하이': 54, '할수': 55, '할줄': 56, '해봐': 57, '해줘': 58, '헬로우': 59}


##---
@route('/chat', method=['GET','POST'])
def getUserMsg():
    msg = request.json.get('message')
    botResult = botAnswer.sendQ(msg)
    if (botResult == '') or (botResult == None):
        return {'msg' : "잘 이해하지 못했어요."}
    else:
        return {'msg' : botResult}

    
    
#----- 결과 도출 함수 -----
def tokenize(sent):
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

def vocab_result(x, vocab):
    if x.argmax() != 0: return vocab[int(x.argmax())-1]
    else: return False
    
def v_s(data, s_m):
    a = []
    for i in data:
        if i == None:
            a.append(word_idx['UNK'])
        else:
            a.append(word_idx[i])
    return (pad_sequences([a], maxlen=s_m))

def ref_result(result1, result2, mode='detail'):
    a = distance.euclidean(result1[0], result2[0])
    c =  vocab_result(result2[0], vocab)
    d = max(result2[0])
    if c != False:
        b = c
    if mode == 'detail':
        print('detail_ref / acc :', a, '/', max(result2[0]))
    
    if mode == 'simple' or mode == 'detail':
        print('연관도 :', a)
        print('정확도 :', d)
        if b:
            print('정답 :', ''.join(b))
        else:
            print('답변없음')
    return a, d

def answer_result(result1, threshold=0.9):
    a, b = [], []
    x = vocab_result(result1[0], vocab)
    if x != False:
        if max(result1[0]) > threshold:
            a.append(x)
            a.append(" ")
            a.append(max(result1[0]))
            a.append(" ")
            b.append(x)
            b.append(" ")
        else:
            return False, False
    
    if a != []:
        return ''.join([str(i) for i in a]), ''.join([str(i) for i in b])
    else:
        return False, False


#답변 불러오기
with codecs.open('./answers.json', 'r', encoding='UTF-8') as json_data:
    answers_data = json.load(json_data)['answers']
json_data.close()

class InputAura:
    def __init__(self):
        self.Input_Data = []

    def sendQ(self, Q):
        try:
            threshold = 0.9      #0.9
            ref_threshold = 1.5      #1.5

            
            q_data = tokenize(Q)
            ref_data,Input_Data_A = [], []
            m_acc = []
            results = []
            result_no = 0
            final_answer = None
            
            #맥락 flatten
            Input_Data_F = [y for x in self.Input_Data for y in x]
            
            #맥락 없음
            print('맥락정보 : 없음')
            results.append(model.predict([v_s(tokenize('UNK'), story_maxlen), v_s(q_data, query_maxlen)]))
            _, a = ref_result(results[0], results[0], mode='simple')     #detail / simple / none
            m_acc.append(a)
            
            result_no += 1
            print()

            #각 맥락 별 결과 비교값
            j = []
            for i in self.Input_Data:
                j.extend(i)
                print('맥락정보 :', end="")
                for f in j:
                    print(vocab[f-1], end=" ")
                print()
                
                result = model.predict([pad_sequences([j], story_maxlen, truncating='post'), v_s(q_data, query_maxlen)])
                a, b = ref_result(results[0], result, mode='simple')     #detail / simple / none
                
                if a < ref_threshold:
                    results.append(result)
                    m_acc.append(b)
                    result_no += 1
                print()
            
            #순위 선정하기
            rank_1 = np.argmax(m_acc)

            for i in range(rank_1):
                ref_data.append(self.Input_Data[i])

            #무맥락 답변
            print('무맥락답변 :', end=' ')
            an_no, f_an_no = answer_result(results[0], threshold=threshold)
            if an_no == False:
                an_no, f_an_no = '무슨 뜻인지 모르겠어요', '무슨 뜻인지 모르겠어요'
            print(an_no)

            #최적맥락 답변
            print('최적맥락답변 :', end=' ')
            an_A, f_an_A = answer_result(results[rank_1], threshold=threshold)
            
            if an_A == False:
                an_A, f_an_A = '무슨 뜻인지 모르겠어요', '무슨 뜻인지 모르겠어요'
            print(an_A)

            if ref_data == []:
                final_answer = f_an_no
            else:
                final_answer = f_an_A

            #이전 대화 저장하기
            x_d = []
            for i in v_s(q_data, query_maxlen):
                for j in reversed(i):
                    if j != 0:
                        x_d.insert(0, j)

            self.Input_Data = ref_data + [x_d]

            #종합
            print("=================================")
            print('질문 :',q_data)
            print('전체맥락 :', end=" ")
            for i in Input_Data_F:
                print(vocab[i-1], end=" ")
            print()
            for i in answers_data:
                if i['ID'] == final_answer.strip():
                    final_answer = i['AN']      
            print("\033[1m\033[31m최종답변 :", final_answer)
            
        except KeyError:
            print('※ 사전에 있는 단어를 입력해 주세요.')
            print(vocab)
        return final_answer

botAnswer = InputAura()




# Launch the BottlePy dev server 
import wsgiref.simple_server, os
wsgiref.simple_server.WSGIServer.allow_reuse_address = 0
if os.environ.get("PORT"):
    hostAddr = "0.0.0.0"
else:
    hostAddr = "localhost"

if __name__ == '__main__':
    bottle.run(host=hostAddr, port=int(os.environ.get("PORT", 8080)), debug=True)
    # bottle.run(host=IP, port=int(os.environ.get("PORT", 8080)), debug=True)
