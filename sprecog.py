import speech_recognition as sr
import pyaudio
import numpy as np


def Voice_To_Text():
    final_text = ['[CLS]']
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("請開始說話:")                     # print 一個提示 提醒你可以講話了
        r.adjust_for_ambient_noise(source)     # 函數調整麥克風的噪音:
        audio = r.listen(source)
    try:
        Text = r.recognize_google(audio, language="zh-TW")
    except sr.UnknownValueError:
        Text = "無法翻譯"
    except sr.RequestError as e:
        Text = "無法翻譯{0}".format(e)

    final_text.extend(list(Text))  # output sentence list

    return final_text


def formatoutput(final_text):
    BIO_format = list('O'*len(final_text))  # load sentence list

    print(final_text)
    print(list(range(0, len(final_text))))
    m = input('get(0) or spend(1) money:')
    B_item = input('B-item location:')
    I_item = input('I-item end location:')
    B_money = input('B-money location:')
    I_money = input('I-money end location:')

    BIO_format[0] = int(m)
    BIO_format[int(B_item)] = 'B-time'
    BIO_format[int(B_item)+1: int(I_item) +
               1] = ['I-item' for i in range(int(I_item) - int(B_item))]
    BIO_format[int(B_money)] = 'B-money'
    BIO_format[int(B_money)+1: int(I_money) +
               1] = ['I-money' for i in range(int(I_money) - int(B_money))]
    return BIO_format


Text = Voice_To_Text()
BIO_Text = formatoutput(Text)

print(Text)
