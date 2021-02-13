from utils import *
import os
import random
from tkinter import *
import speech_recognition as sr
import pyaudio


def Voice_To_Text():
    final_text = []
    r = sr.Recognizer()
    with sr.Microphone() as source:
        entryVariable.delete(0, 'end')
        label1.config(text='Lets talk')
        print("請開始說話:")

        # adjust microphone noise
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        input_text = r.recognize_google(audio, language="zh-TW")
    except sr.UnknownValueError:
        input_text = "無法翻譯"
    except sr.RequestError as e:
        input_text = "無法翻譯{0}".format(e)

    final_text.extend(list(Text))  # output sentence list
    entryVariable.insert(0, final_text)
    label1.config(text='')
    print(final_text)
    return final_text


def confirm():
    sen = entryVariable.get()
    print(sen)
    slots = predict_sf(
        sen, "C:\\Users\\leosh\\OneDrive\\Desktop\\weights-improvement-19.hdf5")
    intent = get_intent(sen)

    item, money = extract(intent, slots, sen)
    text = label4.cget("text") + item + "\n"
    label4.configure(text=text)
    text = label5.cget("text") + money + "\n"
    label5.configure(text=text)


def clean():
    label4.configure(text="")
    label5.configure(text="")


window = Tk()
window.geometry("350x200")

label1 = Label(window, text='')
btn1 = Button(window, text="麥克風", bg="yellow", command=Voice_To_Text)
entryVariable = Entry(window, text="這是文字方塊", width=30)
btn2 = Button(window, text="確定", bg="yellow", command=confirm)
label2 = Label(window, text='項目')
label3 = Label(window, text='收支出  ')
labele = Label(window, text='')
label4 = Label(window, text='')  # item
label5 = Label(window, text='')  # price
btn3 = Button(window, text="清空", bg="yellow", command=clean)

label1.grid(row=0, column=1)
btn1.grid(row=1, column=0)
entryVariable.grid(row=1, column=1)
btn2.grid(row=1, column=2)
labele.grid(row=2, column=0)
label2.grid(row=3, column=0)
label3.grid(row=3, column=2)
label4.grid(row=4, column=0)
label5.grid(row=4, column=2)
btn3.grid(row=1, column=3)
# window.maxsize(400,400) #int
window.mainloop()
