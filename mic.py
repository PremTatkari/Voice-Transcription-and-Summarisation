#pip install SpeechRecognition
#pip install nltk
#in IDLE shell write following commands


import speech_recognition as sr
import pyaudio
import nltk
from time import sleep
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
r = sr.Recognizer()

text1 = """There are many techniques available to generate extractive summarization.
To keep it simple, I will be using an unsupervised learning approach to
find the sentences similarity and rank them. Summarization can be
defined as a task of producing a concise and fluent summary while
preserving key information and overall meaning. One benefit of this will
be, you do not need to train and build a model prior start using it for
your project. It is good to understand Cosine similarity to make the best
use of the code you are going to see. Cosine similarity is a measure of
similarity between two non-zero vectors of an inner product space that
measures the cosine of the angle between them. Its measures cosine of
the angle between vectors. The angle will be 0 if sentences are similar.""" 

text = ""
list = []
go = True

def sumarizer():
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text)
    
       
    freqTable = dict()
    for word in words:
        word = word.lower()
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    sentences = sent_tokenize(text)
    sentenceValue = dict()
       
    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq
       
       
       
    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]

    
    try:    
        average = int(sumValues / len(sentenceValue))
    
    except:
        print("Not enough sentences")

    summary = ""
    
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
            summary += sentence
    print("\nSummary: " + summary)
    list.append({"summary": summary})
    return

def transcript():
    global text
    text = ""
    MyText = ""
    print("Listening")
    list.clear()       
    global go
    while go:    
        try:
            with sr.Microphone() as source2:
            
                r.adjust_for_ambient_noise(source2, duration=0.2)
                audio2 = r.listen(source2)
                MyText = r.recognize_google(audio2)
                sleep(1)
                MyText = MyText.capitalize() + ". "
                print(MyText)
                list.append({"text": MyText})
                text = text + MyText
                sleep(1)  
        except:
            continue


