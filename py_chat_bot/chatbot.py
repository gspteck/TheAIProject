import random, json, pickle, numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

import training

training.trainChatbot()

lemmatizer = WordNetLemmatizer()

intents = json.loads(open("intents.json").read())

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot_model.h5")

def cleanSentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bagOfWords(sentence):
    sentence_words = cleanSentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    
    return np.array(bag)

def predictClass(sentence):
    bow = bagOfWords(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key = lambda x: x[1], reverse = True)
    return_list = []

    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    
    return return_list

def getResponse(intentList, intentJSON):
    tag = intentList[0]["intent"]
    listOfIntents = intentJSON["intents"]

    for i in listOfIntents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result

print("bot is running")

while True:
    message = input("say something: ")
    ints = predictClass(message)
    res = getResponse(ints, intents)
    print(res)
