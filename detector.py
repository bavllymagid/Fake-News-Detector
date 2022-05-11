import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from preprocessing import *
from nltk.stem import PorterStemmer
from sklearn import metrics

data = pd.read_csv('news.csv')

data.drop('Unnamed: 0', inplace=True, axis=1)

label_trans = lambda i: 0 if i == 'FAKE' else 1

data.label = data.label.apply(label_trans)
y = data['label']

raw_text = data.text + data.title

ps = PorterStemmer()

raw_text = [[ps.stem(word) for word in sentence.split(" ")] for sentence in raw_text]


filtered_sentence = stop_word_remove(raw_text)

word_model = Word2Vec(filtered_sentence, min_count = 1, 
                      window = 5, sg = 1)
word_vectors = np.zeros((len(data), 100))
word_vectors = vectors_build(word_vectors, filtered_sentence, word_model, len(data))
X_train, X_test, Y_train, Y_test = train_test_split(word_vectors, y, test_size=0.3, random_state=42)
model = LogisticRegression().fit(X_train, Y_train)
y_pred = model.predict(X_test)

print("mean squared error: " , metrics.mean_squared_error(Y_test, y_pred))
new_report(Y_test, y_pred)




