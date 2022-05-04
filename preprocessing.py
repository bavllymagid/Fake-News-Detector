import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
from string import punctuation
from sklearn.metrics import *



def stop_word_remove(text) :
    table = {ord(char): u' ' for char in punctuation}
    tokenized = [nltk.word_tokenize(review.translate(table)) for review in text]
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words.append("“")
    stop_words.append("”")
    stop_words.append("’")
    stop_words.append("‘")
    stop_words.append("—")
    new_list = [[token.lower() for token in tlist if token not in punctuation 
                 and token.lower() not in stop_words] for tlist in tokenized]
    return new_list

def vectors_build(word_vectors, cleaned, word_model, length):
    for i in range(0, length):
        word_vectors[i] = 0
        for word in cleaned[i]:
            word_vectors[i] += word_model.wv[word]
        if len(cleaned[i]) != 0:
            word_vectors[i] = word_vectors[i] / len(cleaned[i])
    return word_vectors


def new_report(y0_test, y0_pred):
    print ("  Accuracy: {:.5f}  Precision: {:.5f}  Recall: {:.5f}  F-1: {:.5f}"
           .format(accuracy_score(y0_test, y0_pred), precision_score(y0_test, y0_pred), 
                   recall_score(y0_test, y0_pred), f1_score(y0_test, y0_pred)))