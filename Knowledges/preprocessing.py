# coding=utf-8

# import sklearn
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
# from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import movie_reviews, wordnet
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

import pickle
import random

from core_idea import Idea

path = 'C:\Users\yann\Documents\Mes fichiers\Cours\GeorgiaTech\Fall 2016\CS   7637 - Knowledge based AI\Project3\Data\\'
ps = PorterStemmer()

lemmatizer = WordNetLemmatizer()

def loadFile(filename):
    # file = open(path+filename, 'r')
    # text = [line[:-2].decode('utf-8') for line in file.readlines()]
    # file.close()

    text = "﻿Other Georgia Tech-affiliated buildings in the area host the Center for Quality Growth and Regional Development, the Georgia Tech Enterprise Innovation Institute, the Advanced Technology Development Center, VentureLab, and the Georgia Electronics Design Center. Technology Square also hosts a variety of restaurants and businesses, including the headquarters of notable consulting companies like Accenture and also including the official Institute bookstore, a Barnes & Noble bookstore, and a Georgia Tech-themed Waffle House.[57][61]"
    text = text.decode('utf-8')
    return [text]


def preproc_it(data):
    stop_words = set(stopwords.words('english'))
    return [lemmatizer.lemmatize(w) for w in word_tokenize(data) if not w in stop_words]


def preproc_tag(data):
    custom_sent_tokenizer = PunktSentenceTokenizer(data)
    tokenized = custom_sent_tokenizer.tokenize(data)
    return tokenized


def preproc(data):
    txt = []
    for text in data:
        txt.append(Idea(text).generate())
        # txt.append(preproc_it(text))
        # txt.append(preproc_tag(text))
    return txt


def main_preprocess():
    data = loadFile('gatech_wiki_clean.csv')

    for line in data:
        print line

    filt_harm_data = preproc(data)

    for line in filt_harm_data:
        print line


    all_words = nltk.FreqDist(filt_harm_data)

    word_features = list(all_words.keys())[:3000]


    ## Learner
    featuresets = word_features
    # set that we'll train our classifier with
    training_set = featuresets[:1900]

    # set that we'll test against.
    testing_set = featuresets[1900:]
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print("Classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
    classifier.show_most_informative_features(15)

    ## Save a model
    save_classifier = open("naivebayes.pickle", "wb")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()

    return -1






if __name__ == '__main__':
    main_preprocess()