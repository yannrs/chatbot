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
from sklearn.cluster import KMeans

import pickle
import random
import os

from core_idea import *

path = 'C:\Users\yann\Documents\Mes fichiers\Cours\GeorgiaTech\Fall 2016\CS   7637 - Knowledge based AI\Project3\Data\\'
ps = PorterStemmer()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
NB_FEATURES = 30
RATE_LEARNER = 0.8

def loadFile(filename):
    file = open(path+filename, 'r')
    text = [line[:-2].decode('utf-8') for line in file.readlines()]
    file.close()
    return text

    # text = "Other Georgia Tech-affiliated buildings in the area host the Center for Quality Growth and Regional Development, the Georgia Tech Enterprise Innovation Institute, the Advanced Technology Development Center, VentureLab, and the Georgia Electronics Design Center. Technology Square also hosts a variety of restaurants and businesses, including the headquarters of notable consulting companies like Accenture and also including the official Institute bookstore, a Barnes & Noble bookstore, and a Georgia Tech-themed Waffle House.[57][61]"
    # text = text.decode('utf-8')
    # return [text]

stop_words = set(stopwords.words('english'))

def preproc_it(data):
    return [lemmatizer.lemmatize(w).lower().encode('ascii', 'ignore') for w in word_tokenize(data) if not w in stop_words]


def preproc_tag(data):
    custom_sent_tokenizer = PunktSentenceTokenizer(data)
    tokenized = custom_sent_tokenizer.tokenize(data)
    return tokenized


def preproc(data):
    txt = []
    for text in data:
        # txt.append(generateIdeas(text))
        txt.append(preproc_it(text))
        # txt += preproc_it(text)
        # txt.append(preproc_tag(text))
    return txt


""" Generate ideas from one paragraphs
Input:
    - data: []
Output:
    - [ [ideas] ]
"""
def preproc_ideas(data):
    txt = []
    id = 0
    for text in data:
        txt.append(generateIdeas(text, id))
        id += 1
    return txt


""" Merge sublist =>[[]] to []
"""
def mergeData(data):
    n_data = []
    for d in data:
        n_data += d
    return n_data

""" Update all objects ideas with feature selected
"""
def update_ideas(ideas, word_features):
    for idea in ideas:
        for idea2 in idea:
            idea2.add_feature(word_features)

""" Convert ideas to something learnable
"""
def preproc_learn(ideas):
    data_to_learn = []
    for idea in ideas:
        data_to_learn.append((idea.features, idea.id))
    return data_to_learn


def readAllData():
    listFile = os.listdir(path + 'Courses')
    listFile = ['Courses\\' + name for name in listFile]
    listFile = []
    listFile.append('gatech_wiki_clean.csv')
    data = []
    for file in listFile:
        data += loadFile(file)
    return data


def saveData(fileName, data):
    file = open(path + fileName, 'w')
    for d in data:
        file.writelines(str(d)+"\n")
    file.close()


def main_preprocess():
    ##################
    ## Create Features

    # Read all the data
    data = readAllData()
    print "data", len(data), data

    # Clean data
    data_cleaned = preproc(data)
    print "data_cleaned", len(data_cleaned),  data_cleaned

    # Merge data
    data_merged = mergeData(data_cleaned)
    print "data_merged", len(data_merged),  data_merged

    # Rank words
    all_words = nltk.FreqDist(data_merged)
    print "all_words", len(all_words), all_words

    # Extract main features
    word_features = list(all_words.keys())[:NB_FEATURES]
    print "word_features", len(word_features), word_features



    ##################
    ## Create Ideas

    # From data collected, generate one idea for each subtext: [ ideas, ...]
    ideas_txts = preproc_ideas(data)
    print "ideas_txts", len(ideas_txts)#, ideas_txts



    ##################
    ## Characterized Ideas from features

    # Update Ideas, by adding link to features selected
    update_ideas(ideas_txts, word_features)
    print "ideas_txts", len(ideas_txts)#, ideas_txts



    ##################
    ## Learning from Ideas

    # Merge all set of ideas
    ideas_merged = mergeData(ideas_txts)
    print "ideas_merged", len(ideas_merged), ideas_merged

    # Cut and shuffle data to have a good learning
    ideas_learn = preproc_learn(ideas_merged)
    print "ideas_learn", len(ideas_learn), ideas_learn

    cut = int(len(ideas_learn)*RATE_LEARNER)
    print "cut", cut
    random.shuffle(ideas_learn)
    # set that we'll train our classifier with
    training_set = ideas_learn[:cut]
    # set that we'll test against.
    testing_set = ideas_learn[cut:]

    # Learn overall pattern => K-mean
    Kmeans_classifier = SklearnClassifier(KMeans(n_clusters=len(ideas_txts)))
    Kmeans_classifier.train(training_set)
    # print("Kmeans_classifier accuracy percent:", (nltk.classify.accuracy(Kmeans_classifier, testing_set))*100)

    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print("Classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
    classifier.show_most_informative_features(15)

    # test_learner(training_set, testing_set)

    ##################
    ## Save the Learner

    # Save the model
    save_classifier = open("k-means_learner.pickle", "wb")
    pickle.dump(Kmeans_classifier, save_classifier)
    save_classifier.close()

    # Save features extracted
    saveData("save.csv", word_features)
    saveIdeas('saveIdeas.csv', ideas_txts)

    print ">>>>>>>>>>>>>>>>>>>><"
    print "FINISH !"

    return -1


def loadClassifier(name):
    classifier_f = open(name, "rb")

    classifier = pickle.load(classifier_f)
    classifier_f.close()
    return classifier


def saveIdeas(filename, ideas):
    file = open(filename, 'w')
    for idea in ideas:
        s = ';'.join([str(idea2.toSave()) for idea2 in idea])
        file.writelines(s + '\n')
    file.close()


def loadIdeas(filename):
    out = [[]]
    file = open(filename, 'r')
    for ideas in file.readlines():
        list_idea = ideas.split(';')
        new_ideas = []
        for idea2 in list_idea:
            new_ideas.append(Idea.loadIdea(idea2))
        out.append(new_ideas)
    file.close()
    return new_ideas

def test_learner(training_set, testing_set):
    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)
    print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(training_set)
    print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training_set)
    print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

    SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
    SGDClassifier_classifier.train(training_set)
    print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

    SVC_classifier = SklearnClassifier(SVC())
    SVC_classifier.train(training_set)
    print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_set)
    print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)



if __name__ == '__main__':
    main_preprocess()