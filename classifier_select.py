# coding=utf-8
import pickle
from variables import PATH, NAME_CLASSIFIERS
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from time import time
import json
from core_concept import Concept
import os

#################################################
###       Save functions

def save_knowledge_models(models):
    for model, name in models:
        save_knowledge_generic(model, name)


def save_knowledge_topics(filename, topics):
    i = 0
    for concepts in topics:
        file = open('Models\\' + str(i) + '_' + filename, 'w')
        concepts_j = '##'.join([json.dumps(concept.toSave()) for concept in concepts['concept']])
        s = {'name': str(concepts['name']), 'concept': concepts_j}
        file.writelines(json.dumps(s) + '\n')
        i+=1
        file.close()


def save_knowledge_generic(data, name):
    save_classifier = open(PATH + "Models\\"+name+".pickle", "wb")
    pickle.dump(data, save_classifier)
    save_classifier.close()



#################################################
###       Load functions

def load_knowledge():
    global ALL_CLASSIFIER, VECTORIZER
    ALL_CLASSIFIER = load_knowledge_models()
    VECTORIZER = load_knowledge_generic('vectorizer')


def load_knowledge_models(names=NAME_CLASSIFIERS):
    classifier_all = []
    for name in names:
        classifier_all.append(load_knowledge_generic(name))
    return classifier_all


def load_knowledge_topics(filename):
    out = []
    listFile = os.listdir('Models')
    n = len(listFile)-1
    for i in range(0, n):
        # if i < 25:
        #     continue
        file = open('Models\\' + str(i) + '_' + filename, 'r')
        out_c = {}
        concepts_j = file.readlines()[0]
        concepts_t = json.loads(concepts_j)
        concepts = concepts_t['concept'].split('##')
        out_c['name'] = concepts_t['name']
        out_c['concept'] = [Concept().loadConcept(json.loads(concept)) for concept in concepts]
        out.append(out_c)
        file.close()
    return out


def load_knowledge_generic(name):
    classifier_f = open(PATH + "Models\\"+name+".pickle", "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    return classifier


VECTORIZER = load_knowledge_generic('vectorizer')
ALL_CLASSIFIER = load_knowledge_models()



#################################################
###       General Learner & predicter

def train_classifier(X, y):
    results = []
    for clf, name in (
                    (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
                    (Perceptron(n_iter=50), "Perceptron"),
                    (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
                    (KNeighborsClassifier(n_neighbors=10), "kNN"),
                    (RandomForestClassifier(n_estimators=100), "Random forest"),
                    (MultinomialNB(alpha=.01), "MultinomialNB"),
                    (BernoulliNB(alpha=.01), "BernoulliNB"),
                    (NearestCentroid(), "NearestCentroid")):
        print('=' * 80)
        print(name)
        clf.fit(X, y)
        results.append((clf, name))

        t0 = time()
        pred = clf.predict(X)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)

        score = metrics.accuracy_score(y, pred)
        print("accuracy:   %0.3f" % score)

    return results


def general_kw_predict(X):
    y_guess = []
    y_max = -1
    y_proba_max = -1
    y_name = ""
    i = 0
    for clf in ALL_CLASSIFIER:
        y_guess.append((clf.predict(X), clf.predict_proba(X)))
        if y_proba_max == -1:
            y_max = y_guess[-1][0][0]
            y_proba_max = y_guess[-1][1][0][y_max]
            y_name = NAME_CLASSIFIERS[0]
        if y_proba_max < y_guess[-1][1][0][y_guess[-1][0][0]]:
            y_max = y_guess[-1][0][0]
            y_proba_max = y_guess[-1][1][0][y_max]
            y_name = NAME_CLASSIFIERS[i]
        i += 1

    print 'general_kw_predict', y_proba_max, y_max, y_name
    print y_guess

    return y_max


def get_model_knn(training_set, n=3):
    return NearestNeighbors(n_neighbors=n, algorithm='ball_tree').fit(training_set)

