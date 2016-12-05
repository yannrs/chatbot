# coding=utf-8
from variables import WEIGHT_SENTENCE, MAIN_ATTRIBUTE
from Knowledges.preprocessing import *
from core_idea import *
# from classifier_select import train_cluster, select_cluster

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import numpy as np
from scipy.sparse import hstack, vstack


from time import time
MAX_FEATURES = 10000
class Concept:
    def __init__(self, txt="", label="", features=[]):
        self.ideas = []     # Ideas used to describe the concept, and to train the model

        self.idea_feature_count = []
        self.idea_feature_word = []
        self.idea_vectorizer = -1
        self.idea_model_label = []

        self.concept_feature_count = features
        self.concept_feature_word = []
        self.concept_vectorizer = -1

        self.text = txt     # Raw text used to create ideas
        self.label = label  # Main label of this concept

        self.model = -1     # Model to decide if a text is part of this concept

    def __repr__(self):
        return str(self.label) + ' - Nb ideas:' + str(len(self.ideas)) + ' ; Nb features:' + str(len(self.idea_feature_word))

    """ Convert ideas to something learnable
    """
    def get_data_to_learn(self):
        return self.label, self.feature_concept

    def generate(self):
        t0 = time()
        ## Create the concept from the text
        #   -> Extract features to characterize the text
        #   -> Create a bag of word
        #   -> Find ideas
        self.generate_local_features()
        print("generate_local_features  done in %0.3fs" % (time() - t0))
        t0 = time()
        self.ideas = generateIdeas(self.text, self.label)

        # Update Ideas, by adding link to features selected
        # update_ideas(self.ideas, self.idea_feature_word)
        print("generate  done in %0.3fs" % (time() - t0))
        t0 = time()
        update_ideas_v2(self.ideas, self.idea_vectorizer)

        # a = merge_synonym(self.feature_idea)

        self.generate_model_ideas()

        print("Update done in %0.3fs" % (time() - t0))
        return -1


    def generate_partial(self):
        t0 = time()
        self.generate_local_features()
        update_ideas_v2(self.ideas, self.idea_vectorizer)
        self.generate_model_ideas()


    def generate_local_features(self):
        # t0 = time()
        ## Clean the text & Vectorize words
        vectorizer = TfidfVectorizer(max_features=MAX_FEATURES,
                                     stop_words='english',
                                     ngram_range=(1, 2), lowercase=True,
                                     use_idf=True)
        # n_txt = ' '.join(merge_synonym(self.text.split(' '), True))
        # print len(n_txt), len(self.text)
        X = vectorizer.fit_transform([self.text])
        # print X

        # ## Dimension reduction with LSA
        # # Vectorizer results are normalized, which makes KMeans behave as
        # # spherical k-means for better results. Since LSA/SVD results are
        # # not normalized, we have to redo the normalization.
        # svd = TruncatedSVD(8)
        # normalizer = Normalizer(copy=False)
        # lsa = make_pipeline(svd, normalizer)
        #
        # X = lsa.fit_transform(X)
        # print 'X', X
        # print("generate_global_features done in %0.3fs" % (time() - t0))
        self.idea_feature_word = vectorizer.get_feature_names()
        self.idea_feature_count = X
        self.idea_vectorizer = vectorizer
        print "len(self.idea_feature_word)", len(self.idea_feature_word)

    """ From the set of idea which define this concept, create a cluster of common sense ideas
    Input:
        - None
    Output:
        - None
    """
    def generate_model_ideas(self):
        # TODO: generate a model which can select ideas according to a user input (idea)
        training_set = []
        for idea in self.ideas:
            if training_set == []:
                training_set = idea.features_vect
            else:
                training_set = vstack((training_set, idea.features_vect))

        self.model = train_cluster(training_set)
        self.idea_model_label = self.model.labels_

    """ From an external idea, select the set of idea which are the closest
    Input:
        - idea_user: Idea
    Output:
        - [ idea1, idea2, ...]
    """
    def predict_idea(self, idea_user):
        print idea_user
        # Convert this idea to the environment of this concept
        feature_idea = self.idea_vectorizer.transform([idea_user.text])

        # Predict which ideas are linked to the reference idea
        label = select_cluster(self.model, feature_idea)
        print 'label', len(label), label
        return label

        # # Get idea which are on the same cluster than the user idea
        # useful_ideas = []
        # print 'self.idea_model_label', self.idea_model_label
        # for i in range(0, len(self.ideas)):
        #     if self.idea_model_label[i] == label:
        #         useful_ideas.append(self.ideas[i])
        #
        # return useful_ideas

    def toSave(self):
        content = {}
        content['ideas'] = '&&'.join([json.dumps(idea.toSave()) for idea in self.ideas])
        content['label'] = str(self.label)
        content['text'] = self.text
        # content['vectoriser'] = json.dumps(self.idea_vectorizer)
        return content

    def loadConcept(self, dico):
        ideas = dico['ideas'].split('&&')
        self.ideas = [Idea().loadIdea(json.loads(idea)) for idea in ideas]
        self.label = dico['label']
        self.text = dico['text']
        # self.idea_vectorizer = json.loads(dico['vectorizer'])
        return self

""" Train a Kmean from a set of ideas
Input:
    - X: scipy.sparse.csr_matrix
Output:
    - sklearnModel: MiniBatchKMeans
"""
def train_cluster(X):
    true_k = int(X.shape[0]*0.95)
    print true_k
    print 'X', type(X), X.shape
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000, verbose="")
    print("Clustering sparse data with %s" % km)
    t0 = time()
    km.fit(X)
    print("done in %0.3fs" % (time() - t0))


    # original_space_centroids = svd.inverse_transform(km.cluster_centers_)
    # order_centroids = original_space_centroids.argsort()[:, ::-1]
    # terms = vectorizer.get_feature_names()
    # for i in range(true_k):
    #     print("Cluster %d:" % i)#, end='')
    #     print order_centroids.shape
    #     for ind in order_centroids[i, :10]:
    #         print(' %s' % terms[ind])#, end='')
    #     print()
    return km


""" From a model and features of an idea, select a label
Input:
    - model: Sklearn model
    - X: scipy.sparse.csr_matrix
Output:
    - [label]
"""
def select_cluster(model, X):
    return model.predict(X)

if __name__ == '__main__':
    text = "﻿Other Georgia Tech-affiliated buildings in the area host the Center for Quality Growth and Regional Development, the Georgia Tech Enterprise Innovation Institute, the Advanced Technology Development Center, VentureLab, and the Georgia Electronics Design Center. Technology Square also hosts a variety of restaurants and businesses, including the headquarters of notable consulting companies like Accenture and also including the official Institute bookstore, a Barnes & Noble bookstore, and a Georgia Tech-themed Waffle House."
    text = text.decode('utf-8')
    # text = readAllData()
    c = Concept(text)
    c.generate()

