# coding=utf-8
from variables import WEIGHT_SENTENCE, MAIN_ATTRIBUTE
from Knowledges.preprocessing import *
from core_idea import *

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from time import time
MAX_FEATURES = 10000
class Concept:
    def __init__(self, txt="", label=""):
        self.ideas = []     # Ideas used to describe the concept, and to train the model
        self.feature_idea = []
        self.feature_concept = []
        self.text = txt     # Raw text used to create ideas
        self.label = label  # Main label of this concept
        self.model = -1     # Model to decide if a text is part of this concept

    def __repr__(self):
        return str(self.label) + ' - Nb ideas:' + str(len(self.ideas)) + ' ; Nb features:' + str(len(self.feature_idea))

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
        self.generate_global_features()
        self.ideas = generateIdeas(self.text)

        # Update Ideas, by adding link to features selected
        update_ideas(self.ideas, self.feature_idea)
        # print "ideas_txts", len(self.ideas)#, ideas_txts
        # print "feature_idea", len(self.feature_idea), self.feature_idea
        # a = merge_synonym(self.feature_idea)
        # print 'feature_idea', len(a), a
        # print 'ideas', len(self.ideas)#, self.ideas

        print("generate done in %0.3fs" % (time() - t0))
        return -1


    def generate_global_features(self):
        # t0 = time()
        ## Clean the text & Vectorize words
        # vectorizer = TfidfVectorizer(max_df=0.5, max_features=MAX_FEATURES,
        #                              min_df=2, stop_words='english',
        #                              ngram_range=(1, 2), lowercase=True,
        #                              use_idf=True)
        vectorizer = TfidfVectorizer(max_features=MAX_FEATURES,
                                     stop_words='english',
                                     ngram_range=(1, 2), lowercase=True,
                                     use_idf=True)
        n_txt = ' '.join(merge_synonym(self.text.split(' '), True))
        # print len(n_txt), len(self.text)
        X = vectorizer.fit_transform([self.text])
        self.feature_idea = vectorizer.get_feature_names()
        self.feature_concept = X
        print "len(self.feature_idea)", len(self.feature_idea)
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

        self.feature_concept = X


    def generate_model(self):
        return -1



if __name__ == '__main__':
    text = "﻿Other Georgia Tech-affiliated buildings in the area host the Center for Quality Growth and Regional Development, the Georgia Tech Enterprise Innovation Institute, the Advanced Technology Development Center, VentureLab, and the Georgia Electronics Design Center. Technology Square also hosts a variety of restaurants and businesses, including the headquarters of notable consulting companies like Accenture and also including the official Institute bookstore, a Barnes & Noble bookstore, and a Georgia Tech-themed Waffle House."
    text = text.decode('utf-8')
    # text = readAllData()
    c = Concept(text)
    c.generate()

