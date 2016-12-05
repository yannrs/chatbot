# coding=utf-8
from core_concept import Concept, MAX_FEATURES
from Knowledges.preprocessing import *
from classifier_select import train_classifier, save_knowledge_models, save_knowledge_generic, load_knowledge_topics
from classifier_select import save_knowledge_topics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans

from time import time


MODE = 2
SAVE_KNOWLEDGE = 1


""" From files create the knowledge which will be used by the bot
Input:
    - None
Output:
    - [ {'name': , 'concept':[ concept, ...]}, ... ]
"""
def create_knowledge():

    ## Read Data -> list of text
    data = readAllData_dico()

    # ## Create concept from data according to one method
    # topics = exploit_txt_ap2(data)
    # print topics
    #
    # ## Learn to dissociate Concepts
    # model = classify_concept(topics)

    ###########################
    ### General analysis
    models, vectorizer, X, y, label_y = exploit_txt_ap3(data)

    ## Save model
    save_knowledge_models(models)
    save_knowledge_generic(vectorizer, 'vectorizer')


    ###########################
    ### Local analysis
    if MODE == SAVE_KNOWLEDGE:
        topics = exploit_txt_ap2(data, X, label_y)
        print topics

        save_knowledge_topics(SAVED_CONCEPTS, topics)
    else:
        topics = exploit_txt_ap2_wt_load(SAVED_CONCEPTS)
        print topics

    return topics


def classify_concept(topics):
    training_set_X = []
    training_set_y = []
    for list_concept in topics:
        for concept in list_concept['concept']:
            y, X = concept.get_data_to_learn()
            training_set_X.append(X)
            training_set_y.append(y)

    print training_set_y, training_set_X

    results = train_classifier(X, y)
    return results

### Approach 1
## Analyse file by file, where one file contains one concept by line

"""
Input:
    - Data: [ {'filename': 'name1.csv', 'line': [ String ], }, ... ]
Ouput:
    - [{'name': String, 'concept': [concept1, ...]}, ...]
"""
def exploit_txt_ap1(data):
    ### For each file
    ## For each line
    # Create a concept from the line:
    #   -> Find ideas
    #   -> Create a bag of word
    #   -> Extract features to characterize the line
    ## -> Train a classifier on concepts

    topics = []

    for file in data:
        topic = {'name': file['name'], 'concept': []}
        for line in file['text']:
            if len(line) > 2:
                c = Concept(line, file['name'])
                c.generate()

                topic['concept'].append(c)
        topics.append(topic)

    return topics


### Approach 2
## Analyse file by file, where one file contains only one concept
"""
Input:
    - Data: [ {'filename': 'name1.csv', 'line': [ String ], }, ... ]
Ouput:
    - [{'name': String, 'concept': [concept]}, ...]
"""
def exploit_txt_ap2(data, X, label_y):
    ### For each file
    # Create a concept from the file:
    #   -> Find ideas
    #   -> Create a bag of word
    #   -> Extract features to characterize the line
    ## -> Train a classifier on concepts

    topics = []

    index = 0
    for file in data:
        topic = {'name': file['name'], 'concept': []}
        line = ' '.join(file['text'])
        if len(line) > 2:
            c = Concept(line, label_y[file['name']], X[index:(index+len(file['text']))])
            c.generate()
            index += len(file['text'])

            topic['concept'].append(c)
        topics.append(topic)
        # if index > 0:
        #     break

    return topics


def exploit_txt_ap2_wt_load(filename):
    topics = load_knowledge_topics(filename)

    for concepts in topics:
        for concept in concepts['concept']:
            concept.generate_partial()

    return topics

### Approach 3
## Analyse everything as one file
"""
Input:
    - Data: [ {'filename': 'name1.csv', 'line': [ String ], }, ... ]
Ouput:
    - [{'name': String, 'concept': [concept]}, ...]
"""
def exploit_txt_ap3(data):
    ### Merge each file
    # -> Create a bag of word
    # -> Extract features to characterize the bags
    ## -> Train a classifier on concepts

    text = []
    text_svm = []
    text_y = []
    for file in data:
        text += file['text']
        text_y += [file['name'] for i in xrange(len(file['text']))]
        text_svm += [(file['name'], line) for line in file['text']]

    vectorizer = TfidfVectorizer(max_df=0.5, max_features=MAX_FEATURES,
                                 min_df=2, stop_words='english',
                                 ngram_range=(1, 2), lowercase=True,
                                 use_idf=True)
    # n_txt = ' '.join(merge_synonym(text.split(' '), True))
    # print len(n_txt), len(self.text)
    X = vectorizer.fit_transform(text)
    print X.shape

    ## Test
    print vectorizer.get_feature_names()
    print len(vectorizer.get_feature_names())

    # svd = TruncatedSVD(8)
    # normalizer = Normalizer(copy=False)
    # lsa = make_pipeline(svd, normalizer)
    #
    # X = lsa.fit_transform(X)
    # print len(X), len(X[0])
    true_k = len(data)
    print true_k
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


    # Convert label to array
    label_y = {}
    last = 0
    training_y = []
    for y in text_y:
        if y not in label_y:
            label_y[y] = last
            last += 1
        training_y.append(label_y[y])
    print len(training_y)

    results = train_classifier(X, training_y)

    save_knowledge_generic(vectorizer, 'vectorizer')

    return results, vectorizer, X, training_y, label_y




if __name__ == '__main__':
    create_knowledge()