# coding=utf-8
from core_concept import Concept, MAX_FEATURES
from Knowledges.preprocessing import *
from classifier_select import train_classifier, save_knowledge_models, save_knowledge_generic, load_knowledge_topics
from classifier_select import save_knowledge_topics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from time import time
import copy

MODE = 2
SAVE_KNOWLEDGE = 1


#################################################
###       Main function

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

    ###########################
    ### General analysis
    models, vectorizer, X, y, label_y = exploit_txt_ap3(data)

    ### Local analysis
    if MODE == SAVE_KNOWLEDGE:
        topics = exploit_txt_ap2(data, X, label_y)
        print topics

        ## Save model
        save_knowledge_models(models)
        save_knowledge_generic(vectorizer, 'vectorizer')
        save_knowledge_topics(SAVED_CONCEPTS, topics)
        map_knowledge(topics)
    else:
        topics = exploit_txt_ap2_wt_load(SAVED_CONCEPTS)
        print topics


    return topics


#################################################
###       Different way to manage the data

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

    return topics


def exploit_txt_ap2_wt_load(filename):
    topics = load_knowledge_topics(filename)

    for concepts in topics:
        for concept in concepts['concept']:
            concept.generate_partial()

    return topics

### Approach 3
""" Analyse everything as one file
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

    ###############################
    ##      Vectorize the text
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=MAX_FEATURES,
                                 min_df=2, stop_words='english',
                                 ngram_range=(1, 3), lowercase=True,
                                 use_idf=True)
    X = vectorizer.fit_transform(text)
    # ## OR
    # n_txt = ' '.join(merge_synonym(text.split(' '), True))
    # X = vectorizer.fit_transform(n_txt)
    print 'X.shape', X.shape

    ## Dimensional reduction
    # svd = TruncatedSVD(8)
    # normalizer = Normalizer(copy=False)
    # lsa = make_pipeline(svd, normalizer)
    #
    # X = lsa.fit_transform(X)
    # print len(X), len(X[0])

    ## Convert label to array
    label_y = {}
    last = 0
    training_y = []
    for y in text_y:
        if y not in label_y:
            label_y[y] = last
            last += 1
        training_y.append(label_y[y])
    print len(training_y)


    ###############################
    ##      Learn from the vector
    true_k = len(data)
    print true_k
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000, verbose="")
    print("Clustering sparse data with %s" % km)
    t0 = time()
    km.fit(X)
    print("done in %0.3fs" % (time() - t0))

    results = train_classifier(X, training_y)

    return results, vectorizer, X, training_y, label_y



############################################################
###       Tests

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


# TODO: MAP of the knowledge = nb of idea by concept, nb of attribute
def map_knowledge(topics):
    stats = []
    for concept_d in topics:
        concept = concept_d['concept'][0]
        attibute_concept = {}
        attibute_concept['nb'] = len(concept.ideas)
        attibute_concept['nb_attribute_idea'] = len(concept.idea_vectorizer.get_feature_names())
        attibute_concept['length_text'] = len(concept.text)
        attibute_concept['length_text_idea_mean'] = 0
        attibute_concept['length_text_idea_min'] = len(concept.ideas[0].text)
        attibute_concept['length_text_idea_max'] = len(concept.ideas[0].text)
        attibute_concept['length_text_idea'] = []
        for idea in concept.ideas:
            attibute_concept['length_text_idea'].append(len(idea.text))
            attibute_concept['length_text_idea_mean'] += len(idea.text)
            if len(idea.text) < attibute_concept['length_text_idea_min']:
                attibute_concept['length_text_idea_min'] = len(idea.text)
            if len(idea.text) > attibute_concept['length_text_idea_max']:
                attibute_concept['length_text_idea_max'] = len(idea.text)
        attibute_concept['length_text_idea_mean'] /= float(max(attibute_concept['nb'], 1))

        stats.append(attibute_concept)

    saveData_dico('map_knowledge.csv', stats)


def plot_knowledge(topics):
    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

if __name__ == '__main__':
    create_knowledge()