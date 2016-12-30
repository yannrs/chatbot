# coding=utf-8
from core_concept import Concept, MAX_FEATURES
from Knowledges.preprocessing import *
from classifier_select import train_classifier, save_knowledge_models, save_knowledge_generic, load_knowledge_topics
from classifier_select import save_knowledge_topics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans

from util import plot_knowledge
from time import time

MODE = 2        # Mode 1: Create all the knowledge; Mode 2: Used saved knowledge
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

    ###########################
    ### General analysis
    models, vectorizer, X, y, label_y = exploit_txt_ap3(data)
    save_knowledge_models(models)
    # plot_knowledge(X, label_y)
    ### Local analysis
    if MODE == SAVE_KNOWLEDGE:
        topics = exploit_txt_ap2(data, X, label_y)
        print topics

        ## Save model
        save_knowledge_generic(vectorizer, 'vectorizer')
        save_knowledge_topics(SAVED_CONCEPTS, topics)
        map_knowledge(topics)
    else:
        topics = exploit_txt_ap2_wt_load(SAVED_CONCEPTS)
        map_knowledge(topics)

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
    # plot_knowledge(X, [1 for k in xrange(26)], "All")

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
        attibute_concept['nb_text_idea_word_mean'] = 0
        attibute_concept['nb_text_idea_word_min'] = concept.ideas[0].text.count(' ')
        attibute_concept['nb_text_idea_word_max'] = concept.ideas[0].text.count(' ')
        attibute_concept['length_text_idea'] = []
        attibute_concept['nb_text_idea_word'] = []
        for idea in concept.ideas:
            attibute_concept['length_text_idea'].append(len(idea.text))
            attibute_concept['length_text_idea_mean'] += len(idea.text)
            if len(idea.text) < attibute_concept['length_text_idea_min']:
                attibute_concept['length_text_idea_min'] = len(idea.text)
            if len(idea.text) > attibute_concept['length_text_idea_max']:
                attibute_concept['length_text_idea_max'] = len(idea.text)
            count_words = idea.text.count(' ')
            attibute_concept['nb_text_idea_word_mean'] += count_words
            if count_words < attibute_concept['nb_text_idea_word_min']:
                attibute_concept['nb_text_idea_word_min'] = count_words
            if count_words > attibute_concept['nb_text_idea_word_max']:
                attibute_concept['nb_text_idea_word_max'] = count_words
            attibute_concept['nb_text_idea_word'].append(count_words)
        attibute_concept['length_text_idea_mean'] /= float(max(attibute_concept['nb'], 1))
        attibute_concept['nb_text_idea_word_mean'] /= float(max(attibute_concept['nb'], 1))

        stats.append(attibute_concept)

    saveData_dico('map_knowledge.csv', stats)


if __name__ == '__main__':
    create_knowledge()