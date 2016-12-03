# coding=utf-8
from core_concept import *
from Knowledges.preprocessing import *
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier

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
    models, X, y = exploit_txt_ap3(data)

    ## Save model
    save_knowledge_models(models)


    ###########################
    ### Local analysis
    concepts = exploit_txt_ap2(data)
    print concepts



    return -1


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
def exploit_txt_ap2(data):
    ### For each file
    # Create a concept from the file:
    #   -> Find ideas
    #   -> Create a bag of word
    #   -> Extract features to characterize the line
    ## -> Train a classifier on concepts

    topics = []

    for file in data:
        topic = {'name': file['name'], 'concept': []}
        line = ' '.join(file['text'])
        if len(line) > 2:
            c = Concept(line, file['name'])
            c.generate()

            topic['concept'].append(c)
        topics.append(topic)

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

    return results, X, training_y



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


def save_knowledge_models(models):
    for model, name in models:
        save_classifier = open(path + "Models\\"+name+".pickle", "wb")
        pickle.dump(model, save_classifier)
        save_classifier.close()




if __name__ == '__main__':
    create_knowledge()