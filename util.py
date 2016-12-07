
import numpy as np
import matplotlib.pyplot as plt
import calendar, time

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

i_ = 0

def plot_knowledge(X, y_label, title=""):
    global i_
    i_ = get_id()
    n_digits = int(len(y_label)*0.5)
    reduced_data = PCA(n_components=2).fit_transform(X.toarray())
    kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 0.5, reduced_data[:, 0].max() + 0.5
    y_min, y_max = reduced_data[:, 1].min() - 0.5, reduced_data[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross\n' +
              str(title))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    # plt.imshow(Z, interpolation='nearest',
    #            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    #            cmap=plt.cm.Paired,
    #            aspect='auto', origin='lower')
    # plt.savefig('figures\\' + str(i_) + '_' + str(title) + '_dot.png')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    plt.savefig('figures\\' + str(i_) + '_' + str(title) + '_dot.png')

    # Plot the centroids as a white X
    # centroids = kmeans.cluster_centers_
    # plt.scatter(centroids[:, 0], centroids[:, 1],
    #             marker='x', s=169, linewidths=3,
    #             color='w', zorder=10)
    #
    # plt.savefig('figures\\' + str(i_) + '_' + str(title) + '_all.png')
    i_+=1
    print i_
    # plt.show()

def get_id():
    return int(calendar.timegm(time.gmtime()))

def map_knowledge(topics):
    stats = []
    for concept_d in topics:
        concept = concept_d['concept'][0]
        attibute_concept = {}
        attibute_concept['nb_idea'] = len(concept.ideas)
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

    saveData_dico(str(get_id) + '_map_knowledge.csv', stats)

SEPARATOR = ';'
def saveData_dico(fileName, data):
    file = open(fileName, 'w')
    file.writelines(SEPARATOR.join(data[0].keys())+"\n")
    for d in data:
        file.writelines(SEPARATOR.join([str(d[key]) for key in d])+"\n")
    file.close()
