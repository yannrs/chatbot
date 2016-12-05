# coding=utf-8
from Knowledges.preprocessing import *
from core_idea import *

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
    print "ideas_merged", len(ideas_merged)#, ideas_merged

    # Cut and shuffle data to have a good learning
    ideas_learn = preproc_learn(ideas_merged)
    print "ideas_learn", len(ideas_learn)#, ideas_learn

    cut = int(len(ideas_learn)*RATE_LEARNER)
    print "cut", cut
    random.shuffle(ideas_learn)
    # set that we'll train our classifier with
    training_set = ideas_learn[:cut]
    # set that we'll test against.
    testing_set = ideas_learn[cut:]

    # Learn overall pattern => K-mean
    Kmeans_classifier = SklearnClassifier(KMeans(n_clusters=len(ideas_txts)))
    Kmeans_classifier.train(ideas_learn)
    Kmeans_classifier
    # print("Kmeans_classifier accuracy percent:", (nltk.classify.accuracy(Kmeans_classifier, testing_set))*100)

    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print("Classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
    classifier.show_most_informative_features(15)

    # test_learner(training_set, testing_set)

    ##################
    ## Save the Learner

    # Save the model
    save_classifier = open(PATH + "Models\k-means_learner.pickle", "wb")
    pickle.dump(Kmeans_classifier, save_classifier)
    save_classifier.close()
    save_classifier = open(PATH + "Models\\naivebayes.pickle", "wb")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()

    # Save features extracted
    saveData(PATH + "save.csv", word_features)
    saveIdeas(PATH + 'saveIdeas.csv', ideas_txts)

    print ">>>>>>>>>>>>>>>>>>>><"
    print "FINISH !"

    return -1


if __name__ == '__main__':
    main_preprocess()