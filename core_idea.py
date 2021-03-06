# coding=utf-8
from nltk.corpus import wordnet
from nltk.corpus import state_union
from scipy.sparse.linalg import norm

from Knowledges.preprocessing import *
import json, copy

master_tokenizer = PunktSentenceTokenizer(state_union.raw(PATH+"gatech_wiki_clean_v3.csv"))


#################################################
###       Main Class: IDEA
class Idea:
    def __init__(self, text="", id=-1):
        self.id = id
        self.label = id
        self.text = text            # ''
        self.frame = {}             # {'NP':[ 'Charles', ....], 'VB': [...] }
        self.features = {}          # {'word1': 1, 'word2':0, 'word3': 10, .... }
        self.features_vect = []     # count from a main vectorizer
        self.vectorizer = -1

    def __str__(self):
        return 'Idea: ' + str(self.frame)

    def __repr__(self):
        return 'Idea: ' + str(self.id) + '; text:' + self.text + ' [' + str(self.frame) + ']'

    """ From the text, generate a fram which will represent the idea
    """
    def generate(self):
        self.frame = generateIdea(self.text)
        return self

    """ For each word selected as a feature, check if the word is present into the text
    Input:
        - word_feature = [ w1, w2, ...]
    Output:
        - None
    """
    def add_feature(self, word_features):
        self.features = {}
        for word in word_features:
            # self.features['contains({})'.format(word)] = self.text.count(word)
            # self.features['contains({})'.format(word)] = (word in self.text)
            # key = word.encode('ascii', 'ignore')
            # if
            self.features[word] = (word in self.text)
            # self.features[word.encode('utf-8', 'ignore')] = (word in self.text)

    def add_features_vect(self, vectorizer):
        self.features_vect = vectorizer.transform([self.text])
        self.vectorizer = vectorizer
        return self

    def toSave(self):
        content = {}
        content['id'] = self.id
        content['text'] = self.text
        content['frame'] = json.dumps(self.frame)
        content['features'] = json.dumps(self.features)
        # content['feature_vect'] = json.dumps(self.features_vect)
        return content

    def loadIdea(self, dico):
        self.id = dico['id']
        self.label = dico['id']
        self.text = dico['text']
        self.frame = json.loads(dico['frame'])
        self.features = json.loads(dico['features'])
        # self.features_vect = json.loads(dico['feature_vect'])
        return self

    """ Try to quantify the distance between 2 ideas
    """
    def compare(self, idea):
        out = 0
        error = ""
        for label in self.frame.keys():
            label_wordnet = select_wordnet(label)
            if label in idea.frame and label in MAIN_ATTRIBUTE and label_wordnet:
                mean_all_label = 0.0
                for word in self.frame[label]:
                    try:
                        w1 = wordnet.synset(str(word)+'.'+label_wordnet+'.01')
                        mean_label = 0.0
                        for word2 in idea.frame.get(label, []):
                            try:
                                w2 = wordnet.synset(str(word2)+'.'+label_wordnet+'.01')
                                mean_label += w1.wup_similarity(w2)
                            except:
                                error += word2 + ';'
                        mean_all_label += mean_label / float(len(idea.frame.get(label, [''])))
                        # print mean_all_label, mean_label
                    except:
                        mean_all_label = 0
                out += WEIGHT_SENTENCE.get(label, WEIGHT_SENTENCE['DEFAULT']) * mean_all_label / float(len(self.frame.get(label, [''])))
        print "Idea.compare : error:", error
        return out

    def compare_v2(self, idea):
        return norm(idea.features_vect - self.features_vect)

    """ Reduce the number word on the frame, by using synonym
    """
    def reduction_frame(self):
        for k in self.frame:
            self.frame[k] = merge_synonym(self.frame[k])
        return self


#################################################
###       For the creation and update of an Idea
""" Generate a set of ideas by splitting the text into subtext
Input:
    - text: String
    - id: Int/String; will be the base of label used for Ideas generated
Output:
    - [ Idea, ... ]
"""
def generateIdeas(text, id=0):
    # Cut sentences by meaning
    tokenized_txt = master_tokenizer.tokenize(text)

    # Label content
    ideas = []
    id_ = str(id) + '_'
    i = 0
    for token in tokenized_txt:
        ideas.append(Idea(token, id_ + str(i)).generate())
        if token != ideas[-1].text:
            print "-------------------- !!!!!!!!!!"
        i += 1

    return ideas


""" Preprocess the text and then generate frame from it
Input:
    - text: String
Output:
    - { 'VB': ['run', ...], ... }
"""
def generateIdea(text):
    words = preproc_it(text)
    tagged = nltk.pos_tag(words)
    return createFrame(tagged)


""" Aggregate common tags together on a dico
Input:
    - sentence: [ ('run','VB'), ...]
    - frame: {}
Output:
    - { 'VB': ['run', ...], ... }
"""
def createFrame(sentence):
    frame = {}
    for _value, att in sentence:
        if att not in frame:
            frame[att] = [_value.lower()]
        else:
            if _value.lower() not in frame[att]:
                frame[att].append(_value.lower())
    return copy.deepcopy(frame)


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


""" Update all objects ideas with feature selected
"""
def update_ideas(ideas, word_features):
    for idea in ideas:
        if type(idea) == list:
            for idea2 in idea:
                idea2.add_feature(word_features)
        else:
            idea.add_feature(word_features)


""" Update all objects ideas by generating feature from a vectorizer
"""
def update_ideas_v2(ideas, word_features):
    for idea in ideas:
        if type(idea) == list:
            for idea2 in idea:
                idea2.add_features_vect(word_features)
        else:
            idea.add_features_vect(word_features)


""" Return the max and mean distance between ideas
Input:
    - listIdea: [Idea]
Output:
    - {'nb': 0, 'max': 0, 'mean': 0}
"""
def analyseIdeas(listIdea):
    if not listIdea:
        return {'nb': 0, 'max': 0, 'mean': 0}
    d_max = -1
    d_mean = 0
    for idea in listIdea:
        for idea2 in listIdea:
            if idea is not idea2:
                d = idea.compare_v2(idea2)
                d_mean += d
                if d_max < d:
                    d_max = d
    d_mean /= len(listIdea)**2
    return {'nb': len(listIdea), 'max': d_max, 'mean': d_mean}



#################################################
###       Useful functions
def select_wordnet(label):
    if label in ['VBP', 'VBD', 'VBD', 'VB']:
        return 'v'
    elif label in ['NNP', 'NNS', 'NN', 'NNPS']:
        return 'n'



#################################################
###       Import/Export Ideas

def saveIdeas(filename, ideas):
    file = open(filename, 'w')
    for idea in ideas:
        s = '&&'.join([json.dumps(idea2.toSave()) for idea2 in idea])
        file.writelines(s + '\n')
    file.close()


def loadIdeas(filename):
    out = [[]]
    file = open(filename, 'r')
    for ideas in file.readlines():
        list_idea = ideas.split('&&')
        new_ideas = []
        for idea2 in list_idea:
            decjson = json.loads(idea2)
            new_ideas.append(Idea("").loadIdea(decjson))
        out.append(copy.deepcopy(new_ideas))
    file.close()
    return out



############################################################
###       Test

""" From a text, the function generate frames of common sens
Input:
    - text: String
Output:
    - [ frame={ 'DT', 'VB':, ...}, ... ]
"""
def generateIdeas_(text):
    # Cut sentences by meaning
    tokenized_txt = master_tokenizer.tokenize(text)

    # Label content
    labeled_txt = []
    frames_txt = []
    for token in tokenized_txt:
        # words = nltk.word_tokenize(token)
        words = preproc_it(token)
        tagged = nltk.pos_tag(words)
        labeled_txt.append(tagged)
        frames_txt.append(createFrame(tagged))

    return frames_txt


if __name__ == '__main__':
    # init_tokenizer()

    text = "﻿Other Georgia Tech-affiliated buildings in the area host the Center for Quality Growth and Regional Development, the Georgia Tech Enterprise Innovation Institute, the Advanced Technology Development Center, VentureLab, and the Georgia Electronics Design Center. Technology Square also hosts a variety of restaurants and businesses, including the headquarters of notable consulting companies like Accenture and also including the official Institute bookstore, a Barnes & Noble bookstore, and a Georgia Tech-themed Waffle House.[57][61]"
    text = text.decode('utf-8')
    id = Idea(text).generate()
    print id
    text2 = "﻿Other Georgia Tech-affiliated house in the area host the higher for Quality Decrease and Country Development, the Georgia Tech Enterprise Innovation Institute, the Advanced Technology Development Center, VentureLab, and the Georgia Electronics Design Center. Technology Square also hosts a variety of restaurants and businesses, including the headquarters of notable consulting companies like Accenture and also including the official Institute bookstore, a Barnes & Noble bookstore, and a Georgia Tech-themed Waffle House.[57][61]"
    text2 = text2.decode('utf-8')

    print id.compare(Idea(text2).generate())

