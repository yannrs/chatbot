# coding=utf-8
import nltk
import copy
from nltk.corpus import wordnet
from nltk.corpus import state_union
from nltk.corpus import stopwords
from nltk.tokenize import PunktSentenceTokenizer

from variables import WEIGHT_SENTENCE, MAIN_ATTRIBUTE
from Knowledges.preprocessing import *
import json

class Idea:
    def __init__(self, text, id=-1):
        self.id = id
        self.text = text            # ''
        self.frame = {}             # {'NP':[ 'Charles', ....], 'VB': [...] }
        self.features = {}          # {'word1': 1, 'word2':0, 'word3': 10, .... }

    def __str__(self):
        return 'Idea: ' + str(self.frame)

    def __repr__(self):
        return 'Idea: ' + str(self.id) + ' [' + str(self.frame) + ']'

    """ From the text, generate a fram which will represent the idea
    """
    def generate(self):
        # print begin
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


    def toSave(self):
        content = {}
        content['id'] = self.id
        content['text'] = self.text
        content['frame'] = json.dumps(self.frame)
        content['features'] = json.dumps(self.features)
        return content

    def loadIdea(self, dico):
        self.id = dico['id']
        self.text = dico['text']
        self.frame = json.loads(dico['frame'])
        self.features = json.loads(dico['features'])
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
        print "error", error
        return out




def select_wordnet(label):
    if label in ['VBP', 'VBD', 'VBD', 'VB']:
        return 'v'
    elif label in ['NNP', 'NNS', 'NN', 'NNPS']:
        return 'n'



master_tokenizer = PunktSentenceTokenizer(state_union.raw("2005-GWBush.txt"))

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

""" Generate a set of ideas by splitting the text into subtext

"""
def generateIdeas(text, id=0):
    # Cut sentences by meaning
    tokenized_txt = master_tokenizer.tokenize(text)

    # Label content
    ideas = []
    id = id
    for token in tokenized_txt:
        ideas.append(Idea(token, id).generate())

    return ideas


def generateIdea(text):
    words = preproc_it(text)
    tagged = nltk.pos_tag(words)
    return createFrame(tagged)


attribute = []
""" Functionca
"""
def createFrame(sentence, frame={}):
    for _value, att in sentence:
        if att not in attribute:
            attribute.append(att)
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



""" Return the max and mean distance between idea
"""
def analyseIdeas(listIdea):
    if not listIdea:
        return {'nb': 0, 'max': 0, 'mean': 0}
    d_max = -1
    d_mean = 0
    for idea in listIdea:
        for idea2 in listIdea:
            if idea is not idea2:
                d = idea.compare(idea2)
                d_mean += d
                if d_max < d:
                    d_max = d
    d_mean /= 2*len(listIdea)
    return {'nb': len(listIdea), 'max': d_max, 'mean': d_mean}


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

if __name__ == '__main__':
    # init_tokenizer()

    text = "﻿Other Georgia Tech-affiliated buildings in the area host the Center for Quality Growth and Regional Development, the Georgia Tech Enterprise Innovation Institute, the Advanced Technology Development Center, VentureLab, and the Georgia Electronics Design Center. Technology Square also hosts a variety of restaurants and businesses, including the headquarters of notable consulting companies like Accenture and also including the official Institute bookstore, a Barnes & Noble bookstore, and a Georgia Tech-themed Waffle House.[57][61]"
    text = text.decode('utf-8')
    id = Idea(text).generate()
    print id
    text2 = "﻿Other Georgia Tech-affiliated house in the area host the higher for Quality Decrease and Country Development, the Georgia Tech Enterprise Innovation Institute, the Advanced Technology Development Center, VentureLab, and the Georgia Electronics Design Center. Technology Square also hosts a variety of restaurants and businesses, including the headquarters of notable consulting companies like Accenture and also including the official Institute bookstore, a Barnes & Noble bookstore, and a Georgia Tech-themed Waffle House.[57][61]"
    text2 = text2.decode('utf-8')

    print id.compare(Idea(text2).generate())

