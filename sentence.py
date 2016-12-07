# coding=utf-8
from core_idea import Idea

"""
"""
def aggregate_key_words(ideas):
    answer = ''
    for idea in ideas:
        answer += ';'.join(list(idea.vectorizer.inverse_transform(idea.features_vect)[0]))
    return answer


def aggregate_text(ideas):
    answer = ''
    for idea in ideas:
        answer += ' ' + idea.text
    return answer


def generate_sentence(ideas):
    sentences = ''

    # Merge text ideas
    sentences = aggregate_text(ideas)

    # Merge Frame ideas
    idea_all = merge_idea(ideas)

    # Regenerate sentences from idea
    sentences2 = idea_to_sentence(idea_all)

    return sentences


def merge_idea(ideas):
    s = ''
    for idea in ideas:
        s += idea.text
    idea = Idea(s).generate().add_features_vect(ideas[0].vectorizer)

    k = 0
    for key in idea.frame:
        k += len(idea.frame[key])
    idea.reduction_frame()
    k2 = 0
    for key in idea.frame:
        k2 += len(idea.frame[key])

    print k, k2


    return idea


def idea_to_sentence(idea):
    sentence = ''



    return sentence



############################################################
###       Test


if __name__ == '__main__':
    print 'Try'
    # generate_sentence(idea)