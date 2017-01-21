# coding=utf-8

import copy
from classifier_select import general_kw_predict
from core_idea import analyseIdeas
from variables import *
from main_knowledge import create_knowledge
from sentence import generate_sentence
SENTENCE_OPENING = 1
SENTENCE_CLOSE = 2
SENTENCE_QUESTION = 3
SENTENCE_CLAIM = 4

# Source: http://www.phrasemix.com/collections/15-ways-to-say-hello-in-english
OPEN_WORDS = ['Hello', 'Good morning', 'Morning', 'Good afternoon', 'Good evening',
              'Hey', 'What\'s up', 'Sup', 'How\'s it going', 'Howdy', 'Well hello',
              'Why hello there', 'Yo', 'Greetings', 'Look who it is', 'Look what the cat dragged in']

QUESTION_WORDS = ['What', 'When', 'Why', 'Which', 'Who', 'How', 'Whose', 'Whom']

# Source: http://www.phrasemix.com/collections/15-ways-to-say-goodbye-in-english
CLOSING_WORDS = ['Goodbye', 'Farewell', 'Have a good day', 'Take care',
                 'Bye', 'Bye bye', 'Later!', 'See you later.', 'Talk to you later',
                 'Have a good one', 'So long', 'All right then', 'Catch you later',
                 'Peace!', 'Peace out', 'I\'m out!', 'Smell you later', 'Adios',
                 'Ciao!', 'Au revoir.', 'Sayonara!']

knowledge_idea = create_knowledge()


#################################################
###       Main Functions

""" From ideas, and the mood provided by the user, build the more appropriate answer
Input:
    - ideas = [ idea ]
    - mood = UNKNWON
Output:
    - String
"""
def bot_intelligence(ideas, mood):
    # select type answer
    type = select_type_input(ideas)

    # Construct an answer
    answer = generate_answer(ideas)

    return answer


""" From the user input, generate an answer
Input:
    - user_ideas: [ideas]
Output:
    - String
"""
def generate_answer(user_ideas):
    answer = ''

    # Understand which type of user input we have.
    type_input = []
    for idea in user_ideas:
        type_input.append(select_type_input(idea.text.lower()))

    # Select the right mode to answer
    for mode in type_input:
        if mode == SENTENCE_OPENING:
            answer += OPEN_WORDS[0]

        elif mode == SENTENCE_CLOSE:
            answer += CLOSING_WORDS[0]

        elif mode == SENTENCE_QUESTION:
            answer += answer_from_knowledge(user_ideas)

        else:
            answer += answer_from_knowledge(user_ideas)

    return answer


""" Select the right behavior to have according the user input
Input:
    - user_text: String
Output:
    - SENTENCE_OPENING, SENTENCE_QUESTION, SENTENCE_CLOSE or SENTENCE_CLAIM
"""
def select_type_input(user_text):
    # Check if it's an opening sentence for a new discussion
    for key_word in OPEN_WORDS:
        if key_word.lower() in user_text:
            return SENTENCE_OPENING

    # Check if it's a question
    for key_word in QUESTION_WORDS:
        if key_word.lower() in user_text:
            return SENTENCE_QUESTION

    # Check if it's the end of the conversation
    for key_word in CLOSING_WORDS:
        if key_word.lower() in user_text:
            return SENTENCE_CLOSE

    # Check if it's a claim
    return SENTENCE_CLAIM



#################################################
###       Reasonning from user's input

""" From the knowledge created during the preprocessing stage, create an answer
Input:
    - ideas = [idea]
Output:
    - String
"""
def answer_from_knowledge(ideas):
    # Check for near idea
    new_ideas, status = get_new_ideas(ideas)

    # Generate the appropriate answer
    if status == IDEA_NEW:
        ## We found something to say
        # First try: just concat key feature
        # answer = aggregate_key_words(new_ideas)
        # print answer

        # Second try:
        answer = generate_sentence(new_ideas)

    elif status == IDEA_TOO_WIDE:
        # The user have to select a more specific subject
         answer = "Sorry could you be more precise ?"

    elif status == IDEA_TOO_FAR:
        # Ask to focus on the right topic
         answer = "Sorry could we come back to a Georgia tech related topic ?"

    else:
        # We don't understand the request or the saying of the person
        answer = "Sorry I don't understand what you said."

    return answer


""" Try to find related ideas in the aim of answering to one question
Input:
    - ideas: [ideas]
Output:
    - [ ideas ]
    - status = IDEA_NEW or IDEA_TOO_WIDE or IDEA_TOO_FAR
"""
def get_new_ideas(knowledge_user):
    idea_new = []
    status = IDEA_NEW

    # Get all potentials concepts:
    concepts = get_concepts(knowledge_user)

    # From concept found, get potential useful ideas
    idea_new = get_ideas(concepts, knowledge_user)

    # Check the wide & the number of idea found
    stats = analyseIdeas(idea_new)
    print 'stats', stats
    if stats['max'] > THRESHOLD_DISP:
        status = IDEA_TOO_FAR
    elif stats['nb'] > THRESHOLD_NB:
        status = IDEA_TOO_WIDE
    elif stats['nb'] == 0:
        status = IDEA_NO
    else:
        status = IDEA_NEW

    return idea_new, status


""" From the input provided by the user, select first big concept which can fit
Input:
    - knowledge_user = [idea]
Output:
    - [concept]
"""
def get_concepts(knowledge_user):
    concept_new = []

    # Get Label of potential useful concept
    concept_label = []
    for idea in knowledge_user:
        concept_label.append(str(general_kw_predict(idea.features_vect)))
    print "concept_label", concept_label
    # concept_label = [str(k) for k in concept_label]

    # Get Concepts from label found
    for k_sub in knowledge_idea:
            if k_sub.label in concept_label:
                print k_sub
                concept_new.append(k_sub)
    print 'concept_new', len(concept_new)
    return concept_new


""" As a second step, from Concepts found, try to focus on ideas
Input:
    - knowledge_user = [idea]
Output:
    - [concept]
"""
def get_ideas(concepts, knowlegde_user):
    idea_new_all = []
    for concept in concepts:
        idea_label = []
        idea_new = []

        # For each concept found try to predict which idea will be useful
        for idea in knowlegde_user:
            idea_label.append(concept.predict_idea(idea))

        print "idea_label", len(idea_label), idea_label

        # Get the ideas from labels found
        for i in range(0, len(concept.ideas)):
            if concept.idea_model_label[i] in idea_label:
                idea_new.append(concept.ideas[i])
        print 'idea_new', len(idea_new)

        idea_new_all += copy.deepcopy(idea_new)

    print 'idea_new_all', len(idea_new_all)
    return idea_new_all


# 2nd Approach: Get Ideas from the pool of one concept
def find_idea(concepts, knowlegde_user):
    idea_new_all = []
    for concept in concepts:
        idea_label = []
        idea_new = []

        # For each concept found try to predict which idea will be useful
        for idea in knowlegde_user:
            idea_label.append(concept.predict_knn(idea.features_vect))
        print "idea_label", len(idea_label), idea_label

        # Get the ideas from labels found
        for i in range(0, len(idea_label)):
            for j in range(0, len(idea_label[i])):
                if concept.ideas[j] not in idea_new:
                    idea_new.append(concept.ideas[j])
        print 'idea_new', len(idea_new)

        idea_new_all += copy.deepcopy(idea_new)

    print 'idea_new_all', len(idea_new_all)
    return idea_new_all

# 3d Approach: get Idea from the overall pool
def find_idea_all(knowledge_user):
    print ''



############################################################
###       Test


