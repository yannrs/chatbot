
from variables import BOT_ID, SLACK_BOT_TOKEN


import time
from slackclient import SlackClient
from core_idea import *
from variables import *
from classifier_select import load_knowledge, VECTORIZER
from intelligence import bot_intelligence, get_new_ideas

# constants
AT_BOT = "<@" + BOT_ID + ">"
AT_CHANNEL_BOT = "D2Y397HBK"


# instantiate Slack & Twilio clients
slack_client = SlackClient(SLACK_BOT_TOKEN)

#################################################
###       Main functions


def main_slack():
    initialization_step()
    if slack_client.rtm_connect():
        print("StarterBot connected and running!")
        track_idea = {}
        while True:
            # try:
            one_iteration(track_idea)
            time.sleep(READ_WEBSOCKET_DELAY)
            # except Exception as e:
            #     print e
    else:
        print("Connection failed. Invalid Slack token or bot ID?")


def initialization_step():
    print ('='*80)
    print ('>'*80)
    print ('='*80)
    print 'Initialization'
    t0 = time.time()
    load_knowledge()
    print("initialization_step done in %0.3fs" % (time.time() - t0))
    print '\nReady to begin'


def one_iteration(track_idea):
        bot_input = slack_client.rtm_read()
        for output in bot_input:
            print output
            if output and 'text' in output and \
                    (AT_BOT in output['text'] or \
                             (AT_CHANNEL_BOT in output.get('channel', '') and \
                              BOT_ID not in output.get('user', ''))):

                ideas, mood = parse_bot_input(output['text'])

                # Save the evolution of ideas
                if output['user'] in track_idea:
                    track_idea[output['user']].append((ideas, mood))
                else:
                    track_idea[output['user']] = [(ideas, mood)]

                answer = generate_bot_answer(ideas, mood)

                send_answer(answer, output['channel'])



#################################################
###       Parse Input Information

""" From the text sent by a Human, this function converts sentences to ideas, and add the mood as an extra feature
Input:
    - bot_input: String
Output:
    - [ideas], mood
"""
def parse_bot_input(bot_input):
    ideas = []
    mood = -1

    ideas = generateIdeas(bot_input)
    ideas = [idea.add_features_vect(VECTORIZER) for idea in ideas]

    return ideas, mood



#################################################
###       Generate an answer

""" From ideas and the mood of the user, generate an answer or a new say
Input:
    - ideas: [ideas]
    - mood: mood
Output:
    - String
"""
def generate_bot_answer(ideas, mood):
    # Generate the appropriate answer
    answer = bot_intelligence(ideas, mood)

    return answer



#################################################
###       Send the answer

def send_answer(response, channel):
    slack_client.api_call("chat.postMessage", channel=channel,
                          text=response, as_user=True)




############################################################
###       Tests

def test_main(txt):
    ideas, mood = parse_bot_input(txt)
    print "parse_bot_input", ideas, mood

    ## Check for near idea
    new_ideas, status = get_new_ideas(ideas)
    print 'get_new_ideas', len(new_ideas), status

    # Generate the appropriate answer
    answer = ""
    if status == IDEA_NEW:
        # We found something to say
         answer = generate_bot_answer(new_ideas, mood)

    elif status == IDEA_TOO_WIDE:
        # The user have to select a more specific subject
         answer = "Sorry could you be more precise ?"

    elif status == IDEA_TOO_FAR:
        # Ask to focus on the right topic
         answer = "Sorry could we come back to a Georgia tech related topic ?"

    else:
        # We don't understand the request or the saying of the person
        answer = "Sorry I don't understand what you said."

    print "answer: ", answer
    return answer


if __name__ == '__main__':
    initialization_step()
    test_main('georgia tech is the best')