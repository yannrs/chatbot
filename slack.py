
from variables import BOT_ID, SLACK_BOT_TOKEN


import os
import time
from slackclient import SlackClient
from core_idea import Idea

# constants
AT_BOT = "<@" + BOT_ID + ">"
AT_CHANNEL_BOT = "D2Y397HBK"

IDEA_NEW = 0
IDEA_TOO_WIDE = 1
IDEA_TOO_FAR = 2


# instantiate Slack & Twilio clients
slack_client = SlackClient(SLACK_BOT_TOKEN)


""" From the text sent by a Human, this function converts sentences to ideas, and add the mood as an extra feature
Input:
    - bot_input: String
Output:
    - [ideas], mood
"""
def parse_bot_input(bot_input):
    ideas = []
    mood = -1

    for part in bot_input:
        ideas.apppend(Idea(part).generate())

    return ideas, mood


""" Try to find near ideas or answer to one question
Input:
    - ideas: [ideas]
Output:
    - [ ideas ]
    - status = IDEA_NEW or IDEA_TOO_WIDE or IDEA_TOO_FAR
"""
def get_new_ideas(ideas):
    idea_new = []

    return idea_new, 2


""" From ideas and the mood of the user, generate an answer or a new say
Input:
    - ideas: [ideas]
    - mood: mood
Output:
    - String
"""
def generate_bot_answer(ideas, mood):
    return "-1"


def send_answer(response, channel):
    slack_client.api_call("chat.postMessage", channel=channel,
                          text=response, as_user=True)


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

                ## Check for near idea
                new_ideas, status = get_new_ideas(ideas)

                # Generate the appropriate answer
                answer = ""
                if status == IDEA_NEW:
                    # We found something to say
                     answer = generate_bot_answer(ideas, mood)

                elif status == IDEA_TOO_WIDE:
                    # The user have to select a more specific subject
                     answer = "Sorry could you be more precise ?"

                elif status == IDEA_TOO_FAR:
                    # Ask to focus on the right topic
                     answer = "Sorry could we come back to a Georgia tech related topic ?"

                send_answer(answer, output['channel'])


def main_slack():
    READ_WEBSOCKET_DELAY = 1    # 1 second delay between reading from firehose
    if slack_client.rtm_connect():
        print("StarterBot connected and running!")
        track_idea = {}
        while True:
            one_iteration(track_idea)
            time.sleep(READ_WEBSOCKET_DELAY)
    else:
        print("Connection failed. Invalid Slack token or bot ID?")

