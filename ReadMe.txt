################ PROJECT 3 - Alternative ################
# From Class: CS 7637 - Knowledge-Based AI, Fall 2016
# Author: Yann RAVEL-SIBILLOT
# Date started: 11/05/2016		(5 November 2016)

# Final Report: 
https://mega.nz/#!E0VQiJpZ!n0ZlTV9GdxPkvTgz_Ya6QbQ_ERhTzqgyhRGZxRB2bug


# Required:
python 2.7
scikit-learn >0.18
nltk >3
slackclient

# TODO before running the project:
Set the PATH on variables.py where are located data


# Architecture of file!
# Launchable:
main_knowledge.py : while create the knowledge needed to use the bot
	if MODE = 1 on the file, then create all the knowledge from scratch
	if MODE = 2	on the file, then load ideas and concept, and recreate models
main.py: 			while launch every thing, the bot and the creation of knowledge, if the MODE = 1 on the file "main_knowledge.py"


Models are saved on the folder Data\Models
Concepts are saved on the folder Models



# Other files
slack.py: 			functions to launch, communicate with Slack, the main loop is defined here

core_concept.py:	definiton of the class Concept
core_idea.py:		definiton of the class Idea

intelligence.py: 	functions to reason from the user's input
sentence.py:		functions to create sentence from ideas
classifier_select.py: functions to load and save knowledge + predict and train classifiers

util.py: 			functions to plot and save statistical data
variables.py: 		most of global variables used


# On the folder Knowledges:
Knowledges\preprocessing.py: 	functions to filter, merge, read data from file text
Knowledges\formatData.py: 		functions to prefilter the data


# On the folder Data:
gatech_wiki_clean_v3.csv: 		contains all the knowledge from Wikipedia
Courses\*: 						contains all transcripts from the KBAI class

