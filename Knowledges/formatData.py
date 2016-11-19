

import os, sys


path = 'C:\Users\yann\Videos\Mooc\Knowledge-Based AI_ Cognitive Systems Videos\Knowledge-Based AI_ Cognitive Systems Subtitles\\'


## Convert srt file to text file:

def convert_srt2text(filename):
    file = open(filename, "r")
    text = ""
    i = 0
    tempText = []
    for line in file.readlines():
        tempText.append(line)
        i += 1
        if tempText[-1] == "\n":
            text += ' ' + ' '.join(tempText[2:-1])
            tempText = []

    file.close()
    return text.replace('\n', '')


def saveList(list, filename):
    file = open(filename, "w")
    for line in list:
        file.writelines(str(line) + '\n')
    file.close()


def aggregateAllTopic(folderName):
    listFile = os.listdir(path+folderName)
    print listFile
    text = []
    i = 0
    for file in listFile:
        text.append(convert_srt2text(path + '\\' + folderName + "\\" + file))

    saveList(text, path + folderName + '.csv')


def applyAllFolder(mainPath):
    listFolder = os.listdir(mainPath)
    print listFolder
    for folder in listFolder:
        aggregateAllTopic(folder)


if __name__ == '__main__':
    applyAllFolder(path)