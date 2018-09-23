#!/usr/bin/env python
from __future__ import print_function

import re
import string
import argparse
import json

__author__ = "Agustin Marinovic Sfeir"  # and Michael Kliger, Robin Zhang
__email__ = "amarinovic@ucla.edu"   

# Some useful data.
_CONTRACTIONS = {
    "tis": "'tis",
    "aint": "ain't",
    "amnt": "amn't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hell": "he'll",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "id": "i'd",
    "ill": "i'll",
    "im": "i'm",
    "ive": "i've",
    "isnt": "isn't",
    "itd": "it'd",
    "itll": "it'll",
    "its": "it's",
    "mightnt": "mightn't",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "oclock": "o'clock",
    "ol": "'ol",
    "oughtnt": "oughtn't",
    "shant": "shan't",
    "shed": "she'd",
    "shell": "she'll",
    "shes": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "somebodys": "somebody's",
    "someones": "someone's",
    "somethings": "something's",
    "thatll": "that'll",
    "thats": "that's",
    "thatd": "that'd",
    "thered": "there'd",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "wed": "we'd",
    "wedve": "wed've",
    "well": "we'll",
    "were": "we're",
    "weve": "we've",
    "werent": "weren't",
    "whatd": "what'd",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whodve": "whod've",
    "wholl": "who'll",
    "whore": "who're",
    "whos": "who's",
    "whove": "who've",
    "whyd": "why'd",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "yall": "y'all",
    "youd": "you'd",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've"
}


def sanitize(text):
    punctuation = r"[*\(\){}\[\]<>/\+=_~`'\"@#\$%^&]"   # All special characters to remove

    s2 = ' '+text+' '
    # Replace \t and \n with ' '
    s2 = re.sub(r'\s', ' ', s2)

    # Delete url
    # Embedded URL [text](link)
    s2 = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\g<1>", s2)

    s2 = re.sub(r'http[^ ]+\.[^ ]+ ', ' ', s2)

    # Replace punctuation with " {} " where {} = chunk of .,?!;:
    s2 = re.sub(r' [\.,\?!:;]+', r' \g<0> ', s2)
    s2 = re.sub(r'[\.,\?!:;]+ ', r' \g<0> ', s2)

    # Separate begining and ending punctuation  # TODO remove (?)
    s2 = re.sub(r'^[\.,\?!:;]+',r' \g<0> ', s2)
    s2 = re.sub(r'[\.,\?!:;]+$',r' \g<0> ', s2)

    s2 = re.sub(r'^{}+'.format(punctuation), r' \g<0> ', s2)
    s2 = re.sub(r'{}+$'.format(punctuation), r' \g<0> ', s2)

    # Remove and split punctuation
    s_temp = "[]()"
    rec1 = re.compile(r' ([\.,\?!:;]+)([\.,\?!:;]+)')
    rec2 = re.compile(r'([\.,\?!:;]+)([\.,\?!:;]+) ')

    spec1 = re.compile(r" {}+".format(punctuation))
    spec2 = re.compile(r"{}+ ".format(punctuation))

    while(s_temp != s2):    # I wish I could not use a loop, but thus is life
        s_temp = s2
        # De-cluster valid punctuation
        s2 = rec1.sub(r' \g<1> \g<2> ', s2)
        s2 = rec2.sub(r' \g<1> \g<2> ', s2)
        # Remove invalid punctuation
        s2 = spec1.sub(' ', s2)
        s2 = spec2.sub(' ', s2)
        # Redo edge removal
        s2 = re.sub(r'( [\.,\?!:;]+)', r'\g<0> ', s2)
        s2 = re.sub(r'([\.,\?!:;]+ )', r' \g<0>', s2)
        # Remove multiple white space
        s2 = re.sub(r' +', ' ', s2)

    # Replace multiple space with single and lower everything
    clean = re.sub(r' +', ' ', s2).lower().strip()

    return [clean, unigram(clean), bigram(clean), trigram(clean)]


def unigram(s):
    str = ""
    a = s.split(" ")
    for i in range(len(a)):
        if not a[i] in ",.;:!?":
            str += a[i]
            str += " "

    return str.strip()


def bigram(s):
    str = ""
    a = s.split(" ")
    for i in range(len(a) - 1):
        if not (a[i] in ",.;:!?" or a[i+1] in ",.;:!?"):
            str += "{}_{} ".format(a[i], a[i + 1])

    return str.strip()


def trigram(s):
    str = ""
    a = s.split(" ")
    for i in range(len(a) - 2):
        if not (a[i] in ",.;:!?" or a[i+1] in ",.;:!?" or a[i+2] in ",.;:!?"):
            str += "{}_{}_{} ".format(a[i], a[i+1], a[i+2])

    return str.strip()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()

    for line in open(args.filename):
        body = json.loads(line)['body']
        print(sanitize(body))   # Nasty


