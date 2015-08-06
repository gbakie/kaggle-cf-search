"""
CrowdFlower Search Relevance Challenge (Kaggle)
preproc_desc.py: preprocess product description and create new train and test dataset with the result 
__author__: gbakie
"""

import csv
import re
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag 

POS_TOKENS = ['NNP', 'NN', 'NNS', 'CD', 'JJ']

def extract_pos_tokens(text):
    tok = RegexpTokenizer(r'\w+')
    pos_tokens = []

    if '<' in text or '{' in text:
        return ""

    text = re.sub("[^a-zA-Z0-9]"," ", text)
    tokens = tok.tokenize(text)
    tags = pos_tag(tokens)
    for (token, tag) in tags:
        if tag in POS_TOKENS:
            pos_tokens.append(token)

    return " ".join(pos_tokens)

def pre_training():
    fd = ['id', 'query', 'product_title', 'product_description', 'median_relevance', 'relevance_variance']
    writer = csv.DictWriter(open("../input/train_new.csv", "w"), fieldnames=fd)

    writer.writeheader()
    for line in csv.DictReader(open("../input/train.csv")):
        for name, conv in {"product_description":extract_pos_tokens}.items():
            line[name] = conv(line[name])

        writer.writerow(line)

def pre_testing():
    fd = ['id', 'query', 'product_title', 'product_description']
    writer = csv.DictWriter(open("../input/test_new.csv", "w"), fieldnames=fd)

    writer.writeheader()
    for line in csv.DictReader(open("../input/test.csv")):
        for name, conv in {"product_description":extract_pos_tokens}.items():
            line[name] = conv(line[name])
        writer.writerow(line)

pre_training()
pre_testing()
