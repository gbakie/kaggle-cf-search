"""
CrowdFlower Search Relevance Challenge (Kaggle)
cf_io.py: io functions
__author__: gbakie
"""

import csv

# CSV train format: id, query, product_title, product_description, median_relevance, relevance_variance
# CSV test format: id, query, product_title, product_description

def load_training():
    training = []
    for line in csv.DictReader(open("../input/train_new.csv")):
        for name, conv in {"id":int, "median_relevance":int,
                            "relevance_variance":float}.items():
            line[name] = conv(line[name])

        training.append(line)
    return training

def load_testing():
    testing = []
    for line in csv.DictReader(open("../input/test_new.csv")):
        for name, conv in {"id":int}.items():
            line[name] = conv(line[name])

        testing.append(line)

    return testing

def write_output(test, rates):
    out = csv.writer(open("search_eval.csv", "w"))
    out.writerow(["id","prediction"])

    for line,rate in zip(test,rates):
        out.writerow([line["id"], rate])

