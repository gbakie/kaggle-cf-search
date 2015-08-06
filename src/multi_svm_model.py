"""
CrowdFlower Search Relevance Challenge (Kaggle)
multi_svm_model.py: build one SVM model for each query (train and test queries are the same). 
__author__: gbakie
"""

import re
import numpy as np
from collections import defaultdict
from operator import itemgetter
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import *
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from cf_io import load_training, load_testing, write_output


# Model parameters and constants
SEED = 7
TITLE_WEIGHT = 2
SVM_C = 2.
CUSTOM_STOP_WORDS = ['http','www','img','border','color','style','padding','table','font','inch','width','height','td', 'tr', 'nbsp', 'strong', 'li', 'ul']

stop_words = set(stopwords.words('english'))
stop_words.update(CUSTOM_STOP_WORDS)

stemmer = PorterStemmer()

def group_products_by_queries(data, test=False):
    dict_queries = defaultdict(list)

    for line in data:
        id = line["id"]
        query = line["query"].lower()
        title = line["product_title"].lower()
        title = re.sub("[^a-zA-Z0-9]"," ", title)
        desc = line["product_description"].lower()
        if '<' in desc or '{' in desc:
            desc = ""
        else:
            desc = re.sub("[^a-zA-Z0-9]"," ", desc)
        prod_text = " ".join((title*TITLE_WEIGHT, desc))

        prod = [id,prod_text,preprocess(title)]
        #prod = [id,prod_text,title]
        if test == False:
            rate = line["median_relevance"]
            var = line["relevance_variance"]
            if var == 0:
                var = 4.
            else:
                var = 1./var 
        
            prod.extend([rate,var])
            
        dict_queries[query].append(prod)

    return dict_queries

# used for DEBUG
def dump_queries(train, test):
    for query,prods_train in train.items():
        print "Query: %s" % query
        print "-------TRAIN--------"
        for prod in prods_train:
            print prod[2]

        print "\n\n"
        print "-------TEST---------"
        prods_test = test[query]
        for prod in prods_test:
            print prod[2]

        print "\n\n"

def get_product_vocab(dict_queries):
    tok = RegexpTokenizer(r'\w+')
    vocab = {}

    for query,v in dict_queries.items():
        words = defaultdict(int)

        for prod in v:
            w_prod = tok.tokenize(prod[1])
            for w in w_prod:
                #wt = stem(wt)
                if not re.match(r'\d+$', w) and \
                    len(w) > 1 and \
                    w not in stop_words: 
                    words[w] += 1

        vocab[query] = words.keys()
        #vocab[query] = [k for (k, v) in words.iteritems() if v > 1]

        """
        print "Query: " + query
        sorted_w = sorted(words.items(), key=lambda x:x[1], reverse=True)
        print sorted_w
        """
    
    return vocab

# tokenize sentence and apply stemmer to each token
def preprocess(sent):
    tok = RegexpTokenizer(r'\w+')
    
    words = tok.tokenize(sent)
    new_words = []
    for w in words:
        new_words.append(stemmer.stem(w))

    return " ".join(new_words)


def build_features(dict_queries, vocab, test=False):
    features = {}

    for query,v in dict_queries.items():
        q_vocab = vocab[query]
        prod_texts = []
        Y = []
        weights = []
        ids = []
        titles = []

        if test == False:
            for prod in v:
                prod_texts.append(prod[1])
                titles.append(prod[2])
                Y.append(prod[3])
                weights.append(prod[4])
        else:
            for prod in v:
                ids.append(prod[0])
                prod_texts.append(prod[1])
                titles.append(prod[2])
            
    
        vec_query = TfidfVectorizer(max_df=0.8,
                                 min_df=1, stop_words=None,
                                 use_idf=False, smooth_idf=False,
                                 strip_accents='unicode',
                                 analyzer='word',
                                 vocabulary=q_vocab,
                                 #sublinear_tf=True,
                                 )
        X = vec_query.fit_transform(prod_texts).todense()

        cos_sim = cosine_sim(preprocess(query), titles)
        X = np.hstack((X,cos_sim))

        if test == False:
            Y = np.asarray(Y)
            features[query] = (X, Y, weights)
        else:
            ids = np.asarray(ids)
            features[query] = (X, ids)

    return features


# calculate cosine similarity between query and title
def cosine_sim(query, titles):
    vec = TfidfVectorizer(max_df=1.,
                         min_df=1, stop_words='english',
                         use_idf=False, smooth_idf=False,
                         binary=True,
                         strip_accents='unicode',
                         analyzer='word',
                         )

    full_data = titles + list(query)

    vec.fit(full_data)
    X_query = vec.transform([query])
    X_title = vec.transform(titles)

    scores = cosine_similarity(X_query, X_title)

    return scores.T


# train a SVM model for each one of the training queries
# use models to predict relevance on test dataset
def train_and_classify(train, test):
    prods_train = group_products_by_queries(train)
    vocab = get_product_vocab(prods_train)

    f = build_features(prods_train, vocab)
    clfs = {}
    for query,products in f.items():
        (X, Y, weights) = products
        print "query: %s. number of products: %d" % (query, X.shape[0])
        clf = svm.SVC(C=SVM_C, random_state=SEED, kernel='linear')
        clf.fit(X,Y)
        #clf.fit(X,Y,weights)

        clfs[query] = clf

    prods_test = group_products_by_queries(test,True)
    f_t = build_features(prods_test,vocab,True)

    res = {}
    for query,vs in f_t.items():
        clf = clfs[query]

        (v,id) = vs
        pred = clf.predict(v)
        for i,p in zip(id,pred):
            res[i] = p

    pred = []
    for k in sorted(res.iterkeys()):
        pred.append(res[k])

    return pred

# used for DEBUG
def analyse(train, test):
    prods_train = group_products_by_queries(train)
    prods_test = group_products_by_queries(test,True)

    dump_queries(prods_train, prods_test)

def main():
    train = load_training()
    test = load_testing()

    #analyse(train, test)
    pred = train_and_classify(train,test)
    write_output(test,pred)

if __name__ == '__main__':
    main()

