from elasticsearch import Elasticsearch
from datetime import datetime
import numpy as np
from functions import *

def getUnigramFeatures(es, labels, doc_map, unigrams, is_train_set = 1):
    print 'load features'
    set_type = 'train' if is_train_set else 'test'
    f = open('../results/unigrams/features_' + set_type + '.txt', 'w')
    cnt = 0
    for i, word in enumerate(unigrams):
        scroll = getScrollInfoFeatures(es, word, is_train_set)
        scroll_id = scroll['_scroll_id']
        while True:
            if len(scroll['hits']['hits']) == 0:
                break
            for doc in scroll['hits']['hits']:
                doc_id = doc['fields']['docno'][0]
                score = doc['_score']

                values = (doc_map[doc_id], i, score)
                f.write(' '.join(map(lambda v: str(v), values)) + '\n')
                cnt += 1
            scroll = es.scroll(scroll_id = scroll_id, scroll= "1m")
            scroll_id = scroll['_scroll_id']
        if (i + 1) % 5000 == 0:
            print 'process', (i + 1), \
                    'unigrams, number of sample features is', cnt
    f.close()

def loadVocabulary():
    with open('../results/vocabulary.txt', 'r') as f:
        unigrams = []
        cnt = 0
        for line in f.readlines():
            word_stemmed, word = line.rstrip('\n').split(' ')
            unigrams.append(word)
            cnt += 1
        print 'size of vocabulary is', cnt
    return unigrams

if __name__ == '__main__':
    now = datetime.now()

    es = Elasticsearch()
    unigrams = loadVocabulary()
    unigrams_flag = 'unigrams'

    print 'load train set'
    is_train_set = 1
    labels, doc_map = getLabels(es, is_train_set)
    dumpLabels(unigrams_flag, labels, is_train_set)
    getUnigramFeatures(es, labels, doc_map, unigrams, is_train_set)

    print 'load test set'
    is_train_set = 0
    labels, doc_map = getLabels(es, is_train_set)
    dumpLabels(unigrams_flag, labels, is_train_set)
    getUnigramFeatures(es, labels, doc_map, unigrams, is_train_set)

    print 'running time is', datetime.now() - now
