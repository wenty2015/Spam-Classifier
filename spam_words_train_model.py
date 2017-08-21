from elasticsearch import Elasticsearch
from sklearn import ensemble, tree, linear_model, naive_bayes
import numpy as np
from scipy.sparse import csr_matrix
from functions import *

def loadSpamWords(flag = 'website'):
    with open('../data/spam_words_' + flag + '.txt', 'r') as f:
        words = filter(lambda w: len(w.strip(' ')) > 0,
                        f.read().replace('\n', ' ').replace('$', '')\
                        .replace('%', ' percent ').split(' '))
        words = sorted(list(set(words)))
    with open('../results/' + flag + '/spam_words.txt', 'w') as f:
        for word in words:
            f.write(word + '\n')
    return words

def getFeatures(es, doc_map, spam_words, is_train_set):
    print 'load features'
    row, col, val = [], [], []
    for i, w in enumerate(spam_words):
        scroll = getScrollInfoFeatures(es, w, is_train_set)
        scroll_id = scroll['_scroll_id']
        while True:
            if len(scroll['hits']['hits']) == 0:
                break
            for doc in scroll['hits']['hits']:
                doc_id = doc['fields']['docno'][0]
                score = doc['_score']
                row.append(doc_map[doc_id])
                col.append(i)
                val.append(score)
            scroll = es.scroll(scroll_id = scroll_id, scroll= "1m")
            scroll_id = scroll['_scroll_id']
    features = csr_matrix((val, (row, col)),
                            shape=(len(doc_map.keys()), len(spam_words))).toarray()
    return features

def outputDataset(spam_words_flag, dataset, is_train_set):
    set_type = 'train' if is_train_set else 'test'
    print 'output', set_type, 'set'
    file_name = ''.join(['../results/', spam_words_flag, '/samples_',
                        set_type, '.txt'])
    labels, features = dataset['labels'], dataset['features']
    with open(file_name, 'w') as f:
        for label, feature in zip(labels, features):
            feature_text = ','.join(map(lambda f: str(f), feature))
            doc, lab = label
            f.write(' '.join([doc, str(lab), feature_text]) + '\n')

def loadDataset(es, spam_words_flag, spam_words, is_train_set):
    labels, doc_map = getLabels(es, is_train_set)
    dumpLabels(spam_words_flag, labels, is_train_set)

    features = getFeatures(es, doc_map, spam_words, is_train_set)
    dataset = {'labels': labels, 'features': features}
    return dataset

if __name__ == '__main__':
    es = Elasticsearch()

    spam_words_flag = 'website'

    spam_words = loadSpamWords(spam_words_flag)

    print 'load train set'
    is_train_set = 1
    train_set = loadDataset(es, spam_words_flag, spam_words, is_train_set)
    outputDataset(spam_words_flag, train_set, is_train_set)

    print 'load test set'
    is_train_set = 0
    test_set = loadDataset(es, spam_words_flag, spam_words, is_train_set)
    outputDataset(spam_words_flag, test_set, is_train_set)

    # decision tree
    model = tree.DecisionTreeClassifier(random_state = 64)
    model_name = 'decision tree'
    trainModel(spam_words_flag, train_set, test_set, model, model_name)

    # random forest
    model = ensemble.RandomForestClassifier(n_estimators = 100, random_state = 64)
    model_name = 'random forest'
    trainModel(spam_words_flag, train_set, test_set, model, model_name)

    # logistic regress
    model_name = 'logistic regression'
    model = linear_model.LogisticRegression()
    C = crossValidation(train_set, model, model_name)
    model.C = C
    trainModel(spam_words_flag, train_set, test_set, model, model_name)

    # naive bayes
    model = naive_bayes.GaussianNB()
    model_name = 'naive bayes'
    trainModel(spam_words_flag, train_set, test_set, model, model_name)
