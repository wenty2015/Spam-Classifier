from sklearn import ensemble
from datetime import datetime
import numpy as np
from scipy.sparse import csr_matrix
from functions import *

def loadFeatures(labels, unigrams, is_train_set):
    row, col, val = [], [], []
    set_type = 'train' if is_train_set else 'test'
    print 'load features for', set_type

    with open('../results/unigrams/features_' + set_type + '.txt', 'r') as f:
        for line in f.readlines():
            r, c, v = line.rstrip('\n').split(' ')
            row.append(int(r))
            col.append(int(c))
            val.append(float(v))
    features = csr_matrix((val, (row, col)), shape=(len(labels), len(unigrams)))
    return features

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

def loadLabels(is_train_set):
    set_type = 'train' if is_train_set else 'test'
    foler_name = '../results/unigrams/'
    file_name = 'labels_' + set_type + '.txt'
    labels = []
    with open(foler_name + file_name, 'r') as f:
        for line in f.readlines():
            doc_id, label = line.rstrip('\n').split(' ')
            labels.append((doc_id, int(label)))
    spam_size = np.sum(map(lambda l: l[1], labels))
    print 'spam sample size = %d, ham sample size = %d' % \
                (spam_size, len(labels) - spam_size)
    return labels

def getTopFeatures(model, topK, unigrams):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:topK]
    words = [unigrams[i] for i in indices]
    print 'top %d features:' % topK
    print ','.join(words)
    with open('../results/unigrams/top_features.txt', 'w') as f:
        for i in indices:
            f.write(unigrams[i] + ' ' + str(importance[i]) + '\n')

if __name__ == '__main__':
    now = datetime.now()

    unigrams = loadVocabulary()
    unigrams_flag = 'unigrams'

    print 'load train set'
    is_train_set = 1
    labels = loadLabels(is_train_set)
    features = loadFeatures(labels, unigrams, is_train_set)
    train_set = {'labels': labels, 'features': features}

    print 'load test set'
    is_train_set = 0
    labels = loadLabels(is_train_set)
    features = loadFeatures(labels, unigrams, is_train_set)
    test_set = {'labels': labels, 'features': features}

    model = ensemble.RandomForestClassifier(n_estimators = 20, random_state = 512)
    model_name = 'random forest'
    trainModel(unigrams_flag, train_set, test_set, model, model_name)

    model = ensemble.RandomForestClassifier(n_estimators = 50, random_state = 512)
    trainModel(unigrams_flag, train_set, test_set, model, model_name)

    topK = 100
    getTopFeatures(model, topK, unigrams)
    print 'running time is', datetime.now() - now
