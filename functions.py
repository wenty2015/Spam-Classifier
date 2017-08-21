from elasticsearch import Elasticsearch
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import recall_score, precision_score, accuracy_score, \
                            f1_score, roc_auc_score
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def getLabels(es, is_train_set = 1):
    print 'load labels'
    labels, doc_map = [], {}

    scroll = getScrollInfoLabel(es, is_train_set)
    scroll_id = scroll['_scroll_id']
    cnt = 0
    while True:
        if len(scroll['hits']['hits']) == 0:
            break
        for doc in scroll['hits']['hits']:
            doc_info = doc['fields']
            doc_id = doc_info['docno'][0]
            labels.append((doc_id, doc_info['label'][0]))
            doc_map[doc_id] = cnt
            cnt += 1
        scroll = es.scroll(scroll_id = scroll_id, scroll= "5m")
        scroll_id = scroll['_scroll_id']
    print 'loaded', len(labels), 'mails'
    return labels, doc_map

def dumpLabels(unigrams_flag, labels, is_train_set):
    set_type = 'train' if is_train_set else 'test'
    foler_name = '../results/' + unigrams_flag + '/'
    file_name = 'labels_' + set_type + '.txt'
    with open(foler_name + file_name, 'w') as f:
        for l in labels:
            f.write(' '.join(map(lambda d: str(d), l)) + '\n')

def getScrollInfoLabel(es, is_train_set = 1, size = 5000):
    return es.search(
        index = 'spam', doc_type = 'document', size = size,
        scroll = '1m',
        body = { "query":{
                    "match":{ "is_train_set": is_train_set }
                },
                "script_fields": {
                    "docno":{ "script":{
                            "lang": "groovy",
                            "inline": "_source['docno']" } },
                    "label":{ "script":{
                            "lang": "groovy",
                            "inline": "_source['label']" } }
                }
            })

def getScrollInfoFeatures(es, w, is_train_set = 1, size = 1000):
    if len(w.split(' ')) > 1:
        return es.search(
            index = 'spam', doc_type = 'document', size = size,
            scroll = '1m',
            body = { "query":
                        {"bool": {
                              "must":
                                { "match_phrase":
                                    { "text": {
                                        "query": w ,
                                        "slop": 5}}},
                              "filter": {
                                "term": { "is_train_set": is_train_set }
                              }
                            }
                        },
                        "script_fields": {
                            "docno":{ "script":{
                                    "lang": "groovy",
                                    "inline": "_source['docno']" } }
                        }
                })
    else:
        return es.search(
            index = 'spam', doc_type = 'document', size = size,
            scroll = '1m',
            body = { "query":{
                        "bool":{
                            "must": [
                                { "match": { "is_train_set":  is_train_set}},
                                { "match": { "text": w}}
                              ]
                        }
                    },
                    "script_fields": {
                        "docno":{ "script":{
                                "lang": "groovy",
                                "inline": "_source['docno']" } }
                    }
                })

def getData(dataset):
    features = dataset['features']
    labels = np.asarray(map(lambda d: d[1], dataset['labels']))
    docs = map(lambda d: d[0], dataset['labels'])
    return features, labels, docs

def trainModel(spam_words_flag, train_set, test_set, model, model_name):
    print 'train model using', model_name
    print 'train set'
    features, labels, docs = getData(train_set)
    model = model.fit(features, labels)
    outputResult(spam_words_flag, model, model_name, features, labels, docs, 1)

    print 'test set'
    features, labels, docs = getData(test_set)
    outputResult(spam_words_flag, model, model_name, features, labels, docs, 0)

def crossValidation(dataset, model, model_name):
    print 'cross validation for model', model_name
    kf = KFold(n_splits = 5, shuffle = True)
    mean = []
    c_list = [10 ** i for i in xrange(-2, 3)]
    features, labels, docs = getData(dataset)
    for c in c_list:
        model.C = c
        score = []
        for train_index, test_index in kf.split(features):
            model = model.fit(features[train_index,:], labels[train_index])
            score.append(model.score(features[test_index,:], labels[test_index]))
        mean.append(np.mean(score))
    print mean
    C = c_list[np.argsort(mean)[-1]]
    print 'optimal C', C
    return C

def plotPrecisionRecall(precision, recall, model_name, set_type, spam_words_flag):
    plt.figure()
    plt.plot(recall, precision)
    plt.title(' '.join(['precision-recall plot for', set_type, model_name]))
    plt.xlabel('recall')
    plt.ylabel('precision')
    foler_name = '../results/' + spam_words_flag + '/plots/'
    file_name = ' '.join([set_type, model_name]) + '.png'
    plt.savefig(foler_name + file_name)

def plotROCCurve(fpr, tpr, model_name, set_type, spam_words_flag):
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title(' '.join(['ROC curve for', set_type, model_name]))
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    foler_name = '../results/' + spam_words_flag + '/plots/'
    file_name = ' '.join([set_type, model_name, 'ROC']) + '.png'
    plt.savefig(foler_name + file_name)

def outputResult(spam_words_flag, model, model_name, features, labels, docs,
                is_train_set):
    set_type = 'train' if is_train_set else 'test'
    # print 'mean accuray of', set_type, model.score(features, labels)
    file_name = ''.join(['../results/', spam_words_flag, '/', model_name, '_',
                        set_type, '.txt'])
    with open(file_name, 'w') as f:
        # find the index of label 1
        index_1 = model.classes_.tolist().index(1)
        predict = map(lambda p: p[index_1], model.predict_proba(features))

        precision, recall, _ = precision_recall_curve(labels, predict)
        plotPrecisionRecall(precision, recall, model_name, set_type, spam_words_flag)

        fpr, tpr, _ = roc_curve(labels, predict)
        plotROCCurve(fpr, tpr, model_name, set_type, spam_words_flag)

        predict_labels = model.predict(features)
        print 'precision = %.4f, recall = %.4f, f1-score = %.4f, accuracy = %.4f, auc = %.4f' %\
                            (precision_score(labels, predict_labels),
                            recall_score(labels, predict_labels),
                            f1_score(labels, predict_labels),
                            accuracy_score(labels, predict_labels),
                            roc_auc_score(labels, predict_labels))

        result = sorted(zip(predict, docs, labels),
                        key = lambda i: [-i[0], i[1]])
        for r in result:
            pred, doc_id, label = r
            f.write(' '.join([doc_id, str(pred), str(label)]) + '\n')
