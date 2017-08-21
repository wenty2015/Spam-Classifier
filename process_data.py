import numpy as np
from elasticsearch import Elasticsearch
from bs4 import BeautifulSoup
import email
from datetime import datetime

class Label():
    def __init__(self, label, mail_name):
        self.label = 1 if label == 'spam' else 0
        self.mail_name = mail_name

def printStats(labels, prop):
    length = len(labels)
    print 'number of samples is', length
    print 'proportion of train set is', prop
    train_length = int(length * prop)
    train_spam = np.sum(map(lambda l: l.label, labels[:train_length]))
    print 'train set:', train_spam, 'spams', train_length - train_spam, 'hams'
    test_length = length - train_length
    test_spam = np.sum(map(lambda l: l.label, labels[train_length:]))
    print 'test set:', test_spam, 'spams', test_length - test_spam, 'hams'

def processLabels():
    labels_file = '../data/trec07p/full/index'
    labels = []
    with open(labels_file, 'r') as f:
        for line in f.readlines():
            l, mail = line.rstrip('\n ').split()
            mail_name = mail.split('/')[-1]
            labels.append(Label(l, mail_name))
    return labels

def processText(text):
    return text.replace('_', ' _ ').replace('.', '. ')

def loadSubject(mail):
    mail_subject = ''
    if 'Subject' in mail:
        content = processText(mail['Subject'])
        mail_subject += 'Subject: '+ content + '\n'
    if 'From' in mail:
        content = processText(mail['From'])
        mail_subject += 'From: '+ content + '\n'
    if 'To' in mail:
        content = processText(mail['To'])
        mail_subject += 'To: '+ content + '\n'
    return mail_subject

def loadMail(mail_name, es, vocabulary = None):
    with open('../data/trec07p/data/' + mail_name, 'r') as f:
        mail = f.read().decode("ascii", errors = 'ignore')
    mail = email.message_from_string(mail)

    mail_content = loadSubject(mail)
    if len(mail_content) > 0:
        vocabulary = updateVocabulary(mail_content, es, vocabulary)

    if mail.is_multipart():
        pl_list = getAllPayLoads(mail)
        for part in pl_list:
            conten_type = part.__getitem__('Content-Type')
            if conten_type:
                mail_content += 'Content-Type: ' + conten_type + '\n'
                if conten_type[:4].lower() == 'text':
                    text = processText(part.get_payload())
                    text, vocabulary = cleanText(text, es, vocabulary)
                    mail_content += text + '\n'
    else:
        conten_type = mail.__getitem__('Content-Type')
        if conten_type:
            mail_content += 'Content-Type: ' + conten_type + '\n'
            if conten_type[:4].lower() == 'text':
                text = processText(mail.get_payload())
                text, vocabulary = cleanText(text, es, vocabulary)
                mail_content += text + '\n'
    return mail_content, vocabulary

def getAllPayLoads(mail):
    pl, pl_list = [mail], []
    while pl:
        p = pl.pop()
        if isinstance(p.get_payload(), list):
            pl.extend(p.get_payload())
        else:
            pl_list.append(p)
    return pl_list

def updateVocabulary(text, es, vocabulary = None):
    if text != '' and vocabulary is not None:
        tokens = es.indices.analyze(index = 'spam',
                                    body={"analyzer" : "my_english",
                                            "text": text })['tokens']
        if len(tokens) > 0:
            ori_tokens = es.indices.analyze(index = 'spam',
                                            body={"analyzer" : "standard",
                                                    "text": text })['tokens']
            ori_words = {}
            for t in ori_tokens:
                ori_words[t['position']] = t['token']

            words = map(lambda t: (t['token'], t['position']) , tokens)
            words = filter(lambda w:
                        (w[0].isdigit() and 1 < float(w[0]) < 1000 and
                            w[0][0] != '0')
                        or (w[0].replace('.', '').isalpha() and len(w[0]) < 20)
                        or (w[0].replace('_', '').isalpha() and len(w[0]) < 20),
                    words)
            for w in words:
                word, position = w
                if word not in vocabulary:
                    vocabulary[word] = ori_words[position]
    return vocabulary

def cleanText(text, es, vocabulary = None):
    # print text
    soup = BeautifulSoup(text, "lxml")
    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    text = ' '.join(soup.stripped_strings)
    text = text.replace(' _', ' ').replace('_ ', ' ')
    vocabulary = updateVocabulary(text, es, vocabulary)
    return text, vocabulary

def dumpES(cnt, mail, label, flag):
    es.index(index = 'spam',
             doc_type = 'document',
             id = cnt,
             body = {"docno": label.mail_name,
                    "text": mail,
                    "label": label.label,
                    "is_train_set": flag})

def dumpVocabulary(vocabulary):
    with open('../results/vocabulary.txt', 'w') as f:
        for w_stemmed in sorted(vocabulary.keys()):
            w = vocabulary[w_stemmed]
            word_stemmed = w_stemmed.encode('ascii', 'ignore').decode('ascii')
            if len(word_stemmed) > 0:
                word = w.encode('ascii', 'ignore').decode('ascii')
                f.write(' '.join([word_stemmed, word]) + '\n')
    print 'size of vocabulary is', len(vocabulary)

if __name__ == '__main__':
    vocabulary = {}

    labels = processLabels()
    np.random.seed(512)
    np.random.shuffle(labels)

    TRAIN_PROPORTION = 0.8
    length = len(labels)
    train_length = int(length * TRAIN_PROPORTION)
    printStats(labels, TRAIN_PROPORTION)

    es = Elasticsearch()

    now = datetime.now()
    print 'load train set'
    for i in xrange(train_length):
        #print labels[i].mail_name
        mail, vocabulary = loadMail(labels[i].mail_name, es, vocabulary)
        dumpES(i, mail.replace('%', ' percent ').replace('$', ' dollar '),
                    labels[i], 1)
        if (i + 1) % 1000 == 0:
            print i + 1, 'documents, size of vocabulary is', len(vocabulary)

    dumpVocabulary(vocabulary)

    print 'load test set'
    for i in xrange(train_length, length):
        #print labels[i].mail_name
        mail, _ = loadMail(labels[i].mail_name, es)
        dumpES(i, mail.replace('%', ' percent ').replace('$', ' dollar '),
                    labels[i], 0)
        if (i + 1 - train_length) % 1000 == 0:
            print i + 1 - train_length, 'documents'
    print 'running time', datetime.now() - now
