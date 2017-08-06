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

def loadMail(mail_name, es, vocabulary = None):
    with open('../data/trec07p/data/' + mail_name, 'r') as f:
        mail = f.read().decode("ascii", errors = 'ignore')
    mail = email.message_from_string(mail)
    mail_content = ''
    # subject
    if 'Subject' in mail:
        vocabulary = updateVocabulary(mail['Subject'], es, vocabulary)
        mail_content += mail['Subject']
    if mail.is_multipart():
        pl_list = getAllPayLoads(mail)
        for part in pl_list:
            conten_type = part.__getitem__('Content-Type')
            if conten_type and conten_type[:4].lower() == 'text':
                text, vocabulary = cleanText(part.get_payload(), es, vocabulary)
                mail_content += text + ' '
    else:
        conten_type = mail.__getitem__('Content-Type')
        if conten_type and conten_type.[:4].lower() == 'text':
            text, vocabulary = cleanText(mail.get_payload(), es, vocabulary)
            mail_content += text + ' '
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
    if vocabulary is not None:
        tokens = es.indices.analyze(index = 'spam',
                                    body={"analyzer" : "my_english",
                                            "text": text })['tokens']
        if len(tokens) > 0:
            words = map(lambda t: t['token'] , tokens)
            words = filter(lambda w:
                        (w.isdigit() and 1 < float(w) < 10000)
                        or (w.replace('.', '').isalpha() and len(w) < 25)
                        or (w.replace('_', '').isalpha() and len(w) < 25),
                    words)
            #print set(words) - vocabulary
            vocabulary = vocabulary | set(words)
    return vocabulary

def cleanText(text, es, vocabulary = None):
    # print text
    soup = BeautifulSoup(text, "lxml")
    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    text = ' '.join(soup.stripped_strings)
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
        for w in sorted(list(vocabulary)):
            word = w.encode('ascii', 'ignore').decode('ascii')
            if len(word) > 0:
                f.write(word + '\n')
    print 'size of vocabulary is', len(vocabulary)

if __name__ == '__main__':
    vocabulary = set()

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
        dumpES(i, mail, labels[i], 1)
        if (i + 1) % 1000 == 0:
            print i + 1, 'documents'

    dumpVocabulary(vocabulary)

    print 'load test set'
    for i in xrange(train_length, length):
        #print labels[i].mail_name
        mail = loadMail(labels[i].mail_name, es)
        dumpES(i, mail, labels[i], 0)
        if (i + 1 - train_length) % 1000 == 0:
            print i + 1 - train_length, 'documents'
    print 'running time', datetime.now() - now
