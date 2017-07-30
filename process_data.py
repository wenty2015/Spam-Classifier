import numpy as np
from elasticsearch import Elasticsearch
from bs4 import BeautifulSoup
import enchant, re, email
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

def loadMail(mail_name):
    with open('../data/trec07p/data/' + mail_name, 'r') as f:
        mail = f.read().decode("ascii", errors = 'ignore')
    mail = email.message_from_string(mail)
    mail_content = ''
    # subject
    if 'Subject' in mail:
        mail_content += removeNoneEnglish(mail['Subject'])
    if mail.is_multipart():
        pl_list = getAllPayLoads(mail)
        for part in pl_list:
            mail_content += cleanText(part.get_payload())
    else:
        mail_content += cleanText(mail.get_payload())
    return mail_content

def getAllPayLoads(mail):
    pl, pl_list = [mail], []
    while pl:
        p = pl.pop()
        if isinstance(p.get_payload(), list):
            pl.extend(p.get_payload())
        else:
            pl_list.append(p)
    return pl_list

def removeNoneEnglish(text):
    words = re.findall(r"\w+(?:\.\w+)*",text) #TOKENIZER.tokenize(text)
    words = filter(lambda w: 25 > len(w) > 1 and
                                (DICT.check(w.lower()) or DICT.check(w)
                                    or (w.isdigit() and 0 < int(w) < 3000)),
                            words)
    return ' '.join(words) + '\n'

def cleanText(text):
    # print text
    soup = BeautifulSoup(text, "lxml")
    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    text = ' '.join(soup.stripped_strings)
    # remove non-English words
    return removeNoneEnglish(text)

def dumpES(cnt, mail, label, flag):
    es.index(index = 'spam',
             doc_type = 'document',
             id = cnt,
             body = {"docno": label.mail_name,
                    "text": mail,
                    "label": label.label,
                    "is_train_set": flag})

#TOKENIZER = RegexpTokenizer(r"\w+(?:\.\w+)*")
DICT = enchant.Dict('en_US')

if __name__ == '__main__':
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
        mail = loadMail(labels[i].mail_name)
        dumpES(i, mail, labels[i], 1)
        if (i + 1)%5000 == 0:
            print i + 1, 'documents'

    print 'load test set'
    for i in xrange(train_length, length):
        #print labels[i].mail_name
        mail = loadMail(labels[i].mail_name)
        dumpES(i, mail, labels[i], 0)
        if (i + 1 - train_length) %5000 == 0:
            print i + 1 - train_length, 'documents'
    print 'running time', datetime.now() - now
