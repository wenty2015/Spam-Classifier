from sklearn.metrics import accuracy_score
import numpy as np

def evaluation(file_name, topK = 50):
    labels = []
    with open(file_name, 'r') as f:
        for line in f.readlines()[:topK]:
            results = line.rstrip('\n').split(' ')
            labels.append(int(results[2]))
    print '%f percent are spam emails' % (np.sum(labels) * 100. / topK)

def accuracy(file_name, threshold = .5):
    cnt, accuracy_cnt = 0, 0
    with open(file_name, 'r') as f:
        for line in f.readlines():
            cnt += 1
            results = line.rstrip('\n').split(' ')
            predict, label = float(results[1]), int(results[2])
            if predict < threshold and label == 0 \
                or predict > threshold and label == 1:
                accuracy_cnt += 1
    print 'threshold = %f, accuracy = %f' % \
            (threshold, accuracy_cnt * 1. / cnt)

file_name = ['../results/website/random forest_test.txt',
            '../results/unigrams/random forest_test.txt']
print_name = ['spam list', 'unigrams']
for f, p in zip(file_name, print_name):
    print 'top documents for', p
    evaluation(f)
    for threshold in [0.4, 0.45, 0.5, 0.55, 0.6]:
        accuracy(f, threshold)
