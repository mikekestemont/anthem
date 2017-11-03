import os
from itertools import product

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix

texts, labels = [], []
direc = '../data/tagged/rich/'
ext = '.txt'
min_len = 150
nb_mfi = 300
ngrams = (1, 1)
target_authors = ['marnix', 'datheen', 'haecht', 'fruytiers', 'heere']
#target_authors = ['haecht', 'datheen']

def identity(x):
    return x

def plot_confusion_matrix(cm, target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.tick_params(labelsize=6)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=90)
    plt.yticks(tick_marks, target_names)
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def main():
    wilhelmus = None
    for filename in os.listdir(direc):

        if not filename.endswith(ext):
            continue
        
        cs = filename.replace(ext, '').split('+')
        if len(cs) == 2:
            o, id_ = cs
        else:
            continue

        if o in target_authors or 'wilhelmus' in id_:

            with open(direc + filename, 'r') as f:
                tlps = []
                for line in f.readlines():
                    line = line.strip().lower()
                    tok, lem, pos = line.split()
                    lem_comps, pos_comps = lem.split('+'), pos.split('+')
                    if len(lem_comps) == len(pos_comps):
                        for l, p in zip(lem_comps, pos_comps):
                            if 'punc' not in p and '???' not in l:
                                tlps.append(l + '_' + p)
                    else:
                        #print('error:', line)
                        pass

                if 'wilhelmus' in id_:
                    wilhelmus = tlps
                else:
                    texts.append(tlps)
                    labels.append(o)

    vectorizer = TfidfVectorizer(analyzer=identity, ngram_range=ngrams,
                                 use_idf=False, max_features=nb_mfi, norm='l1')
    X_ = vectorizer.fit_transform(texts + [wilhelmus]).toarray()
    vocabulary = vectorizer.get_feature_names()

    scaled_X = StandardScaler(with_mean=False).fit_transform(X_)

    X = scaled_X[:-1, :]
    W = scaled_X[-1, :]

    label_encoder = LabelEncoder()
    int_labels = label_encoder.fit_transform(labels)

    clf = SVC(kernel='linear', probability=True)
    nb = X.shape[0]
    loo = LeaveOneOut(nb)

    silver, gold = [], []
    for train, test in loo:
        X_train, X_test = X[train], X[test]
        y_test = [int_labels[i] for i in test]
        y_train = [int_labels[i] for i in train]
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        silver.append(pred[0])
        gold.append(y_test[0])

    print('Accuracy after LOO:', accuracy_score(silver, gold))

    # confusion matrix
    plt.clf()
    T = label_encoder.inverse_transform(gold)
    P = label_encoder.inverse_transform(silver)
    cm = confusion_matrix(T, P, labels=label_encoder.classes_)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=2)
    sns.plt.figure()
    plot_confusion_matrix(cm_normalized, target_names=label_encoder.classes_)
    
    # wilhelmus attribution:
    clf.fit(X, int_labels)
    probas = list(clf.predict_proba([W]).ravel())
    info = []
    for idx, label in enumerate(label_encoder.classes_):
        info.append(label + ': ' + str("%.2f" % probas[idx]))
    plt.title('LOO Attribution accuracy: ' + str(round(accuracy_score(silver, gold), 2))\
                 + '\nWilhelmus attribution probabilities:\n'+\
                  ' | '.join(info), fontsize=7)

    sns.plt.savefig('../figures/conf_matrix.pdf')

if __name__ == '__main__':
    main()



