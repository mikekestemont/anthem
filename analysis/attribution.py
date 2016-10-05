import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import numpy as np


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix


##### TMP ###########################################
texts, authors = [], []
for filename in os.listdir('../data/clean/'):
    if not filename.endswith('.txt'):
        continue
    target_authors = ['datheen', 'fruytiers', 'coornhert', 'marnix']

    author = filename.split('+')[0]
    if author.lower() in target_authors:    
        print(filename)
        print(author)
        with open('../data/clean/'+filename, 'r') as f:
            texts.append(f.read())
        authors.append(author)

print(len(texts))
print(len(authors))

wilhelmus = open('../data/clean/geuz+_geu001etku01_0038.txt').read()
##### TMP ###########################################

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(4, 4), use_idf=False, max_features=1000, norm='l1')
scaler = StandardScaler(with_mean=False)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(authors)

X = vectorizer.fit_transform(texts).toarray()
X = scaler.fit_transform(X)

w = vectorizer.transform([wilhelmus]).toarray()
w = scaler.transform(w)

#clf = KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='cityblock')
clf = SVC()
nb = X.shape[0]
loo = LeaveOneOut(nb)


silver, gold = [], []
for train, test in loo:
    X_train, X_test = X[train], X[test]
    y_test = [y[i] for i in test]
    y_train = [y[i] for i in train]
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    silver.append(pred[0])
    gold.append(y_test[0])

print(silver)
print(gold)

print('Accuracy after LOO:', accuracy_score(silver, gold))

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
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# confusion matrix
plt.clf()
T = label_encoder.inverse_transform(gold)
P = label_encoder.inverse_transform(silver)
cm = confusion_matrix(T, P, labels=label_encoder.classes_)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
np.set_printoptions(precision=2)
sns.plt.figure()
plot_confusion_matrix(cm_normalized, target_names=label_encoder.classes_)
sns.plt.savefig('conf_matrix.pdf')


# wilhelmus attribution:
clf.fit(X, y)
print(w.shape)
attrib = label_encoder.inverse_transform(clf.predict(w))[0]
print('Wilhelmus attributed to:', attrib)



