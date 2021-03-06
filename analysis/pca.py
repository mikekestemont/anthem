import os

import numpy as np

import seaborn as sb
sb.set_style('white')
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn import svm


def identity(x):
    return x

texts, labels = [], []
direc = '../data/tagged/rich/'
ext = '.txt'
min_len = 150
mfi = 300
target_authors = ['datheen', 'marnix']
anon_id = 'wilhelmus' # confront 005 and 006 with marnix

def main():
    anon = None

    for filename in os.listdir(direc):

        if not filename.endswith(ext):
            continue
        
        cs = filename.replace(ext, '').split('+')
        if len(cs) == 2:
            o, id_ = cs
        else:
            continue

        if o in target_authors or anon_id in id_:

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

                if anon_id in id_:
                    anon = tlps
                else:
                    texts.append(tlps)
                    labels.append(o)

    vectorizer = TfidfVectorizer(analyzer=identity, use_idf=False,
                                 max_features=mfi, norm='l1')
    X_ = vectorizer.fit_transform(texts + [anon]).toarray()
    vocabulary = vectorizer.get_feature_names()

    scaled_X = StandardScaler(with_mean=False).fit_transform(X_)

    pca = PCA(n_components=2)
    pca_X = pca.fit_transform(scaled_X)

    X = pca_X[:-1, :]
    A = pca_X[-1, :]

    loadings = pca.components_.transpose()
    var_exp = pca.explained_variance_ratio_

    label_encoder = LabelEncoder()
    int_labels = label_encoder.fit_transform(labels)
    clf = svm.SVC(kernel='linear', probability=True).fit(X, int_labels)
    probas = list(clf.predict_proba([A]).ravel())

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(111)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)


    info = []
    for idx, label in enumerate(label_encoder.classes_):
        info.append(label + ': ' + str("%.2f" % probas[idx]))
    plt.title('Probabilities:\n'+ ' | '.join(info))

    ax1.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    x1, x2 = X[:, 0], X[:, 1]
    plt.scatter(x1, x2, edgecolors='none', facecolors='none')
    for p1, p2, a in zip(x1, x2, labels):
        plt.text(p1, p2, a.lower()[:3], ha='center',
            va='center', fontdict={'family': 'Arial', 'size': 12})

    plt.scatter(A[0], A[1], marker='o', color='black')
    plt.text(A[0], A[1], anon_id[:4], va='baseline', ha='center', color='w')

    ax1.set_xlabel('PC1 ('+ str(round(var_exp[0] * 100, 2)) +'%)')
    ax1.set_ylabel('PC2 ('+ str(round(var_exp[1] * 100, 2)) +'%)')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.savefig('../figures/author_mesh.pdf')
    plt.clf()

    # from: http://stackoverflow.com/questions/10481990/matplotlib-axis-with-two-scales-shared-origin
    def align_yaxis(ax1, v1, ax2, v2):
        #adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1
        _, y1 = ax1.transData.transform((0, v1))
        _, y2 = ax2.transData.transform((0, v2))
        inv = ax2.transData.inverted()
        _, dy = inv.transform((0, 0)) - inv.transform((0, y1 - y2))
        miny, maxy = ax2.get_ylim()
        ax2.set_ylim(miny + dy, maxy + dy)

    def align_xaxis(ax1, v1, ax2, v2):
        #adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1
        x1, _ = ax1.transData.transform((v1, 0))
        x2, _ = ax2.transData.transform((v2, 0))
        inv = ax2.transData.inverted()
        dx, _ = inv.transform((0, 0)) - inv.transform((x1 - x2, 0))
        minx, maxx = ax2.get_xlim()
        ax2.set_xlim(minx + dx, maxx + dx)

    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(111)
    x1, x2 = X[:, 0], X[:, 1]
    ax1.scatter(x1, x2, edgecolors='none', facecolors='none')
    for p1, p2, a in zip(x1, x2, labels):
        ax1.text(p1, p2, a.lower()[:3], ha='center',
                 va='center',
                fontdict={'family': 'Arial', 'size': 12})

    ax1.set_xlabel('PC1 ('+ str(round(var_exp[0] * 100, 2)) +'%)')
    ax1.set_ylabel('PC2 ('+ str(round(var_exp[1] * 100, 2)) +'%)')

    plt.scatter(X[:, 0], X[:, 1], marker=None, color='black')
    plt.text(A[0], A[1], anon_id[:4], va='baseline', ha='center', color='red')

    ax2 = ax1.twinx().twiny()
    l1, l2 = loadings[:, 0], loadings[:, 1]
    ax2.scatter(l1, l2, 100, edgecolors='none', facecolors='none');
    for x, y, l in zip(l1, l2, vectorizer.get_feature_names()):
        ax2.text(x, y, l, ha='center', va="center", color="darkgrey",
                fontdict={'family': 'Arial', 'size': 12})

    # align the axes:
    align_xaxis(ax1, 0, ax2, 0)
    align_yaxis(ax1, 0, ax2, 0)

    # add lines through origins:
    plt.axvline(0, ls='dashed', c='lightgrey')
    plt.axhline(0, ls='dashed', c='lightgrey')
    plt.savefig('../figures/author_loadings.pdf')

if __name__ == '__main__':
    main()
