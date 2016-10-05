import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import numpy as np

import seaborn as sb
sb.set_style('white')
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

def include_pos(pos):
    try:
        target_pos = ['adj', 'n', 'v']
        excl_rest = ['aux_cop']
        base, rest = pos.split('(')
        rest = rest.replace(')', '')
        if base in target_pos and not any(s in rest for s in excl_rest):
            return True
        else:
            return False
    except:
        #print('error:', pos)
        return False

texts, ids_ = [], []
direc = '../data/tagged/rich/'
ext = '.txt'
min_len = 150

for filename in os.listdir(direc):

    if not filename.endswith(ext):
        continue
    
    cs = filename.replace(ext, '').split('+')
    if len(cs) == 2:
        o, id_ = cs
    else:
        continue

    if o == 'geuz':
        id_ = id_.split('_')[-1]
        with open(direc + filename, 'r') as f:
            lemmas = []
            for line in f.readlines():
                line = line.strip().lower()
                tok, lem, pos = line.split()
                lem_comps, pos_comps = lem.split('+'), pos.split('+')
                if len(lem_comps) == len(pos_comps):
                    for l, p in zip(lem_comps, pos_comps):
                        if include_pos(p) and '?' not in l:
                            lemmas.append(l.lower())
                else:
                    #print('error:', line)
                    pass

            if len(lemmas) >= min_len:
                texts.append(lemmas)
                ids_.append(id_)

def identity(x):
    return x


vectorizer = TfidfVectorizer(analyzer=identity, use_idf=True, max_features=300, norm='l1')
X = vectorizer.fit_transform(texts).toarray()
vocabulary = vectorizer.get_feature_names()

X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

loadings = pca.components_.transpose()
var_exp = pca.explained_variance_ratio_

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
x1, x2 = X_pca[:, 0], X_pca[:, 1]
ax1.scatter(x1, x2, edgecolors='none', facecolors='none')
for p1, p2, a in zip(x1, x2, ids_):
    if 'wil' in a:
        print('++++')
        ax1.text(p1, p2, a.lower()[:3], ha='center',
                 va='center', color='red',
                fontdict={'family': 'Arial', 'size': 12})
    else:
        ax1.text(p1, p2, a.lower()[:3], ha='center',
             va='center',
            fontdict={'family': 'Arial', 'size': 12})


ax1.set_xlabel('PC1 ('+ str(round(var_exp[0] * 100, 2)) +'%)')
ax1.set_ylabel('PC2 ('+ str(round(var_exp[1] * 100, 2)) +'%)')


ax2 = ax1.twinx().twiny()
l1, l2 = loadings[:,0], loadings[:,1]
ax2.scatter(l1, l2, 100, edgecolors='none', facecolors='none');
for x, y, l in zip(l1, l2, vocabulary):
    ax2.text(x, y, l, ha='center', va="center", color="darkgrey",
            fontdict={'family': 'Arial', 'size': 12})

# align the axes:
align_xaxis(ax1, 0, ax2, 0)
align_yaxis(ax1, 0, ax2, 0)

# add lines through origins:
plt.axvline(0, ls='dashed', c='lightgrey')
plt.axhline(0, ls='dashed', c='lightgrey')
plt.savefig('geuz_loadings.pdf')



