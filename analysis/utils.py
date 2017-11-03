"""
Note: we have to assign the Wilhelmus to a (random) target author 
to be able to to run the verification experiment. Right now, we
use Datheen, but if you select another author this hardly affect
the results.
"""

import os
import re
import random
import shutil

random.seed(10668786786)

def prepare_verification_data(input_dir = '../data/tagged/rich',
                              include_authors = None):
    author_titles = {}
    for filename in os.listdir(input_dir):
        filename = filename.lower()
        if filename.endswith('.txt'):
            auth, id_ = filename.split('+')
            id_ = os.path.splitext(id_)[0]

            if auth not in include_authors:
                continue
            if 'wilhelmus' == id_:
                continue

            if auth not in author_titles:
                author_titles[auth] = []
            author_titles[auth].append(filename)

    # split in test and dev

    for auth in author_titles:
        titles = author_titles[auth]
        random.shuffle(titles)
        author_titles[auth] = {}
        n = int(len(titles) / 2.0)
        author_titles[auth]['dev'] = titles[n:]
        author_titles[auth]['test'] = titles[:n]

        """
        print(auth)
        print('dev:')
        for title in author_titles[auth]['dev']:
            print('\t', title)
        print('test:')
        for title in author_titles[auth]['test']:
            print('\t', title)
        """
            

    if os.path.isdir('../data/verification/wilh_background'):
        shutil.rmtree('../data/verification/wilh_background')

    if os.path.isdir('../data/verification/wilh_test'):
        shutil.rmtree('../data/verification/wilh_test')

    os.mkdir('../data/verification/wilh_background')
    os.mkdir('../data/verification/wilh_test')


    def file_to_tlps(fp):
        tlps = []

        for line in open(fp, 'r'):
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

        return tlps


    # parse and divide:
    for filename in os.listdir(input_dir):

        filename = filename.lower()

        if filename.endswith('.txt'):
            auth, id_ = filename.split('+')
            id_ = os.path.splitext(id_)[0]
            
            if auth not in include_authors:
                continue

            if 'wilhelmus' == id_:
                continue

            id_ = id_.replace('_', '+')

            f = None

            if filename in author_titles[auth]['dev']:
                if not os.path.isdir('../data/verification/wilh_background/'+auth):
                    os.mkdir('../data/verification/wilh_background/'+auth)

                f = open('../data/verification/wilh_background/'+auth+'/'+id_+'.txt', 'w')

            elif filename in author_titles[auth]['test']:
                if not os.path.isdir('../data/verification/wilh_test/'+auth):
                    os.mkdir('../data/verification/wilh_test/'+auth)

                f = open('../data/verification/wilh_test/'+auth+'/'+id_+'.txt', 'w')

            tlps = file_to_tlps(input_dir + '/' + filename)
            f.write(' '.join(tlps))
            f.close()

    with open('../data/verification/wilh_test/datheen/wilhelmus.txt', 'w') as f:
        tlps = file_to_tlps(input_dir + '/geuz+wilhelmus.txt')
        f.write(' '.join(tlps))
