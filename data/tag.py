"""
Standalone script to train two MBT's: one to lemmatize;
another to to POS-tag. Later the result of both taggers
is joined
"""

from __future__ import print_function

import codecs
import glob
import os
import subprocess


cnt = 0

lemma_f = codecs.open('train_lemma.txt', 'w', 'utf8')
pos_f = codecs.open('train_pos.txt', 'w', 'utf8')
for root, dirs, files in os.walk('tagged'):
    for name in files:
        if name.endswith('.txt'):
            fn = os.path.join(root, name)
            print(fn)
            with codecs.open(fn, 'r', 'utf8') as in_f:
                for line in in_f:
                    cnt += 1
                    if cnt % 15 == 0 and cnt > 10:
                        lemma_f.write('<utt>\n')
                        pos_f.write('<utt>\n')
                    comps = line.strip().replace('/', '')
                    comps = line.strip().replace('/', '')
                    comps = comps.split()
                    try:
                        lemma_f.write('\t'.join((comps[0].replace('~', ''), comps[1]))+'\n')
                        pos_f.write('\t'.join((comps[0].replace('~', ''), comps[2]))+'\n')
                    except:
                        print(line)
                        raise ValueError

print(cnt)
lemma_f.close()
pos_f.close()


subprocess.call("mbtg -T train_lemma.txt -p ddfa -P dFapsss", shell=True)
subprocess.call("mbtg -T train_pos.txt -p ddfa -P dFapsss", shell=True)


for filename in os.listdir('test_files'):
    print(filename)
    if filename.endswith('.txt'):
        bn = os.path.basename(filename)
        g = open('test_files/' + filename, 'r')
        text = g.read()
        g.close()
        words = text.strip().split()

        k = open('test_files/' + filename, 'w') 
        cnt = 0
        for w in words:
            w = w.strip()
            if w:
                k.write(w + '\t//\tL\tP\n')
                cnt += 1
                if cnt % 15 == 0 and cnt:
                    k.write('<utt>\n')
        k.close()

        # LEMMA
        nf_in = filename + '.lemma_in'
        nf_out = filename + '.lemma_out'
        with codecs.open('test_files/' + nf_in, 'w', 'utf8') as f:
            for line in codecs.open('test_files/'+filename, 'r', 'utf8'):
                comps = line.strip().split()
                try:
                    f.write('\t'.join((comps[0].replace('~', ''), comps[1]))+'\n')
                except IndexError:
                    pass
        subprocess.call("mbt -s train_lemma.txt.settings -T test_files/"+nf_in+" > test_files/"+nf_out, shell=True)

        # POS
        nf_in = filename + '.pos_in'
        nf_out = filename + '.pos_out'
        with codecs.open('test_files/'+nf_in, 'w', 'utf8') as f:
            for line in codecs.open('test_files/'+filename, 'r', 'utf8'):
                comps = line.strip().split()
                try:
                    f.write('\t'.join((comps[0].replace('~', ''), comps[2]))+'\n')
                except IndexError:
                    print(line, '!!!!')
                    pass
        subprocess.call("mbt -s train_pos.txt.settings -T test_files/"+nf_in+" > test_files/"+nf_out, shell=True)

        # merge
        nf_lem = codecs.open('test_files/'+filename + '.lemma_out', 'r', 'utf8')
        nf_pos = codecs.open('test_files/'+filename + '.pos_out', 'r', 'utf8')
        lem_lines = nf_lem.readlines()
        pos_lines = nf_pos.readlines()
        nf_lem.close()
        nf_pos.close()

        with codecs.open('test_files/'+filename, 'w', 'utf8') as g:
            for lem_line, pos_line in zip(lem_lines, pos_lines):
                if lem_line.strip() == '<utt>':
                    continue
                try:
                    tok1, kno1, old1, new1 = lem_line.strip().split()
                    tok2, kno2, old2, new2 = pos_line.strip().split()
                    if tok1 == tok2:
                        g.write('\t'.join((tok1, kno1, new1, new2))+'\n')
                except:
                    print(lem_line, 'lemline')
                    print(pos_line, 'posline')
                    raise ValueError



