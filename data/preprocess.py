#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions to extract songs from the original DBNL XML.
"""

from __future__ import print_function

import os
import glob
import re
import shutil
import codecs
import subprocess
from collections import Counter
import unicodedata

from lxml import etree
from sklearn.cluster import AgglomerativeClustering
from bs4 import BeautifulSoup

# some regular expressions:
whitespace = re.compile(r'\s+')
nasty_pagebreaks = re.compile(r'\s*-<pb n="[0-9]+"></pb>\s*')


def xml_to_texts(filename, max_len=None, min_len=50, sample_len=None):
    """
    Function that parses the file under `filename` and extracts all
    chapter nodes marked as a song in the `interp` element. Only songs
    with length >= `min_len` are considered (if specified); all songs
    are truncated to `max_len` (if specified). Texts are segmented
    to samples of `sample_len` (if specified). Returns a tuple of
    (sample, id) pairs, where id will have a sample index added to
    if `sample_len` has been specified.
    """

    texts, ids = [], []

    xml_str = codecs.open(filename, 'r', 'utf8').read()
    xml_str = xml_str.replace(' encoding="UTF-8"', '') # hack needed for lxml parsing
    xml_str = unicode(BeautifulSoup(xml_str, 'lxml')) # remove entities from the xml
    
    # get rid of nasty pagebreak (pb), breaking up tokens across pages:
    xml_str = re.sub(nasty_pagebreaks, '', xml_str)

    tree = etree.fromstring(xml_str)

    # remove cf- and note-elements (which don't contain actual text):
    for element in tree.xpath('.//cf'):
        element.getparent().remove(element)
    
    # individual articles etc. are represented as div's
    # which have the type-attribute set to 'chapter':
    chapter_nodes = [node for node in tree.findall('.//div')
                        if node.attrib and \
                           'type' in node.attrib and \
                           node.attrib['type'] in ('chapter')]

    for chapter_node in chapter_nodes:
        # try to grab title:
        id_ = None
        for n in chapter_node.findall('.//interp'):
            if n.attrib['type'] == 'song':
                id_ = n.attrib['value']
                break
        if not id_:
            continue # ignore notes without an id
        
        # all text in the articles is contained under p-elements:
        text = ''
        for l_node in chapter_node.findall('.//l'):
            # remove elements that contain meta text (note that we exclude all notes!)
            for tag_name in ('note', 'figure', 'table'):
                etree.strip_elements(l_node, tag_name, with_tail=False)

            # collect the actual text:
            l_text = ''.join(l_node.itertext())

            # add the line node:
            text += l_text + '\n'

        text = text.strip()

        if text:
            text = ''.join([c for c in text if (c.isalpha()
                                or c in ("'") or c.isspace())])
            # remove accents:
            nkfd_form = unicodedata.normalize('NFKD', unicode(text))
            text = nkfd_form.encode('ASCII', 'ignore').decode('ASCII')
            text = text.lower().split()

            if max_len:
                text = text[:max_len]
            if min_len:
                if len(text) < min_len:
                    continue

            if sample_len:
                sample_texts, sample_ids = [], []
                sta_cnt, end_cnt = 0, sample_len
                cnt = 1

                while end_cnt <= len(text):
                    sample_texts.append(' '.join(text[sta_cnt : end_cnt]))
                    sample_ids.append(id_ + '-' + str(cnt))

                    cnt += 1
                    sta_cnt += sample_len
                    end_cnt += sample_len

                texts.extend(sample_texts)
                ids.extend(sample_ids)

            else:
                texts.append(' '.join(text))
                ids.append(id_)

    return zip(texts, ids)

def extract_authors(old_dir = 'authors/',
            new_dir = 'clean/',
            min_len = 50,
            max_len = None,
            sample_len = 500):
    """
    Wrapper function which extracts the relevant texts from
    the dbnl-xml files.
    """
    
    for author in os.listdir(old_dir):
        if os.path.isdir(old_dir+author):
            for filename in glob.glob(old_dir+author+'/*.xml'):
                items = xml_to_texts(filename=filename,
                                     min_len=min_len,
                                     max_len=max_len,
                                     sample_len=sample_len)
                
                if items:
                    for text, id_ in items:
                        fn = author + '+' + id_ + '.txt'
                        with codecs.open(new_dir + fn.lower(), 'w', 'utf8') as f:
                            f.write(text)

def extract_extra(old_dir='extra/',
                     new_dir='clean/',
                     min_len=None,
                     max_len=None,
                     sample_len=None):
    
    for filename in os.listdir(old_dir):
        print(filename)
        with open(old_dir + filename, 'r') as f:
            text = f.read()
        words = text.lower().split()

        comps = filename.split('_')
        filename = comps[0] + '+' + '_'.join(comps[1:])

        with open(new_dir + filename.lower(), 'w') as f:
            for w in words:
                w = ''.join([c for c in w if c.isalpha() or c == "'"])
                if w:
                    f.write(w+' ')

def extract_songbook(filepath,
                     new_dir='clean/',
                     min_len=None,
                     max_len=None,
                     sample_len=None):
    """
    Wrapper function to extract the (unsegmented) songs 
    from the Geuzenliedboek.
    """
    
    items = xml_to_texts(filename=filepath,
                         min_len=min_len,
                         max_len=max_len,
                         sample_len=sample_len)
                
    if items:
        for text, id_ in items:
            fn = 'geuz+' + id_.lower() + '.txt'
            with codecs.open(new_dir + fn, 'w', 'utf8') as f:
                f.write(text)

def main():
    new_dir = 'clean'
    if os.path.isdir(new_dir):
        shutil.rmtree(new_dir)
    os.mkdir(new_dir)

    extract_authors()
    extract_extra()
    extract_songbook('songbooks/_geu001etku01_01.xml')

    try:
        shutil.rmtree('test_files')
    except:
        pass
    
    shutil.copytree('clean', 'test_files')

if __name__ == '__main__':
    main()

