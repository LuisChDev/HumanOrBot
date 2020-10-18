""" módulo en el que se extraen las características del texto de los tweets.
"""

from sklearn.feature_extraction.text import CountVectorizer
import xml.etree.ElementTree as xml
import numpy as np
import pickle
import os

localdir = os.path.dirname(os.path.realpath(__file__))
datadir = '/pan19_author_profiling_training_es'


def read_dataset():
    """lee el contenido de los archivos y crea una matriz con cada tweet y su
    respectivo autor."""
    tweets = []
    autores = []
    for file_ in os.listdir(localdir + datadir):
        if file_[0] == 't':
            continue
        tws = []
        tree = xml.ElementTree(file=localdir + datadir + '/' + file_)
        for elem in tree.find('documents').findall('document'):
            tws.append(elem.text)
        tweets.append(tws)
        autores.append(file_[:-4])
    twarr = np.array(tweets)
    auarr = np.array(autores)
    with open('tweets.pkl', 'wb') as f:
        pickle.dump(twarr, f)
    with open('autores.pkl', 'wb') as f:
        pickle.dump(auarr, f)

def train_test_split():
    """a partir del contenido de los archivos truth*.txt, separa este dataset
    en una parte de training y una de testing."""
    train = []
    test = []
    for idx, archivo in enumerate(['/truth-train.txt', 'truth-dev.txt']):
        with open(localdir + datadir + archivo) as f:
            temp = f.readlines()
        for line in temp:
            values = line.split(":::")
            if idx == 0:
                train.append(values)
            else:
                test.append(values)
