"""En este script se desarrolla el preprocesamiento, es decir, se pasa del
dataset crudo a una forma de array que pueda procesarse por las herramientas
de predicción.

Esencialmente, se espera de este archivo dos variables, datarray y datarget,
que deben ser arrays de numpy, la primera de dos dimensiones, con cada row
representando una entrada del dataset con sus respectivos atributos, y datarget
debe ser unidimensional, conteniendo la etiqueta real de cada punto de datos.

Autor: Roque del Río
"""

import xml.etree.ElementTree as ET

# get XML files
def get_files(master):
    dfset = []
    with open("./pan19_author_profiling_training_es/" + master + ".txt", 'r+' ,encoding="utf-8") as file:
        for line in file:
            entry = line.split(':::')
            author = entry[0]
            botvalue = entry[1]
            entrydict = build_dict(author)
            if botvalue == 'bot':
                entrydict.update({"botvalue": 1})
            dfset.append(entrydict)

    return dfset

def build_dict(author):
    file = "./pan19_author_profiling_training_es/" + author + ".xml"

    # XML stuff
    tree = ET.parse(file)
    root = tree.getroot()


    rts = 0
    links = 0
    punctuation = 0
    hashtags = 0
    tags = 0

    for entry in root.iter('document'):
        text = entry.text
        # RT
        rts += text.count('RT @')
        # Link
        links += text.count('http://')
        links += text.count('https://')
        # Punct
        punctuation += text.count('. ')
        punctuation += text.count(', ')
        punctuation += text.count('; ')
        # hashtags
        hashtags += text.count('#')
        # tags
        tags += text.count('@')

    entrydict = {
        "author" : author,
        "rts" : rts,
        "links" : links,
        "punctuation" : punctuation,
        "hashtags" : hashtags,
        "tags" : tags,
        "botvalue" : 0
    }

    return entrydict



