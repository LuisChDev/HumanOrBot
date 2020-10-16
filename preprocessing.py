"""En este script se desarrolla el preprocesamiento, es decir, se pasa del
dataset crudo a una forma de array que pueda procesarse por las herramientas
de predicción.

Esencialmente, se espera de este archivo dos variables, datarray y datarget,
que deben ser arrays de numpy, la primera de dos dimensiones, con cada row
representando una entrada del dataset con sus respectivos atributos, y datarget
debe ser unidimensional, conteniendo la etiqueta real de cada punto de datos.

Autor: Roque del Río
"""

# placeholder. las variables finales deben tener el mismo tipo (nparray).

import xml.etree.ElementTree as ET

# get XML files
def getFiles(master):
    dfset = []
    with open("./data/" + master + ".txt", 'r+' ,encoding="utf-8") as file:
        for line in file:
            entry = line.split(':::')
            author = entry[0]
            botValue = entry[1]
            dict = buildDict(author)
            dfset.append(dict)

    return dfset

def buildDict(author):
    file = "./data/" + author + ".xml"

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

    dict = {
        "author" : author,
        "rts" : rts,
        "links" : links,
        "punctuation" : punctuation,
        "hashtags" : hashtags,
        "tags" : tags
    }

    return dict



