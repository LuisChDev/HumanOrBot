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
from sklearn.datasets import load_iris

dataset = load_iris()
datarray = dataset.data
datarget = dataset.target
