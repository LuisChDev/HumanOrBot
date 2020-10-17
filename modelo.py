"""este script toma los datos preprocesados del dataset y los utiliza para
entrenar el modelo predictivo.

Para entrenar este modelo, empleamos un método de ensamble que incluye

Autor: Luis Chavarriaga
"""

from preprocessing import datarray, datarget
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import pickle

print("entrenando modelo")

pipeline = Pipeline([
    ('count', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('rfc', RandomForestClassifier())
])



if __name__ == "__main__":
    print("cargando el dataset.")
    with open('tweets.pkl', 'rb') as f:
        tweets = pickle.load(f)
