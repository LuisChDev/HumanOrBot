"""este script toma los datos preprocesados del dataset y los utiliza para
entrenar el modelo predictivo.

Para entrenar este modelo, empleamos un método de ensamble que incluye

Autor: Luis Chavarriaga
"""

# from preprocessing import datarray, datarget
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import pickle

# datos preprocesados
with open('train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('dev_data.pkl', 'rb') as f:
    dev_data = pickle.load(f)

print("adaptando los datos")

X_train = pd.DataFrame(train_data, index=[x['author'] for x in train_data])
X_test = pd.DataFrame(dev_data, index=[x['author'] for x in dev_data])

Y_train = X_train['botvalue']
Y_test = X_test['botvalue']

X_train = X_train.drop('author', axis=1).drop('botvalue', axis=1)
X_test = X_test.drop('author', axis=1).drop('botvalue', axis=1)

breakpoint()

print("entrenando modelo")

# la tubería consiste en los siguientes pasos:
# 1. se normalizan los valores respecto a su propio promedio,
# 2. se usan los datos para ajustar un modelo de bosque aleatorio.
pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('rfc', RandomForestClassifier())
])

# Los hiperparámetros del clasificador que serán ajustados con los datos
# mediante cross validation.
hyperparameters = {
    'rfc__max_depth': [None, 6, 4, 2, 1],
    'rfc__max_features': [0.5, 'sqrt', 'log2']
}

# se realiza la búsqueda usando el pipeline con las transformaciones.
clsf = GridSearchCV(pipeline, hyperparameters, cv=10)

# creamos un segundo modelo, esta vez con un clasificador diferente.
pipeline2 = Pipeline([
    ('scale2', StandardScaler()),
    ('gbc', GradientBoostingClassifier())
])

# nuevamente, definimos el modelo incluyendo una etapad de búsqueda de
# hiperparámetros.
hyper2 = {
    'gbc__max_features': [None, 'sqrt', 'log2'],
    'gbc__max_depth': [5, 3, 1]
}
clsf2 = GridSearchCV(pipeline2, hyper2, cv=10)


# podemos entrenar este modelo para hallar los parámetros y los hiperparámetros
clsf.fit(X_train, Y_train)
clsf2.fit(X_train, Y_train)

print("mejores parámetros:", clsf.best_params_)

Y_pred = clsf.predict(X_test)

print('puntaje R^2:', r2_score(Y_test, Y_pred))
print('Error cuadrático medio:', mean_squared_error(Y_test, Y_pred))
print('predicciones:', Y_pred[:5])

print("resultados del gradient boosting:")

Y_pred2 = clsf2.predict(X_test)
print("puntaje R^2:", r2_score(Y_test, Y_pred2))
print("Error cuadrático medio", mean_squared_error(Y_test, Y_pred))

print("predicciones gradient boosting", Y_pred2[:5])
