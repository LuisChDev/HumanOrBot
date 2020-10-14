"""este script toma los datos preprocesados del dataset y los utiliza para
entrenar el modelo predictivo.

Autor: Luis Chavarriaga
"""

from preprocessing import datarray, datarget
from sklearn.ensemble import RandomForestClassifier

print("entrenando modelo")

clf = RandomForestClassifier(random_state=0)
clf.fit(datarray, datarget)

print(clf.predict([[6, 3, 5, 2], [4, 3, 1.4, 0.3]]))
