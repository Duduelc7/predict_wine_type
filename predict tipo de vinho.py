import pandas as pd
import os
arquivo = pd.read_csv("C:/Users/eduar/OneDrive/Documentos/wine_dataset.csv")

#print(arquivo)

arquivo['style'] = arquivo['style'].replace('red', 0 )

arquivo['style'] = arquivo["style"].replace("white", 1)

#print(arquivo)

y = arquivo['style']

x = arquivo.drop("style", axis = 1)

print(y)

print(x)

from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size= 0.3)

print(x_treino.shape)

print(y_treino.shape)

print(x_teste.shape)

print(y_teste.shape)

from sklearn.ensemble import ExtraTreesClassifier

modelo = ExtraTreesClassifier(n_estimators = 100)
modelo.fit(x_treino,y_treino)

resultado = modelo.score(x_teste, y_teste)

print(f"Desempenho: {resultado}")

print(y_teste[250:255])

print(x_teste[250:255])

previsoes = modelo.predict(x_teste[250:255])

print(previsoes)



