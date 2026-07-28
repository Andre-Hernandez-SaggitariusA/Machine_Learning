import numpy as np
import pandas as pd

datos = np.load("valores_entrenamiento.npz")

W1 = datos["W1"]
W2 = datos["W2"]
Wt = datos["Wt"]

b1 = datos["b1"]
b2 = datos["b2"]

def sig(x):

	return 1 / (1 + np.exp(-x))

def der_sig(x):

	return x * (1 - x)
	
df = pd.read_csv("dataset_series_test.csv")
X = df[["1", "2", "3"]].values
y = df["Etiqueta"].values

aciertos = 0
fallos = 0

for j in range(len(X)):
		
	a1 = 0
		
	for i in range(len(X[j])):
			
		z1 = W1 * X[j][i] + Wt * a1 + b1
		a1 = sig(z1)
			
	z2 = W2 * a1 + b2
	a2 = sig(z2)
		
	if y[j] == 1:
		if a2 >= 0.5:
			aciertos += 1
		else:
			fallos += 1
	else:
		if a2 <= 0.5:
			aciertos += 1
		else:
			fallos += 1
			
print("Porcentaje de Acierto: ", 100 * aciertos / len(X)) 
print("Porcentaje de Fallo: ", 100 * fallos / len(X))

