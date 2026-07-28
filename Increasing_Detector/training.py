import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sig(x):

	return 1 / (1 + np.exp(-x))

def der_sig(x):

	return x * (1 - x)

rng = np.random.default_rng(seed=1234)

df = pd.read_csv("dataset_series.csv")
X = df[["1", "2", "3"]].values
y = df["Etiqueta"].values

W1 = rng.uniform(-0.1, 0.1)
b1 = 0

W2 = rng.uniform(-0.1, 0.1)
b2 = 0

Wt = rng.uniform(-0.1, 0.1)

aprendizaje = 0.001

epocas = 25

lista_costo = []
lista_acierto = []
lista_fallo = []

for epoca in range(epocas):

	costo_total = 0
	aciertos = 0
	fallos = 0
	
	for j in range(len(X)):
		
		a1 = 0
		s_t = 0
		s_b = 0
		s_w = 0
		
		for i in range(len(X[j])):
			
			z1 = W1 * X[j][i] + Wt * a1 + b1
			a1 = sig(z1)
			
			s_t = a1 + Wt * der_sig(a1) * s_t
			s_b = Wt * der_sig(a1) * s_b + 1
			s_w = Wt * der_sig(a1) * s_w + X[j][i]
			
		z2 = W2 * a1 + b2
		a2 = sig(z2)
		
		costo = (1 / 2) * ((a2 - y[j])**2)
		costo_total += costo
		
		D2 = (a2 - y[j]) * der_sig(a2)
		G2 = D2 * a1
		
		comun = D2 * W2 * der_sig(a1)
		
		D1 = comun * s_b
		G1 = comun * s_w
		Gt = comun * s_t
		
		W1 -= aprendizaje * G1
		b1 -= aprendizaje * D1
		
		W2 -= aprendizaje * G2
		b2 -= aprendizaje * D2
		
		Wt -= aprendizaje * Gt
		
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
		
	lista_costo.append(costo_total / len(X))
	lista_acierto.append(aciertos)
	lista_fallo.append(fallos)
	
	print("Epoca: ", epoca + 1)

plt.plot(lista_costo)
plt.grid(True)
plt.show()

plt.plot(lista_acierto, label="Aciertos", c="green")
plt.plot(lista_fallo, label="Fallos", c="red")
plt.grid(True)
plt.show()
			
np.savez("valores_entrenamiento.npz", W1=W1, W2=W2, Wt=Wt, b1=b1, b2=b2)
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
