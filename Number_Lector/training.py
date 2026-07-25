import Kit_Redes_Neuronales as krn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

# Ejecutar codigo desde carpeta Number_Lector como:
# PYTHONPATH=.. python training.py

rng = np.random.default_rng(seed=1234)

# Leer CSV de entrenamiento

ruta = "training/train_1.csv"

df = pd.read_csv(ruta)

valores = df["label"].to_numpy()

if ruta == "training/train_1.csv":

	imagenes = df.drop(columns = "label").to_numpy() / 255
	
else:

	imagenes = df.drop(columns = "label").to_numpy()
	
def obtener_imagen(num_fila):
	
	imagen = imagenes[num_fila]
	valor = valores[num_fila]
	
	return imagen, valor

# Numero de neuronas por capa

neuronas_1 = 128
neuronas_2 = 64
salida = 10

# Numero de aciertos y fallos

lista_aciertos = []
lista_fallos = []

# Costo Total Inicial

costo_lista = []

# Valor de aprendizaje

aprendizaje = 0.001

# Cantidad de Pixeles

pixeles = len(imagenes[0])

# Pesos y bias al azar

if os.path.exists("valores_entrenamiento.npz"):
	
	datos = np.load("valores_entrenamiento.npz")

	W1 = datos["W1"]
	W2 = datos["W2"]
	W3 = datos["W3"]

	b1 = datos["b1"]
	b2 = datos["b2"]
	b3 = datos["b3"]

else:

	W1, b1 = krn.inicializador_he_uniforme(pixeles, neuronas_1)

	W2, b2 = krn.inicializador_he_uniforme(neuronas_1, neuronas_2)

	W3, b3 = krn.inicializador_xavier_uniforme(neuronas_2, salida)

# Definir epocas

epocas = 10

# Numero de imagenes

num_im = len(imagenes)

# Tiempo Inicial

t0 = time.perf_counter()

for epoca in range(epocas):

	aciertos = 0
	fallos = 0
	costo_total = 0

	indices = rng.permutation(num_im)

	for j in indices:
	
		# Obtener matriz
	
		imagen, valor = obtener_imagen(j)
		
		# Primera capa
	
		primera_capa = krn.forward_leaky_relu(W1, b1, imagen)

		# Segunda capa	
		
		segunda_capa = krn.forward_leaky_relu(W2, b2, primera_capa)

		# Tercera capa

		tercera_capa = krn.forward_softmax(W3, b3, segunda_capa)
	
		# Calcular costo

		esperado = np.zeros(salida)
	
		esperado[valor] = 1.0
	
		costo = -np.log(tercera_capa[valor])
		costo_total += costo
	
		# Gradiente tercera capa
	
		D3, G3 = krn.back_softmax(tercera_capa, esperado, segunda_capa)
	
		# Gradiente segunda capa
	
		D2, G2 = krn.back_leaky_relu(W3, D3, segunda_capa, primera_capa)
	
		# Gradiente tercera capa
	
		D1, G1 = krn.back_leaky_relu(W2, D2, primera_capa, imagen)
			
		# Mejorar Pesos
	
		W1 -= aprendizaje * G1
		W2 -= aprendizaje * G2
		W3 -= aprendizaje * G3
	
		# Mejorar Bias
	
		b1 -= aprendizaje * D1
		b2 -= aprendizaje * D2
		b3 -= aprendizaje * D3
	
		if np.argmax(tercera_capa) == valor:
		
			aciertos += 1
	
		else:
		
			fallos += 1

	costo_lista.append(costo_total/num_im)
	lista_aciertos.append(aciertos)
	lista_fallos.append(fallos)
	
	print("Epoca: ", epoca + 1)

# Tiempo Final

tf = time.perf_counter()

print("Tiempo: ", tf-t0)
	
plt.plot(costo_lista)
plt.grid(True)
plt.show()

plt.plot(lista_aciertos, label="Aciertos", color= "g")
plt.plot(lista_fallos, label="Fallos", color="r")
plt.legend()
plt.grid(True)
plt.show()

np.savez("valores_entrenamiento.npz", W1=W1, W2=W2, W3=W3, b1=b1, b2=b2, b3=b3)












































