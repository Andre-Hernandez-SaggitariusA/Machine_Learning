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

ruta = "train.csv"

df = pd.read_csv(ruta)

valores = df["label"].to_numpy()

imagenes = df.drop(columns = "label").to_numpy(dtype=np.float64)

if imagenes.max() > 1:

	imagenes /= 255

# Numero de neuronas por capa

neuronas_1 = 128
neuronas_2 = 64
salida = 10

# Numero de aciertos y fallos

lista_aciertos = []
lista_fallos = []

# Costo Total Inicial

lista_costos = []

# Cantidad de Pixeles

pixeles = len(imagenes[0])

# Inicializar Pesos y Bias (Depende de si existe entrenamiento previo o no)

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

epocas = 1

# Numero de imagenes

num_im = int(len(imagenes) / 10)

# Definir Mini_Batch

batch_size = 1

# Valor de aprendizaje

aprendizaje_base = 0.001
aprendizaje = aprendizaje_base * np.sqrt(batch_size)

# Listas para animacion

W1_list = []
W2_list = []
W3_list = []

b1_list = []
b2_list = []
b3_list = []

A_list = []
A1_list = []
A2_list = []
A3_list = []

# Tiempo Inicial

t0 = time.perf_counter()

for epoca in range(epocas):

	t0_epoca = time.perf_counter()

	aciertos = 0
	fallos = 0
	costo_total = 0

	indices = rng.permutation(num_im)

	for i in range(0, num_im, batch_size):
		
		indices_batch = indices[i:i + batch_size]
		
		# Obtener matriz
	
		imagenes_batch = imagenes[indices_batch].T
		valores_batch = valores[indices_batch]
		
		# Tamaño del mini-batch
		
		m = imagenes_batch.shape[1]
		
		# Matriz Esperada
		
		Y = np.zeros((salida, m))
		Y[valores_batch, np.arange(m)] = 1
		
		# Forward Propagation
		
		# Primera capa
	
		A1 = krn.forward_leaky_relu_mlp(W1, b1, imagenes_batch)

		# Segunda capa	
		
		A2 = krn.forward_leaky_relu_mlp(W2, b2, A1)

		# Tercera capa

		A3 = krn.forward_softmax_mlp(W3, b3, A2)
	
		# Calcular costo (Cross Entropy)
				
		costo_batch = -np.sum(Y * np.log(A3))
		costo_total += costo_batch
	
		# Gradiente tercera capa
	
		D3, G3, B3 = krn.back_softmax_mlp(A3, Y, A2)
	
		# Gradiente segunda capa
	
		D2, G2, B2 = krn.back_leaky_relu_mlp(W3, D3, A2, A1)
	
		# Gradiente tercera capa
	
		D1, G1, B1 = krn.back_leaky_relu_mlp(W2, D2, A1, imagenes_batch)
			
		# Mejorar Pesos
	
		W1 -= aprendizaje * G1
		W2 -= aprendizaje * G2
		W3 -= aprendizaje * G3
	
		# Mejorar Bias
	
		b1 -= aprendizaje * B1
		b2 -= aprendizaje * B2
		b3 -= aprendizaje * B3
	
		# Calcular Predicciones
		
		predicciones = np.argmax(A3, axis=0)
		
		aciertos_batch = np.sum(predicciones == valores_batch)
		
		aciertos += int(aciertos_batch)
		fallos += int(m - aciertos_batch)
		
		# Agregar listas para animacion
		
		W1_list.append(W1)
		W2_list.append(W2)
		W3_list.append(W3)
		b1_list.append(b1)
		b2_list.append(b2)
		b3_list.append(b3)
		A_list.append(imagenes_batch)
		A1_list.append(A1)
		A2_list.append(A2)
		A3_list.append(A3)

	lista_costos.append(costo_total / num_im)
	lista_aciertos.append(aciertos)
	lista_fallos.append(fallos)
	
	tf_epoca = time.perf_counter()
	
	print("Epoca: ", epoca + 1)
	print("Costo: ", costo_total / num_im)
	print("Aciertos: ", aciertos)
	print("Fallos: ", fallos)
	print("Tiempo Epoca: ", tf_epoca-t0_epoca)
	
	
# Tiempo Final

tf = time.perf_counter()

print("Tiempo: ", tf-t0)
	
plt.plot(lista_costos)
plt.yscale("log")
plt.grid(True)
plt.show()

plt.plot(lista_aciertos, label="Aciertos", color= "g")
plt.plot(lista_fallos, label="Fallos", color="r")
plt.legend()
plt.grid(True)
plt.show()

np.savez("valores_entrenamiento.npz", W1=W1, W2=W2, W3=W3, b1=b1, b2=b2, b3=b3)












































