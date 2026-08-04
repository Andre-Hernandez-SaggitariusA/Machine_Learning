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

ruta = "training/train_2.csv"

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

pixeles = int(np.sqrt(len(imagenes[0])))

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

	W1, b1 = krn.inicializador_xavier_uniforme(pixeles, neuronas_1)
	
	U1, _ = krn.inicializador_xavier_uniforme(neuronas_1, neuronas_1)

	W2, b2 = krn.inicializador_xavier_uniforme(neuronas_1, neuronas_2)

	U2, _ = krn.inicializador_xavier_uniforme(neuronas_2, neuronas_2)

	W3, b3 = krn.inicializador_xavier_uniforme(neuronas_2, salida)

# Definir epocas

epocas = 30

# Numero de imagenes

num_im = len(imagenes)

# Definir Mini_Batch

batch_size = 1

# Valor de aprendizaje

aprendizaje_base = 0.001
aprendizaje = aprendizaje_base * np.sqrt(batch_size)

# Tiempo Inicial

t0 = time.perf_counter()

for epoca in range(epocas):

	t0_epoca = time.perf_counter()

	aciertos = 0
	fallos = 0
	costo_total = 0

	indices = rng.permutation(num_im)

	for indice in indices:
		
		# Obtener matriz
	
		imagen = imagenes[indice]
		imagen = imagen.reshape(28, 28)
		valor = valores[indice]
		
		# Matriz Esperada
		
		Y = np.zeros(salida)
		Y[valor] = 1.0
		
		# Matrices de Secuencias
		
		A1_past = np.zeros(neuronas_1)
		A2_past = np.zeros(neuronas_2)
		
		A1_past_sequence = []
		A2_past_sequence = []
		
		A1_sequence = []
		A2_sequence = []
		
		for i in range(len(imagen)):
			
			A1_past_sequence.append(A1_past)
			A2_past_sequence.append(A2_past)
			
			# Forward Propagation
		
			# Primera capa
	
			A1 = krn.forward_tanh_rnn(W1, U1, b1, imagen[i], A1_past)

			# Segunda capa	
			
			A2 = krn.forward_tanh_rnn(W2, U2, b2, A1, A2_past)
			
			A1_sequence.append(A1)
			A2_sequence.append(A2)
			
			A1_past = A1
			A2_past = A2

		# Tercera capa

		A3 = krn.forward_softmax_mlp(W3, b3, A2_sequence[-1])
	
		# Calcular costo (Cross Entropy)
				
		costo_batch = -np.sum(Y * np.log(A3))
		costo_total += costo_batch
	
		# Gradiente tercera capa
	
		D3, G3 = krn.back_softmax_mlp(A3, Y, A2_sequence[-1])
	
		G1 = 0
		G2 = 0
		
		GU1 = 0
		GU2 = 0
		
		D1 = 0
		D2 = 0
		
		D1_future = np.zeros(neuronas_1)
		D2_future = np.zeros(neuronas_2)
		D3_zero = np.zeros_like(D3)
		
		for i in reversed(range(len(imagen))):
		
			if i != (len(imagen) - 1):
				
				D3_posterior = D3_zero
				
			else:
				
				D3_posterior = D3.copy()
				
			# Gradiente segunda capa
		
			D2_t, G2_t, GU2_t = krn.back_tanh_rnn(W3, U2, D3_posterior, D2_future, A2_sequence[i], A1_sequence[i], A2_past_sequence[i])
	
			# Gradiente tercera capa
	
			D1_t, G1_t, GU1_t = krn.back_tanh_rnn(W2, U1, D2_t, D1_future, A1_sequence[i], imagen[i], A1_past_sequence[i])
			
			G1 += G1_t
			G2 += G2_t
			
			GU1 += GU1_t
			GU2 += GU2_t
			
			D1 += D1_t
			D2 += D2_t
			
			D1_future = D1_t
			D2_future = D2_t
			
		# Mejorar Pesos
	
		W1 -= aprendizaje * G1
		W2 -= aprendizaje * G2
		W3 -= aprendizaje * G3
	
		# Mejorar Pesos Recurrentes
		
		U1 -= aprendizaje * GU1
		U2 -= aprendizaje * GU2
	
		# Mejorar Bias
	
		b1 -= aprendizaje * D1
		b2 -= aprendizaje * D2
		b3 -= aprendizaje * D3
	
		# Calcular Predicciones
		
		predicciones = np.argmax(A3, axis=0)
		
		aciertos_batch = np.sum(predicciones == valor)
		
		aciertos += int(aciertos_batch)
		fallos += int(1 - aciertos_batch)

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



























