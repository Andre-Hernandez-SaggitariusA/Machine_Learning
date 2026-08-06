import Kit_Redes_Neuronales as krn
import numpy as np
import os
import time
import matplotlib.pyplot as plt

# Ejecutar codigo desde carpeta Number_Lector como:
# PYTHONPATH=.. python training.py

problemas = []
soluciones = []

with open("datos.txt", "r", encoding="utf-8") as archivo:
	
	lineas = [linea.strip() for linea in archivo if linea.strip()]

for i in range(0, len(lineas), 2):
	
	linea_problema = lineas[i]
	linea_solucion = lineas[i+1]
	
	problema_temporal = []
	solucion_temporal = []
	
	for n in linea_problema:
		if n == ".":
			problema_temporal.append(0)
		else:
			problema_temporal.append(int(n)/9)
	
	for n in linea_solucion:
		solucion_temporal.append(int(n)/9)
		
	problemas.append(problema_temporal)
	soluciones.append(solucion_temporal)

problemas = np.array(problemas)
soluciones = np.array(soluciones)

# Generador Random

rng = np.random.default_rng(seed=1234)

# Numero de Sudokus

numero_sudoku = len(problemas)
print(numero_sudoku)

# Numero de neuronas por capa

entrada = 81
neuronas_1 = 512
neuronas_2 = 256
neuronas_3 = 128
salida = 81

# Costo Total Inicial

lista_costos = []

# Inicializar Pesos y Bias (Depende de si existe entrenamiento previo o no)

if os.path.exists("valores_entrenamiento.npz"):
	
	datos = np.load("valores_entrenamiento.npz")

	W1 = datos["W1"]
	W2 = datos["W2"]
	W3 = datos["W3"]
	W4 = datos["W4"]

	b1 = datos["b1"]
	b2 = datos["b2"]
	b3 = datos["b3"]
	b4 = datos["b4"]

else:

	W1, b1 = krn.inicializador_he_uniforme(entrada, neuronas_1)

	W2, b2 = krn.inicializador_he_uniforme(neuronas_1, neuronas_2)
	
	W3, b3 = krn.inicializador_he_uniforme(neuronas_2, neuronas_3)

	W4, b4 = krn.inicializador_xavier_uniforme(neuronas_3, salida)
			
# Definir epocas

epocas = 5000

# Definir Mini_Batch

batch_size = 64

# Valor de aprendizaje

aprendizaje_base = 0.001
aprendizaje = aprendizaje_base * np.sqrt(batch_size)

# Tiempo Inicial

t0 = time.perf_counter()

for epoca in range(epocas):

	t0_epoca = time.perf_counter()

	costo_total = 0

	indices = rng.permutation(numero_sudoku)

	for i in range(0, numero_sudoku, batch_size):
		
		indices_batch = indices[i:i + batch_size]
		
		# Obtener matriz
	
		problema_batch = problemas[indices_batch].T
		
		# Tamaño del mini-batch
		
		m = problema_batch.shape[1]
		
		# Matriz Esperada
		
		solucion_batch = soluciones[indices_batch].T
		
		# Forward Propagation
		
		# Primera capa
	
		A1 = krn.forward_leaky_relu_mlp(W1, b1, problema_batch)

		# Segunda capa	
		
		A2 = krn.forward_leaky_relu_mlp(W2, b2, A1)
		
		# Tercera capa	
		
		A3 = krn.forward_leaky_relu_mlp(W3, b3, A2)

		# Tercera capa

		A4 = krn.forward_sig_mlp(W4, b4, A3)
	
		# Calcular costo (Cross Entropy)
		
		error = A4 - solucion_batch
		
		costo_batch = np.sum(error ** 2)
		costo_total += costo_batch
	
		# Gradiente cuarta capa
	
		D4, G4, B4 = krn.back_sig_last(A4, solucion_batch, A3)
		
		# Gradiente tercera capa
	
		D3, G3, B3 = krn.back_leaky_relu_mlp(W4, D4, A3, A2)
		
		# Gradiente segunda capa
	
		D2, G2, B2 = krn.back_leaky_relu_mlp(W3, D3, A2, A1)
	
		# Gradiente tercera capa
	
		D1, G1, B1 = krn.back_leaky_relu_mlp(W2, D2, A1, problema_batch)
			
		# Mejorar Pesos
	
		W1 -= aprendizaje * G1
		W2 -= aprendizaje * G2
		W3 -= aprendizaje * G3
		W4 -= aprendizaje * G4
	
		# Mejorar Bias
	
		b1 -= aprendizaje * B1
		b2 -= aprendizaje * B2
		b3 -= aprendizaje * B3
		b4 -= aprendizaje * B4

	lista_costos.append(costo_total / numero_sudoku)
	
	tf_epoca = time.perf_counter()
	
	print("Epoca: ", epoca + 1)
	print("Costo: ", costo_total / numero_sudoku)
	print("Tiempo Epoca: ", tf_epoca-t0_epoca)

# Tiempo Final

tf = time.perf_counter()

print("Tiempo: ", tf-t0)
	
plt.plot(lista_costos)
plt.yscale("log")
plt.grid(True)
plt.show()

np.savez("valores_entrenamiento.npz", W1=W1, W2=W2, W3=W3, W4=W4, b1=b1, b2=b2, b3=b3, b4=b4)
		
