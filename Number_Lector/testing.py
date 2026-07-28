import Kit_Redes_Neuronales as krn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ejecutar codigo desde carpeta Number_Lector como:
# PYTHONPATH=.. python testing.py

rng = np.random.default_rng(seed=1234)

# Leer CSV de entrenamiento

ruta = "testing/test_2.csv"

df = pd.read_csv(ruta)

valores = df["label"].to_numpy()

imagenes = df.drop(columns = "label").to_numpy(dtype=np.float64)

if imagenes.max() > 1:

	imagenes /= 255

# Pesos y bias guardados

datos = np.load("valores_entrenamiento.npz")

W1 = datos["W1"]
W2 = datos["W2"]
W3 = datos["W3"]

b1 = datos["b1"]
b2 = datos["b2"]
b3 = datos["b3"]

# Comprobacion

calculado = []
correcto = []

# Numero de imagenes

num_im = len(imagenes)

# Definir Mini_Batch

batch_size = 32

# Indices

indices = rng.permutation(num_im)

for i in range(0, num_im, batch_size):

	indices_batch = indices[i:i + batch_size]
	
	# Obtener matriz
	
	imagenes_batch = imagenes[indices_batch].T
	valores_batch = valores[indices_batch]
	
	# Forward Propagation
	
	# Primera capa
	
	A1 = krn.forward_leaky_relu(W1, b1, imagenes_batch)

	# Segunda capa	
	
	A2 = krn.forward_leaky_relu(W2, b2, A1)

	# Tercera capa
	
	A3 = krn.forward_softmax(W3, b3, A2)
		
	# Calcular valor
		
	prediccion = np.argmax(A3, axis=0)
	
	print(prediccion)
	
	correcto.extend(valores_batch)
	calculado.extend(prediccion)
	
	# Mostrar Imagen

	#plt.imshow(matriz.reshape(28, 28), cmap="gray")
	#plt.show()
	
correcto = np.array(correcto)
calculado = np.array(calculado)
	
errores = np.sum(correcto != calculado)

porcentaje_error = 100 * errores / num_im

print("Porcentaje Error: ", porcentaje_error)	
