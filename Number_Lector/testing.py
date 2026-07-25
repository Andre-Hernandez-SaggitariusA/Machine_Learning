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

if ruta == "testing/test_1.csv":

	imagenes = df.drop(columns = "label").to_numpy() / 255
	
else:

	imagenes = df.drop(columns = "label").to_numpy()
	
def obtener_imagen(num_fila):
	
	imagen = imagenes[num_fila]
	valor = valores[num_fila]
	
	return imagen, valor

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

for j in range(num_im):

	# Obtener matriz
		
	imagen, valor = obtener_imagen(j)
	
	# Primera capa
	
	primera_capa = krn.forward_leaky_relu(W1, b1, imagen)

	# Segunda capa	
	
	segunda_capa = krn.forward_leaky_relu(W2, b2, primera_capa)

	# Tercera capa
	
	tercera_capa = krn.forward_softmax(W3, b3, segunda_capa)
		
	# Calcular valor
		
	prediccion = np.argmax(tercera_capa)
	print(prediccion)
	
	correcto.append(valor)
	calculado.append(prediccion)
	
	# Mostrar Imagen

	#plt.imshow(matriz.reshape(28, 28), cmap="gray")
	#plt.show()
	
correcto = np.array(correcto)
calculado = np.array(calculado)
	
errores = correcto - calculado

errores = [dif for dif in errores if dif != 0]

porcentaje = len(errores) / len(correcto)

print(porcentaje*100)	
