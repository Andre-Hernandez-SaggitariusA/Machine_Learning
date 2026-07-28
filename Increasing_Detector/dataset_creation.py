import random
import pandas as pd

def crear_lista():
	while True:
		
		numeros = random.sample(range(-100, 100), 3)
		
		if numeros[0] < numeros[1] < numeros[2]:
			return numeros, 1
		
		elif numeros[2] < numeros[1] < numeros[0]:
			return numeros, 0
			
datos = []

for n in range(500000):
	
	serie, etiqueta = crear_lista()
	datos.append({
		"1": serie[0],
		"2": serie[1],
		"3": serie[2],
		"Etiqueta": etiqueta
		})
		
df = pd.DataFrame(datos)
df.to_csv("dataset_series_test.csv", index=False)

