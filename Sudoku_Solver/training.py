import Kit_Redes_Neuronales as krn
import numpy as np

# Ejecutar codigo desde carpeta Number_Lector como:
# PYTHONPATH=.. python training.py

problemas = []
soluciones = []
lista_zeros = [0] * 9

with open("datos.txt", "r", encoding="utf-8") as archivo:
	
	lineas = [linea.strip() for linea in archivo if linea.strip()]

for i in range(0, len(lineas), 2):
	
	linea_problema = lineas[i]
	linea_solucion = lineas[i+1]
			
	problema = [0 if n == "." else int(n) for n in linea_problema]
	solucion = [int(n) for n in linea_solucion]
	
	for numero in problema:
	
		if numero == 0:
		
					
		
	
	problema = [lista_zeros if n == 0 else lista_zeros[n-1] = 1 for n in problema]
	
	print(problema)
	print(solucion)	
			
			
			
