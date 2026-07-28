import numpy as np

rng = np.random.default_rng(seed=1234)

# Generar Pesos y Sesgos

def inicializador_uniforme(cantidad_pesos, cantidad_sesgos):
	
	W = rng.uniform(-0.1, 0.1, size=(cantidad_sesgos, cantidad_pesos))
	b = np.zeros((cantidad_sesgos, 1))
	
	return W, b

def inicializador_he_uniforme(cantidad_pesos, cantidad_sesgos):
	
	limite = np.sqrt(6 / cantidad_pesos)
	
	W = rng.uniform(-limite, limite, size=(cantidad_sesgos, cantidad_pesos))
	b = np.zeros((cantidad_sesgos, 1))
	
	return W, b
	
def inicializador_xavier_uniforme(cantidad_pesos, cantidad_sesgos):
	
	limite = np.sqrt(6 / (cantidad_pesos + cantidad_sesgos))
	
	W = rng.uniform(-limite, limite, size=(cantidad_sesgos, cantidad_pesos))
	b = np.zeros((cantidad_sesgos, 1))
	
	return W, b
