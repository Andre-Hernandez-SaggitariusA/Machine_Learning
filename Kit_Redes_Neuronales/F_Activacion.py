import numpy as np

# Funcion Sigmoide

def sig(x):

	valor = 1 / (1 + np.exp(-x))
	
	return valor
	
def der_sig(s):
	
	valor = s * (1 - s)
	
	return valor
	
# Funcion Leaky_ReLu

def leaky_relu(x, alpha=0.01):
	
	valor = np.where(x > 0, x, alpha * x)
	
	return valor
	
def der_leaky_relu(s, alpha=0.01):
	
	valor = np.where(s > 0, 1, alpha)
	
	return valor

# Funcion Tanh

def tanh(x):
	
	valor = np.tanh(x)
	
	return valor
	
def der_tanh(s):
	
	valor = 1 - s**2
	
	return valor

# Funcion Softmax

def softmax(x):

	x = x - np.max(x, axis=0, keepdims=True)
	
	exponente = np.exp(x)
	
	probabilidad = exponente / np.sum(exponente, axis=0, keepdims=True)
	
	return probabilidad
