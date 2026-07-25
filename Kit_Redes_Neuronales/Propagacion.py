from .F_Activacion import *
import numpy as np

# Propagacion Sigmoide

def forward_sig(W, b, valores):

	Z = np.dot(W, valores) + b
	
	capa = sig(Z)
	
	return capa

def back_sig(pesos_s, D_s, valores, anterior):

	D = np.dot(D_s, pesos_s) * der_sig(valores)
	
	G = np.outer(D, anterior)
	
	return D, G

# Propagacion Leaky ReLu

def forward_leaky_relu(W, b, valores):
	
	Z = np.dot(W, valores) + b
	
	capa = leaky_relu(Z)
	
	return capa
	
def back_leaky_relu(pesos_s, D_s, valores, anterior):

	D = np.dot(D_s, pesos_s) * der_leaky_relu(valores)
	
	G = np.outer(D, anterior)
	
	return D, G
	
# Propagacion Softmax

def forward_softmax(W, b, valores):

	Z = np.dot(W, valores) + b
	
	capa = softmax(Z)
	
	return capa
	
def back_softmax(valores, esperado, anterior):
	
	D = valores - esperado
	
	G = np.outer(D, anterior)
	
	return D, G
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
