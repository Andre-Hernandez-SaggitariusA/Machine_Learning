from .F_Activacion import *
import numpy as np

# Propagacion Sigmoide

def forward_sig(W_a, b_a, P):

	Z_a = (W_a @ P) + b_a
	
	A = sig(Z_a)
	
	return A

def back_sig(W_f, D_f, A, P):

	batch_size = P.shape[1]

	D_a = (W_f.T @ D_f) * der_sig(A)
	
	G_a = (D_a @ P.T) / batch_size
	
	B_a = np.sum(D_a, axis=1, keepdims=True) / batch_size
	
	return D_a, G_a, B_a

# Propagacion Leaky ReLu

def forward_leaky_relu(W_a, b_a, P):
	
	Z_a = (W_a @ P) + b_a
	
	A = leaky_relu(Z_a)
	
	return A
	
def back_leaky_relu(W_f, D_f, A, P):

	batch_size = P.shape[1]

	D_a = (W_f.T @ D_f) * der_leaky_relu(A)
	
	G_a = (D_a @ P.T) / batch_size
	
	B_a = np.sum(D_a, axis=1, keepdims=True) / batch_size
	
	return D_a, G_a, B_a
	
# Propagacion Softmax

def forward_softmax(W_a, b_a, P):

	Z_a = (W_a @ P) + b_a
	
	A = softmax(Z_a)
	
	return A
	
def back_softmax(A, Y, P):
	
	batch_size = P.shape[1]
	
	D_a = A - Y
	
	G_a = (D_a @ P.T) / batch_size
	
	B_a = np.sum(D_a, axis=1, keepdims=True) / batch_size
	
	return D_a, G_a, B_a
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
