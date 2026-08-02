from .F_Activacion import *
import numpy as np

# Propagacion Sigmoide

def forward_sig_mlp(W_current, b_current, A_previous):

	Z_current = (W_current @ A_previous) + b_current
	
	A_current = sig(Z_current)
	
	return A_current

def back_sig_mlp(W_posterior, D_posterior, A_current, A_previous):

	batch_size = A_previous.shape[1]

	D_current = (W_posterior.T @ D_posterior) * der_sig(A_current)
	
	G_current = (D_current @ A_previous.T) / batch_size
	
	B_current = np.sum(D_current, axis=1, keepdims=True) / batch_size
	
	return D_current, G_current, B_current

def forward_sig_rnn(W_current, U_current, b_current, A_previous, A_past):

	Z_current = (W_current @ A_previous) + (U_current @ A_past) + b_current
	
	A_current = sig(Z_current)
	
	return A_current
	
# Propagacion Leaky ReLu

def forward_leaky_relu_mlp(W_current, b_current, A_previous):
	
	Z_current = (W_current @ A_previous) + b_current
	
	A_current = leaky_relu(Z_current)
	
	return A_current
	
def back_leaky_relu_mlp(W_posterior, D_posterior, A_current, A_previous):

	batch_size = A_previous.shape[1]

	D_current = (W_posterior.T @ D_posterior) * der_leaky_relu(A_current)
	
	G_current = (D_current @ A_previous.T) / batch_size
	
	B_current = np.sum(D_current, axis=1, keepdims=True) / batch_size
	
	return D_current, G_current, B_current
	
# Propagacion Tanh
	
def forward_tanh_rnn(W_current, U_current, b_current, A_previous, A_past):

	Z_current = (W_current @ A_previous) + (U_current @ A_past) + b_current
	
	A_current = tanh(Z_current)
	
	return A_current
	
def back_tanh_rnn(W_posterior, U_current, D_posterior, D_future, A_current, A_previous, A_past):

	batch_size = A_previous.shape[1]
	
	D_current = (W_posterior.T @ D_posterior + U_current.T @ D_future) * der_tanh(A_current)
	
	G_W_current = (D_current @ A_previous.T) / batch_size
	
	G_U_current = (D_current @ A_past.T) / batch_size
	
	G_B_current = np.sum(D_current, axis=1, keepdims=True) / batch_size
	
	return D_current, G_W_current, G_U_current, G_B_current

# Propagacion Softmax

def forward_softmax_mlp(W_current, b_current, A_previous):

	Z_current = (W_current @ A_previous) + b_current
	
	A_current = softmax(Z_current)
	
	return A_current
	
def back_softmax_mlp(A_current, A_expected, A_previous):
	
	batch_size = A_previous.shape[1]
	
	D_current = A_current - A_expected
	
	G_current = (D_current @ A_previous.T) / batch_size
	
	B_current = np.sum(D_current, axis=1, keepdims=True) / batch_size
	
	return D_current, G_current, B_current
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
