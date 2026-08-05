import numpy as np

def codificar_linea_sudoku(linea):
    # Crear matriz de 81 casillas por 9 posibles números (1-9)
    one_hot = np.zeros((81, 9), dtype=np.int8)
    
    for i, char in enumerate(linea.strip()):
        if char != '.':
            digito = int(char)
            # El número 1 activa el índice 0, el 9 activa el índice 8
            one_hot[i, digito - 1] = 1
            
    return one_hot

# 1. Leer el archivo TXT y procesar las líneas válidas
matrices_procesadas = []

with open("datos.txt", "r", encoding="utf-8") as archivo:
    for linea in archivo:
        linea_limpia = linea.strip()
        # Procesar solo líneas que tengan la longitud de un Sudoku (81 caracteres)
        if len(linea_limpia) == 81:
            matriz_one_hot = codificar_linea_sudoku(linea_limpia)
            matrices_procesadas.append(matriz_one_hot)

# 2. Convertir la lista en un array tridimensional de NumPy
# Dimensiones resultantes: (Número_de_sudokus, 81_casillas, 9_canales)
dataset_one_hot = np.array(matrices_procesadas)

print(f"Procesados con éxito {len(dataset_one_hot)} sudokus.")
print("Forma final del tensor:", dataset_one_hot.shape)

