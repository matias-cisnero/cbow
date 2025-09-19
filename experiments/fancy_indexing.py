import cupy as cp

# Creamos una matriz W2 de prueba para que sea fácil ver el resultado.
# La primera fila es el índice de la columna, la segunda es el índice + 100.
W2 = cp.array([
    [ 0,  1,  2,  3,  4],  # Columna 0, 1, 2, 3, 4
    [100, 101, 102, 103, 104]
])

# Definimos una lista de índices en un orden específico (no ordenado).
i_total = [3, 0, 4]

print("--- Matriz W2 Original ---")
print(W2)
print("\nÍndices a seleccionar (i_total):", i_total)
print("-" * 30)

# Realizamos la selección de columnas
columnas_seleccionadas = W2[:, i_total]

print("--- Columnas Seleccionadas ---")
print(columnas_seleccionadas)