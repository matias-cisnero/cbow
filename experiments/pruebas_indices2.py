import cupy as cp

mi_vector = cp.zeros(5)

# Creamos una lista de índices donde uno es válido (2) y otro es inválido (99).
indices_a_modificar = [2, 99]

print("Vector original:", mi_vector)

# Intentamos restar 1 en las posiciones 2 y 99
mi_vector[indices_a_modificar] -= 1

print("Vector modificado:", mi_vector)