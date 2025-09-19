import cupy as cp

EI = cp.zeros((18, 1))

i_positivos = [6175, 13987, 13725, 24470, 4554, 16036, 9238, 11895]

print(f"Intentando acceder a los índices {i_positivos} en un vector de tamaño {EI.shape}...")

try:
    EI[i_positivos] -= 1
    print("\n¡Sorprendentemente, la operación se ejecutó sin errores!")
    print(EI[i_positivos])
    print("Vector EI")
    print(EI)
except IndexError as e:
    print(f"\n¡ERROR! Se ha producido el error esperado:")
    print(f"-> {e}")