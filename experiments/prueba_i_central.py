# Simulamos los datos
word_to_idx = {'gato': 42}
corpus = ['el', 'perro', 'gato', 'corre']
i = 2 # La posición de 'gato'

# Esta parte es solo el número
numero_indice = word_to_idx[corpus[i]]

# Esta parte es la que está en tu código, con los corchetes
lista_indice = [word_to_idx[corpus[i]]]


print(f"El índice solo: {numero_indice}")
print(f" -> Tipo de dato: {type(numero_indice)}")

print("-" * 30)

print(f"El índice dentro de corchetes: {lista_indice}")
print(f" -> Tipo de dato: {type(lista_indice)}")