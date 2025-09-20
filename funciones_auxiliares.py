import numpy as np
import cupy as cp
import os

def cargar_corpus(ruta_archivo="corpus/corpus.txt"):

    with open(ruta_archivo, "r", encoding="utf-8") as f:
        corpus = f.read().splitlines()
    
    vocab = sorted(set(corpus))
    vocab_size = len(vocab)
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for i, word in enumerate(vocab)}

    print("Tamaño de corpus:", len(corpus))
    print("Tamaño de vocabulario:", vocab_size)

    return corpus, vocab, vocab_size, word_to_idx, idx_to_word

def inicializar_pesos(vocab_size, N, W1= None, W2=None, cparray=False):
    
    libreria = cp if cparray else np
    if W1 is None or W2 is None:
        W1 = libreria.random.normal(0, 0.1, (vocab_size, N))
        W2 = libreria.random.normal(0, 0.1, (N, vocab_size))
    elif cparray:
        W1 = cp.asarray(W1)
        W2 = cp.asarray(W2)
    
    return W1, W2

def softmax_np(u):
    u_max = np.max(u) # Estabiliza restando el máximo
    exp_u = np.exp(u - u_max)
    return exp_u / np.sum(exp_u)

def softmax_cp(u):
    u_max = cp.max(u) # Estabiliza restando el máximo
    exp_u = cp.exp(u - u_max)
    return exp_u / cp.sum(exp_u)

def sigmoide_np(x):
    return 1 / (1 + np.exp(-x))

def sigmoide_cp(x):
    return 1 / (1 + cp.exp(-x))

# Guardado y cargado de modelo

def guardar_modelo(nombre_archivo, W1, W2, eta, N, C, cparray=False):
    if cparray:
        W1 = cp.asnumpy(W1)
        W2 = cp.asnumpy(W2)
    os.makedirs("weights", exist_ok=True)
    ruta_completa = os.path.join("weights", nombre_archivo)

    np.savez(ruta_completa, W1=W1, W2=W2, eta=eta, N=N, C=C)
    print(f"Pesos e hiperparámetros guardados exitosamente en '{ruta_completa}'")

def cargar_modelo(nombre_archivo=str):
    try:
        ruta = os.path.join("weights", nombre_archivo)
        data = np.load(ruta)
        
        W1 = data['W1']
        W2 = data['W2']
    
        N = data['N'].item()
        C = data['C'].item()
        eta = data['eta'].item()

        return W1, W2, N, C, eta
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{nombre_archivo}'.")
        return None, None, None, None, None

# Funciones para el diccionario

def generar_tuplas_central_contexto(corpus, word_to_idx, C=4):
    tuplas = []
    for i in range(C, len(corpus)-C):
        # Palabra central
        palabra_central = corpus[i]
        palabra_central_indice = word_to_idx[palabra_central]

        # Palabras de contexto
        palabras_contexto = corpus[i-C:i] + corpus[i+1:i+C+1]
        palabras_contexto_indices = [word_to_idx[word] for word in palabras_contexto]

        tuplas.append([palabra_central_indice, palabras_contexto_indices])
    return tuplas

def generar_tuplas_central_contexto_negativos(corpus, word_to_idx, C=4, K=5):
    tuplas = []
    distancia_max = C + K
    for i in range(distancia_max, len(corpus) - distancia_max):
        # Índice de la palabra central
        indice_central = word_to_idx[corpus[i]]

        # Índices del contexto interno (POSITIVOS)
        indices_interno = [word_to_idx[w] for w in corpus[i - C : i] + corpus[i + 1 : i + C + 1]]
        
        # Índices del contexto externo (NEGATIVOS)
        indices_externo = [word_to_idx[w] for w in corpus[i - distancia_max : i - C] + corpus[i + C + 1 : i + distancia_max + 1]]
        
        tuplas.append((indice_central, indices_interno, indices_externo))
    return tuplas

# Interacción con el modelo

def ver_palabras_similares(corpus, word_to_idx, idx_to_word, palabra, W1, N=5):
    if palabra in corpus:
        i_palabra = word_to_idx[palabra]

        embedding_palabra = W1[i_palabra]

        productos = W1 @ embedding_palabra # (|V|, N) @ (N, 1)
        #print(f"Shape de embedding: {embedding_palabra.shape}, shape de W1: {W1.shape}, shape de productos: {productos.shape}")
        productos[i_palabra] = -np.inf  # para no recomendarse a sí misma
        indices = np.argpartition(productos, -N)[-N:]
        indices_ordenados = indices[np.argsort(productos[indices])[::-1]]
        similares = [idx_to_word[i] for i in indices_ordenados]

        print(f"Palabras similares a '{palabra}': {similares}")
    else:
        print(f"La palabra {palabra} no existe en el corpus")