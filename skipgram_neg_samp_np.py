import numpy as np
import os
import time

with open("corpus/corpus.txt", "r", encoding="utf-8") as f:
    corpus = f.read().splitlines()

# Vocabulario
vocab = sorted(set(corpus))
vocab_size = len(vocab)

# Diccionarios
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in enumerate(vocab)}

# Tamaño de corpus: 310347
# Tamaño de vocabulario: 30283

def softmax(u):
    u_max = np.max(u) # Estabiliza restando el máximo
    exp_u = np.exp(u - u_max)
    return exp_u / np.sum(exp_u)

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

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

def entrenar_skipgram_neg_samp(corpus, vocab_size, word_to_idx, nombre_pc, epocas=1, η=0.001, N=300, C=4, K=5, W1=None, W2=None, intervalo_guardado=50):
    if W1 is None or W2 is None:
        W1 = np.random.normal(0, 0.1, (vocab_size, N))
        W2 = np.random.normal(0, 0.1, (N, vocab_size))

    indice_tuplas = generar_tuplas_central_contexto_negativos(corpus, word_to_idx, C, K)
    total_pares = len(indice_tuplas)

    print(f"Comienzo de entrenamiento con {epocas} epocas.")
    for epoca in range(epocas):
        for i, (i_central, i_positivos, i_negativos) in enumerate(indice_tuplas):
            i_total = i_positivos + i_negativos

            # ---Propagación---
            h = W1[i_central].reshape(-1, 1)

            u = W2[:, i_total].T @ h

            y = sigmoide(u)

            # ---Retropropagación---
            EI = y.copy()
            EI[:len(i_positivos)] -= 1

            W2[:, i_total] -= η * (h @ EI.T)

            EH = W2[:, i_total] @ EI

            W1[i_central] -= η * EH.T[0]

            if i % 1000 == 0:
                print(f"Época {epoca}, Par: {i}/{total_pares}")

        print(f"Fin de época: {epoca}")

        # ---Guardado de Pesos---
        if epoca % intervalo_guardado == 0 or epoca == epocas - 1:
            nombre_archivo = f'pesos_skipgram_{nombre_pc}_epoca{epoca}.npz'
            np.savez(nombre_archivo, W1=W1, W2=W2, eta=η, N=N, C=C)
            print(f"Pesos e hiperparámetros guardados exitosamente en '{nombre_archivo}'")

    print(f"Entrenamiento con {epocas} terminado.")
    return W1, W2

W1, W2 = entrenar_skipgram_neg_samp(corpus, vocab_size, word_to_idx, "pcmati_numpy", epocas=100, η=0.01, N=20, C=4, K=5, intervalo_guardado=50)