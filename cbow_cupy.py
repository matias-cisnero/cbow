import numpy as np
import cupy as cp

with open("corpus.txt", "r", encoding="utf-8") as f:
    corpus = f.read().splitlines()

# Vocabulario
vocab = sorted(set(corpus))
vocab_size = len(vocab)

# Diccionarios
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in enumerate(vocab)}

def softmax(u):
    u_max = cp.max(u)# estabiliza restando el máximo
    exp_u = cp.exp(u - u_max)
    return exp_u / cp.sum(exp_u)

def generar_pares_central_contexto(corpus, word_to_idx, C=4):

    pares = []

    for i in range(C, len(corpus)-C):
        # Palabra objetivo
        palabra_central = corpus[i]
        palabra_central_indice = word_to_idx[palabra_central]

        # Palabras de entrada
        palabras_contexto = corpus[i-C:i] + corpus[i+1:i+C+1]
        palabras_contexto_indices = [word_to_idx[word] for word in palabras_contexto]

        pares.append([palabra_central_indice, palabras_contexto_indices])
    return pares

def entrenar_cbow(corpus, vocab_size, word_to_idx, nombre_pc, epocas=1, η=0.001, N=300, C=4, W1=None, W2=None, intervalo_guardado=50):
    if W1 is None or W2 is None:
        W1 = cp.random.normal(0, 0.1, (vocab_size, N))
        W2 = cp.random.normal(0, 0.1, (N, vocab_size))
    else:
        W1 = cp.asarray(W1)
        W2 = cp.asarray(W2)

    indice_tuplas = generar_pares_central_contexto(corpus, word_to_idx, C)
    total_pares = len(indice_tuplas)

    print(f"Comienzo de entrenamiento con {epocas} epocas.")
    for epoca in range(epocas):
        for i, (i_central, i_contextos) in enumerate(indice_tuplas):

            # ---Propagación---

            h = cp.mean(W1[i_contextos], axis=0).reshape(-1, 1)

            u = W2.T @ h

            y = softmax(u)

            # ---Retropropagación---

            E = -u[i_central] + cp.log(cp.sum(cp.exp(u)))

            e = y.copy()
            e[i_central] -= 1

            W2 -= η * (h @ e.T)

            EH = W2 @ e

            W1[i_contextos] -= η * (1/C) * EH.T

            if i % 1000 == 0:
                print(f"Época {epoca}, Par: {i}/{total_pares}, Error: {E.item():.4f}")

        print(f"Fin de época: {epoca}")

        # ---Guardado de Pesos---
        if epoca % intervalo_guardado == 0 or epoca == epocas - 1:
            nombre_archivo = f'pesos_cbow_{nombre_pc}_epoca{epoca}.npz'
            W1_np = cp.asnumpy(W1)
            W2_np = cp.asnumpy(W2)
            np.savez(nombre_archivo, W1=W1_np, W2=W2_np, eta=η, N=N, C=C)
            print(f"Pesos e hiperparámetros guardados exitosamente en '{nombre_archivo}'")

    print(f"Entrenamiento con {epocas} terminado.")
    return W1, W2

W1, W2 = entrenar_cbow(corpus, vocab_size, word_to_idx, "pc0", epocas=100, η=0.01, N=20, C=4, intervalo_guardado=50)