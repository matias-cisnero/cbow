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

def softmax(h):
    num = cp.exp(h)
    return num / cp.sum(num)

def entrenar_cbow(corpus, vocab_size, word_to_idx, nombre_pc, epocas=1, η=0.001, N=300, C=4, W1=None, W2=None):
    if W1 == None or W2 == None:
        W1 = cp.random.normal(0, 0.1, (vocab_size, N))
        W2 = cp.random.normal(0, 0.1, (N, vocab_size))
    else:
        W1 = cp.asarray(W1)
        W2 = cp.asarray(W2)

    print(f"Comienzo de entrenamiento con {epocas} epocas.")
    for epoca in range(epocas):
        for i in range(C, len(corpus)-C):

            # Palabra objetivo
            palabra_central = corpus[i]
            palabra_central_indice = word_to_idx[palabra_central]

            # Palabras de entrada
            palabras_contexto = corpus[i-C:i] + corpus[i+1:i+C+1]
            palabras_contexto_indices = [word_to_idx[word] for word in palabras_contexto]

            # ---Propagación---

            h = cp.mean(W1[palabras_contexto_indices], axis=0).reshape(-1, 1)

            u = W2.T @ h

            y = softmax(u)

            # ---Retropropagación---

            E = -u[palabra_central_indice] + cp.log(cp.sum(cp.exp(u)))

            e = y.copy()
            e[palabra_central_indice] -= 1

            W2 -= η * (h @ e.T)

            EH = W2 @ e

            W1[palabras_contexto_indices] -= η * (1/C) * EH.T

            if i % 1000 == 0:
                print(f"Época {epoca}, Palabra: {i}/{len(corpus)}, Error: {E.item():.4f}")

        print(f"Fin de época: {epoca}")

        # ---Guardado de Pesos---
        if epoca % 50 == 0:
            nombre_archivo = f'pesos_cbow_{nombre_pc}_epoca{epoca}.npz'
            W1_np = cp.asnumpy(W1)
            W2_np = cp.asnumpy(W2)
            np.savez(nombre_archivo, W1=W1_np, W2=W2_np, eta=η, N=N, C=C)
            print(f"Pesos e hiperparámetros guardados exitosamente en '{nombre_archivo}'")

    print(f"Entrenamiento con {epocas} terminado.")
    return W1, W2

def cargar_modelo_completo(nombre_archivo='pesos_cbow_pc2_epoca0.npz'):
    """
    Carga los pesos W1, W2 y los hiperparámetros N, C y eta 
    desde un archivo .npz.
    """
    try:
        data = np.load(nombre_archivo)
        
        W1 = data['W1']
        W2 = data['W2']
    
        N = data['N'].item()
        C = data['C'].item()
        eta = data['eta'].item()
        
        print()

        return W1, W2, N, C, eta
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{nombre_archivo}'.")
        return None, None, None, None, None

W1, W2 = entrenar_cbow(corpus, vocab_size, word_to_idx, "pc2", 500, 0.01, 100)

#W1, W2, N, C, eta = cargar_modelo_completo()
#print(f"Shape de W: {W1.shape}, Shape de W':{W2.shape}, Tipo: {type(W1)}")