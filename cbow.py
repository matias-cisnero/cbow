import numpy as np
import matplotlib.pyplot as plt

with open("corpus.txt", "r", encoding="utf-8") as f:
    corpus = f.read().splitlines()

# Vocabulario
vocab = sorted(set(corpus))
vocab_size = len(vocab)

# Diccionarios
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in enumerate(vocab)}

# Función: palabra -> one-hot
def one_hot_pos(word):
    idx = word_to_idx[word]
    return idx

def one_hot_encode(word):
    vector = np.zeros(vocab_size, dtype=int)
    idx = word_to_idx[word]
    vector[idx] = 1
    return vector.reshape(-1, 1)

# Función: one-hot -> palabra
def one_hot_decode(vector):
    idx = int(np.argmax(vector))   # posición del 1
    return idx_to_word[idx]

#print(one_hot_pos("AUTOPISTA"))

# Funciones auxiliares

def softmax(h):
   num = np.exp(h) 
   return num / np.sum(num)

def entrenar_cbow(epocas, η=0.001):
    card_V = vocab_size
    N = 300
    C = 4

    # x ∈ |V| x C <--- Entrada
    # W ∈ |V| x N
    # h ∈ N x 1 <--- Oculta (lineal)
    # W' ∈ N x |V|
    # y ∈ |V| x 1 <--- Salida (softmax)

    W1 = np.random.normal(0, 0.1, (card_V, N))
    W2 = np.random.normal(0, 0.1, (N, card_V))

    print(f"Comienzo de entrenamiento con {epocas} epocas.")
    for epoca in range(epocas):

        #for i in range(C, 8):
        for i in range(C, len(corpus)-C):
            
            palabra_central = corpus[i]
            contexto = corpus[i-C:i] + corpus[i+1:i+C+1]
            
            y = one_hot_encode(palabra_central)
            j_estrella = word_to_idx[palabra_central]
            contexto_indices = [word_to_idx[word] for word in contexto]

            #print("Central:", palabra_central, "-> Contexto indices:", contexto_indices)

            # ---Propagación---

            # Igual a 1/C de np.sum
            h = np.mean(W1[contexto_indices], axis=0).reshape(-1, 1)

            #print(f"shape de h: {h.shape}")
            
            # N x |V| @ N x 1 quiero igual a |V| x 1
            u = W2.T @ h
            ypred = softmax(u)

            #print(f"shape de ypred: {y.shape}")

            # ---Retropropagación---

            E = -u[j_estrella] + np.log(np.sum(np.exp(u)))

            e = ypred - y
            EH = W2 @ e

            #for j in range(card_V):
            #    W2[:, j] -= η * e[j] * h

            W2 -= η * (h @ e.T)

            W1[contexto_indices] -= η * (1/C) * EH.T
            print(f"Recorrido por palabra: {i}/{len(corpus)} con error: [{E}]")

    print(f"Fin de época: {epoca}")
    
    return W1, W2

#i= 4
#C = 4
#palabra_central = corpus[i]
#contexto = corpus[i-C:i] + corpus[i+1:i+C+1]
#print(palabra_central)
#print(contexto)

W1, W2 = entrenar_cbow(1)