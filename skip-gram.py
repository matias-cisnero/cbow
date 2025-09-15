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

def sigmoide(h):
    return 1 / (1 + np.exp(-h))

def entrenar_skipgram(epocas, η=0.001):
    card_V = vocab_size
    N = 300
    C = 4

    # x ∈ |V| x 1 <--- Entrada
    # W ∈ |V| x N
    # h ∈ N x 1 <--- Oculta (lineal)
    # W' ∈ N x |V|
    # C paneles y ∈ |V| x 1 <--- Salida (softmax)

    W1 = np.random.normal(0, 0.1, (card_V, N))
    W2 = np.random.normal(0, 0.1, (N, card_V))

    print(f"Comienzo de entrenamiento con {epocas} epocas.")
    for epoca in range(epocas):

        #for i in range(C, 8):
        for i in range(C, len(corpus)-C):
            
            # Palabra de entrada
            palabra_central = corpus[i]
            palabra_central_indice = word_to_idx[palabra_central]
            
            # Palabras objetivo
            palabras_contexto = corpus[i-C:i] + corpus[i+1:i+C+1]
            palabras_contexto_indices = [word_to_idx[word] for word in palabras_contexto]

            # ---Propagación---

            h = W1[palabra_central_indice].reshape(-1, 1)
            
            u = W2.T @ h

            y = softmax(u)

            # ---Retropropagación---

            EI = y.copy()
            EI[palabras_contexto_indices] -= 1

            EH = W2 @ EI

            # ---Actualización---

            W2 -= η * (h @ EI.T)

            W1 -= η * EH.T
            print(f"Recorrido por palabra: {i}/{len(corpus)} con error: [{1}]")

    print(f"Fin de época: {epoca}")
    
    return W1, W2

def tomar_indices_negativos(negativos):
    pass

def entrenar_skipgram2(epocas, η=0.001):
    card_V = vocab_size
    N = 300
    C = 4
    negativos = 5

    # x ∈ |V| x 1 <--- Entrada
    # W ∈ |V| x N
    # h ∈ N x 1 <--- Oculta (lineal)
    # W' ∈ N x |V|
    # C paneles y ∈ |V| x 1 <--- Salida (softmax)

    W1 = np.random.normal(0, 0.1, (card_V, N))
    W2 = np.random.normal(0, 0.1, (N, card_V))

    print(f"Comienzo de entrenamiento con {epocas} epocas.")
    for epoca in range(epocas):

        #for i in range(C, 8):
        for i in range(C, len(corpus)-C):
            
            # Palabra de entrada
            palabra_central = corpus[i]
            palabra_central_indice = word_to_idx[palabra_central]
            
            # Palabras objetivo
            palabras_contexto = corpus[i-C:i] + corpus[i+1:i+C+1]
            palabras_contexto_indices = [word_to_idx[word] for word in palabras_contexto]

            # Palabras de ejemplos negativos


            # ---Propagación---

            h = W1[palabra_central_indice].reshape(-1, 1)
            
            u = W2.T @ h

            y = softmax(u)

            # ---Retropropagación---

            EI = y.copy()
            EI[palabras_contexto_indices] -= 1

            EH = W2 @ EI

            # ---Actualización---

            W2 -= η * (h @ EI.T)

            W1 -= η * EH.T
            print(f"Recorrido por palabra: {i}/{len(corpus)} con error: [{1}]")

    print(f"Fin de época: {epoca}")
    
    return W1, W2

W1, W2 = entrenar_skipgram2(1)