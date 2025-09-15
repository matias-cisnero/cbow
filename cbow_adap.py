import numpy as np

with open("corpus.txt", "r", encoding="utf-8") as f:
    corpus = f.read().splitlines()

# Vocabulario
vocab = sorted(set(corpus))
vocab_size = len(vocab)

# Diccionarios
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in enumerate(vocab)}

def softmax(h):
    num = np.exp(h - np.max(h))
    return num / np.sum(num)

def entrenar_cbow(corpus, vocab_size, word_to_idx, nombre_pc, epocas=1, eta=0.001, N=300, C=4, beta=0.9, W1=None, W2=None, vW1=None, vW2=None):
    if W1 is None or W2 is None:
        W1 = np.random.normal(0, 0.1, (vocab_size, N))
        W2 = np.random.normal(0, 0.1, (N, vocab_size))
    
    if vW1 is None or vW2 is None:
        vW1 = np.zeros_like(W1)
        vW2 = np.zeros_like(W2)

    print(f"Comienzo de entrenamiento con {epocas} epocas.")
    print(f"Parámetros: η={eta}, β={beta}")
    
    for epoca in range(epocas):
        for i in range(C, len(corpus) - C):
            palabra_central = corpus[i]
            palabra_central_indice = word_to_idx[palabra_central]

            palabras_contexto = corpus[i-C:i] + corpus[i+1:i+C+1]
            palabras_contexto_indices = [word_to_idx[word] for word in palabras_contexto]

            # ---Propagación---
            h = np.mean(W1[palabras_contexto_indices], axis=0).reshape(-1, 1)
            u = W2.T @ h
            y = softmax(u)

            # ---Retropropagación---
            E = -u[palabra_central_indice] + np.log(np.sum(np.exp(u)))

            e = y.copy()
            e[palabra_central_indice] -= 1

            grad_W2 = h @ e.T
            EH = W2 @ e
            grad_W1 = (1/C) * EH.T

            vW2 = beta * vW2 + eta * grad_W2
            vW1[palabras_contexto_indices] = beta * vW1[palabras_contexto_indices] + eta * grad_W1

            W2 -= vW2
            W1[palabras_contexto_indices] -= vW1[palabras_contexto_indices]
            
            print(f"Época {epoca}, Palabra: {i}/{len(corpus)}, Error: {E.item():.4f}")

        print(f"\nFin de época: {epoca}")

        nombre_archivo = f'pesos_cbow_{nombre_pc}_epoca{epoca}.npz'
        np.savez(nombre_archivo, W1=W1, W2=W2, vW1=vW1, vW2=vW2, eta=eta, N=N, C=C, beta=beta)
        print(f"Pesos, velocidades e hiperparámetros guardados en '{nombre_archivo}'")

    print(f"Entrenamiento con {epocas} terminado.")
    return W1, W2

def cargar_modelo_completo(nombre_archivo='pesos_cbow_pc2_epoca0.npz'):
    try:
        data = np.load(nombre_archivo, allow_pickle=True)
        
        W1 = data['W1']
        W2 = data['W2']
        
        N = data['N'].item()
        C = data['C'].item()
        eta = data['eta'].item()
        
        if 'vW1' in data and 'vW2' in data:
            vW1 = data['vW1']
            vW2 = data['vW2']
            print("Vectores de velocidad (momentum) cargados.")
        else:
            vW1, vW2 = None, None
            print("Advertencia: No se encontraron vectores de velocidad. Se iniciarán desde cero.")

        if 'beta' in data:
            beta = data['beta'].item()
        else:
            beta = 0.9
            print(f"Advertencia: No se encontró beta. Se usará el valor por defecto: {beta}")
        
        print(f"Modelo cargado exitosamente desde '{nombre_archivo}'")
            
        return W1, W2, vW1, vW2, N, C, eta, beta
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{nombre_archivo}'.")
        return None, None, None, None, None, None, None, None

W1, W2 = entrenar_cbow(corpus, vocab_size, word_to_idx, "pc2", 500, 0.001, 100)