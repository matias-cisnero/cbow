import numpy as np
from funciones_auxiliares import cargar_corpus, inicializar_pesos, sigmoide_np, generar_tuplas_central_contexto_negativos, guardar_modelo

def entrenar_skipgram_neg_samp(ruta_corpus, nombre_pc, epocas=1, η=0.001, N=300, C=4, K=5, W1=None, W2=None, intervalo_guardado=50):
    
    corpus, vocab, vocab_size, word_to_idx, idx_to_word = cargar_corpus(ruta_corpus)
    W1, W2 = inicializar_pesos(vocab_size, N, W1, W2, cparray=False)
    indice_tuplas = generar_tuplas_central_contexto_negativos(corpus, word_to_idx, C, K)
    total_pares = len(indice_tuplas)

    print(f"Comienzo de entrenamiento con {epocas} epocas.")
    for epoca in range(epocas):
        for i, (i_central, i_positivos, i_negativos) in enumerate(indice_tuplas):
            i_total = i_positivos + i_negativos

            # ---Propagación---
            h = W1[i_central].reshape(-1, 1)
            u = W2[:, i_total].T @ h
            y = sigmoide_np(u)

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
            nombre_archivo = f'pesos_skipgram_neg_samp_{nombre_pc}_epoca{epoca}.npz'
            guardar_modelo(nombre_archivo, W1, W2, eta=η, N=N, C=C, cparray=False)

    print(f"Entrenamiento con {epocas} terminado.")
    return W1, W2

W1, W2 = entrenar_skipgram_neg_samp("corpus/corpus.txt", "pcmati", epocas=10000, η=0.01, N=50, C=4, K=5, intervalo_guardado=100)