import numpy as np

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

W1, W2, N, C, eta = cargar_modelo_completo()
print(f"Shape de W: {W1.shape}, Shape de W':{W2.shape}, Tipo: {type(W1)}")