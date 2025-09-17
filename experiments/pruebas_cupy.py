import cupy as cp

try:
    # 1. Obtener el ID del dispositivo actual
    device_id = cp.cuda.runtime.getDevice()

    # 2. Obtener las propiedades de ese dispositivo
    device_props = cp.cuda.runtime.getDeviceProperties(device_id)

    # 3. El nombre viene en bytes, así que lo decodificamos a string
    device_name = device_props['name'].decode('utf-8')

    print(f"✅ CuPy está utilizando la siguiente GPU:")
    print(f"   ID del Dispositivo: {device_id}")
    print(f"   Nombre: {device_name}")

except cp.cuda.runtime.CUDARuntimeError as e:
    print("❌ Error: No se pudo acceder a la GPU.")
    print("Asegúrate de que los drivers de NVIDIA y el CUDA Toolkit estén bien instalados.")
    print(f"Error original: {e}")