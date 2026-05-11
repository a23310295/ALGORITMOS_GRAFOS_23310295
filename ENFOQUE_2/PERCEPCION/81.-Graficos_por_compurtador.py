import numpy as np

def modelo_phong(punto_superficie, normal, posicion_camara, posicion_luz, color_objeto):
    """
    Calcula la intensidad del color percibido en un punto.
    """
    # 1. Configuración de parámetros perceptuales (coeficientes)
    ka = 0.1  # Ambiental (luz base)
    kd = 0.6  # Difusa (rugosidad de la superficie)
    ks = 0.3  # Especular (brillo/reflejo)
    brillo = 32 # Exponente de suavizado
    
    # 2. Vectores unitarios fundamentales
    L = (posicion_luz - punto_superficie) / np.linalg.norm(posicion_luz - punto_superficie)
    N = normal / np.linalg.norm(normal)
    V = (posicion_camara - punto_superficie) / np.linalg.norm(posicion_camara - punto_superficie)
    
    # Vector de reflexión (R)
    R = 2 * np.dot(L, N) * N - L
    
    # 3. Componentes del Modelo
    # Ambiental: Simula la luz que rebota en todas partes
    ambiental = ka * color_objeto
    
    # Difusa: Depende del ángulo entre la luz y la normal (Ley de Lambert)
    difusa = kd * max(np.dot(L, N), 0) * color_objeto
    
    # Especular: Simula el reflejo directo de la fuente de luz hacia el ojo
    especular = ks * max(np.dot(R, V), 0)**brillo * np.array([1, 1, 1]) # Luz blanca
    
    return ambiental + difusa + especular

# Ejemplo de uso
p_camara = np.array([0, 0, 10])
p_luz = np.array([5, 5, 5])
normal_cara = np.array([0, 0, 1])
color_rojo = np.array([255, 0, 0])

resultado = modelo_phong(np.array([0,0,0]), normal_cara, p_camara, p_luz, color_rojo)
print(f"Color final percibido (RGB): {resultado.astype(int)}")