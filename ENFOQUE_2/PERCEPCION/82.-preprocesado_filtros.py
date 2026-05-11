import cv2
import numpy as np

def preprocesado_percepcion(imagen_path):
    # 1. Reducción de dimensionalidad: Convertir a escala de grises
    # El sistema de percepción a menudo prioriza la luminancia sobre el color.
    img = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
    
    # 2. Suavizado (Reducción de Ruido): Filtro Gaussiano
    # Elimina texturas irrelevantes que el cerebro ignoraría.
    # Usamos un kernel de 5x5
    suavizada = cv2.GaussianBlur(img, (5, 5), 1.4)
    
    # 3. Gradiente de Intensidad: Algoritmo de Sobel
    # Encuentra las zonas de mayor cambio de contraste (bordes).
    grad_x = cv2.Sobel(suavizada, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(suavizada, cv2.CV_64F, 0, 1, ksize=3)
    magnitud = np.sqrt(grad_x**2 + grad_y**2)
    
    # 4. Umbralización (Thresholding): Canny
    # Decidir qué es un borde real y qué es ruido de fondo.
    bordes = cv2.Canny(suavizada, 100, 200)
    
    return bordes

# Nota: Requiere tener una imagen llamada 'input.jpg' o cambiar la ruta.
# resultado = preprocesado_percepcion('input.jpg')