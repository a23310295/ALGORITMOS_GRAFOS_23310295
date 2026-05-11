import cv2
import numpy as np

def procesar_percepcion_avanzada(ruta_imagen):
    # 1. Carga y simplificación
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "Imagen no encontrada"

    # --- DETECCIÓN DE ARISTAS (Gradiente de Sobel) ---
    # Detecta cambios bruscos de intensidad en X y Y
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Combinamos para obtener la magnitud de la arista
    aristas = cv2.magnitude(sobel_x, sobel_y)
    aristas = np.uint8(np.absolute(aristas))

    # --- SEGMENTACIÓN (Umbralización de Otsu) ---
    # Calcula automáticamente el mejor umbral para separar objeto de fondo
    # basado en el histograma de la imagen.
    _, segmentada = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Mostrar resultados
    cv2.imshow('Aristas (Estructura)', aristas)
    cv2.imshow('Segmentacion (Separacion de Objetos)', segmentada)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ejecutar: procesar_percepcion_avanzada('tu_imagen.jpg')