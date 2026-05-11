import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def analizar_percepcion_avanzada(ruta_imagen):
    # 1. Cargar imagen
    img = cv2.imread(ruta_imagen)
    if img is None: return "Imagen no encontrada"
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- ANÁLISIS DE TEXTURA (GLCM) ---
    # Analizamos qué tan "homogénea" es la superficie
    glcm = graycomatrix(gris, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contraste = graycoprops(glcm, 'contrast')[0, 0]
    homogeneidad = graycoprops(glcm, 'homogeneity')[0, 0]
    
    print(f"Percepción de Textura -> Contraste: {contraste:.2f}, Homogeneidad: {homogeneidad:.2f}")

    # --- DETECCIÓN DE SOMBRAS (Detección de áreas oscuras) ---
    # Usamos un umbral invertido para detectar las zonas de sombra
    # El cerebro percibe la sombra como una reducción de luminancia con bordes suaves
    _, sombras = cv2.threshold(gris, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Suavizamos las sombras para que parezcan más "naturales" al ojo
    sombras_suaves = cv2.GaussianBlur(sombras, (15, 15), 0)

    # 3. Mostrar resultados
    cv2.imshow('Textura - Gris', gris)
    cv2.imshow('Percepcion de Sombras', sombras_suaves)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Nota: Para la textura necesitas instalar: pip install scikit-image