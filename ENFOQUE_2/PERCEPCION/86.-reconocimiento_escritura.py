import cv2
import pytesseract

def reconocer_escritura(ruta_imagen):
    # 1. CARGA Y PREPROCESADO
    # La percepción mejora si la imagen está en blanco y negro puro
    img = cv2.imread(ruta_imagen)
    if img is None: return "Imagen no encontrada"
    
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicamos un umbral para resaltar el texto (Binarización)
    # Esto elimina el "ruido" del papel o fondo
    _, binarizada = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. RECONOCIMIENTO (La parte perceptual)
    # Configuramos el idioma (español)
    texto_detectado = pytesseract.image_to_string(binarizada, lang='spa')

    # 3. RESULTADOS
    print("--- Texto Percibido ---")
    print(texto_detectado)
    
    cv2.imshow('Imagen Procesada para OCR', binarizada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Requisito: Tener instalado Tesseract OCR en el sistema y la librería
# pip install pytesseract