from ultralytics import YOLO
import cv2

def reconocimiento_perceptual(ruta_imagen):
    # 1. CARGA DEL MODELO (El "Cerebro" entrenado)
    # YOLOv8n es un modelo ligero que "percibe" 80 categorías diferentes
    model = YOLO('yolov8n.pt') 

    # 2. INFERENCIA (Proceso de Reconocimiento)
    # El modelo analiza patrones, texturas y formas simultáneamente
    resultados = model(ruta_imagen)

    # 3. VISUALIZACIÓN
    # Dibujamos los cuadros delimitadores (Bounding Boxes) y etiquetas
    for r in resultados:
        img_dibujada = r.plot() # Dibuja las detecciones sobre la imagen

        # Mostrar la percepción final
        cv2.imshow('Reconocimiento de Objetos', img_dibujada)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Para ejecutar:
# reconocimiento_perceptual('foto_calle.jpg')