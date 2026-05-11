import cv2
import numpy as np

def etiquetado_lineas_percepcion(ruta_imagen):
    # 1. Preprocesamiento: Detección de bordes
    img = cv2.imread(ruta_imagen)
    if img is None: return "Imagen no encontrada"
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(gris, 50, 150, apertureSize=3)

    # 2. Transformada de Hough: Detecta líneas matemáticas
    # rho: resolución de distancia, theta: resolución de ángulo
    lineas = cv2.HoughLines(bordes, 1, np.pi / 180, 200)

    # 3. Etiquetado lógico
    if lineas is not None:
        for i in range(len(lineas)):
            rho, theta = lineas[i][0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            # Convertimos a coordenadas de píxeles
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # ETIQUETADO PERCEPTUAL (Ejemplo basado en ángulo)
            # El cerebro etiqueta líneas horizontales vs verticales para profundidad
            color = (0, 255, 0) # Verde por defecto
            etiqueta = "Diagonal"
            
            if abs(theta) < 0.1: # Línea vertical aproximada
                color = (255, 0, 0) # Azul
                etiqueta = "Vertical"
            elif abs(theta - np.pi/2) < 0.1: # Línea horizontal aproximada
                color = (0, 0, 255) # Rojo
                etiqueta = "Horizontal"

            cv2.line(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, etiqueta, (int(x0), int(y0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imshow('Etiquetado de Lineas (Hough)', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Ejecutar con una imagen de formas geométricas
# etiquetado_lineas_percepcion('cubo.jpg')