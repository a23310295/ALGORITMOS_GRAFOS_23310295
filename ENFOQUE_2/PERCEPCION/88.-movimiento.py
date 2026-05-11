import cv2
import numpy as np

def detectar_movimiento_denso():
    # 1. Captura de video (Cámara web)
    cap = cv2.VideoCapture(0)
    
    # Leer el primer fotograma
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    # Crear una máscara para la visualización (en formato HSV)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255 # Saturación máxima

    while True:
        ret, frame2 = cap.read()
        if not ret: break
        
        proximo = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # 2. CALCULAR EL FLUJO ÓPTICO
        # Parámetros: imagen_previa, imagen_siguiente, pirámide, niveles, iteraciones...
        flow = cv2.calcOpticalFlowFarneback(prvs, proximo, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # 3. INTERPRETACIÓN PERCEPTUAL
        # Convertir coordenadas cartesianas (x, y) a polares (magnitud, ángulo)
        magnitud, angulo = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # El ángulo define el COLOR (dirección del movimiento)
        hsv[..., 0] = angulo * 180 / np.pi / 2
        # La magnitud define el BRILLO (velocidad del movimiento)
        hsv[..., 2] = cv2.normalize(magnitud, None, 0, 255, cv2.NORM_MINMAX)

        # Convertir de HSV a BGR para mostrarlo
        rgb_movimiento = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('Percepcion de Movimiento (Flujo Optico)', rgb_movimiento)
        cv2.imshow('Imagen Real', frame2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        prvs = proximo

    cap.release()
    cv2.destroyAllWindows()

# detectar_movimiento_denso()