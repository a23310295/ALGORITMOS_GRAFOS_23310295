import numpy as np

class KalmanFilter1D:
    def __init__(self, x0, p0, q, r):
        # Estado inicial x0, varianza inicial p0
        self.x = x0
        self.P = p0
        self.Q = q  # ruido de proceso
        self.R = r  # ruido de medida

    def predict(self):
        # Para un modelo 1D simple sin control y con estado constante
        self.P = self.P + self.Q
        return self.x

    def update(self, z):
        # Ganancia de Kalman
        K = self.P / (self.P + self.R)
        # Actualización del estado con la medida z
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P
        return self.x


def ejemplo_filtro_kalman():
    # Simulación de un proceso 1D
    np.random.seed(42)
    pasos = 20
    verdadera_pos = 0.0
    velocidad = 0.5
    ruido_medida = 1.0

    kalman = KalmanFilter1D(x0=0.0, p0=1.0, q=0.01, r=ruido_medida**2)

    medidas = []
    estimaciones = []
    verdaderas = []

    for i in range(pasos):
        verdadera_pos += velocidad
        medida = verdadera_pos + np.random.normal(0, ruido_medida)

        kalman.predict()
        estimacion = kalman.update(medida)

        verdaderas.append(verdadera_pos)
        medidas.append(medida)
        estimaciones.append(estimacion)

        print(f"Paso {i+1:02d}: verdadera={verdadera_pos:.2f}, medida={medida:.2f}, estimacion={estimacion:.2f}")

    return verdaderas, medidas, estimaciones


if __name__ == "__main__":
    ejemplo_filtro_kalman()
