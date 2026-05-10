import random
import numpy as np
from collections import Counter

# ============================================================
# RAZONAMIENTO PROBABILÍSTICO: MUESTREO DIRECTO Y POR RECHAZO
# ============================================================

# MUESTREO DIRECTO
# Genera muestras directamente de la distribución de probabilidad
# sin restricciones

class MuestreoDirecto:
    """
    Genera muestras directamente de una red bayesiana simple.
    """
    
    def __init__(self):
        # Probabilidades a priori
        self.p_lluvia = 0.3
        self.p_aspersor = 0.4
        
    def generar_muestra(self):
        """
        Genera una muestra de variables: Lluvia, Aspersor, PastohúmedoRiego
        """
        # Paso 1: Muestrear variable independiente (Lluvia)
        lluvia = random.random() < self.p_lluvia
        
        # Paso 2: Muestrear variable independiente (Aspersor)
        aspersor = random.random() < self.p_aspersor
        
        # Paso 3: Muestrear variable dependiente (PastohúmedoRiego)
        # P(Pasto_húmedo=T | Lluvia, Aspersor)
        if lluvia and aspersor:
            pasto_humedo = random.random() < 0.99
        elif lluvia or aspersor:
            pasto_humedo = random.random() < 0.8
        else:
            pasto_humedo = random.random() < 0.1
            
        return {
            'Lluvia': lluvia,
            'Aspersor': aspersor,
            'Pasto_húmedo': pasto_humedo
        }
    
    def generar_muestras(self, n):
        """Genera n muestras"""
        return [self.generar_muestra() for _ in range(n)]
    
    def estimar_probabilidad(self, n, variable, valor):
        """Estima P(variable=valor) a partir de n muestras"""
        muestras = self.generar_muestras(n)
        contador = sum(1 for m in muestras if m[variable] == valor)
        return contador / n


# MUESTREO POR RECHAZO
# Solo acepta muestras que satisfacen evidencia dada
# Rechaza el resto

class MuestreoRechazo:
    """
    Implementa muestreo por rechazo con evidencia.
    P(X|e) se estima usando solo las muestras consistentes con e.
    """
    
    def __init__(self):
        self.p_lluvia = 0.3
        self.p_aspersor = 0.4
    
    def generar_muestra(self):
        """Genera una muestra completa"""
        lluvia = random.random() < self.p_lluvia
        aspersor = random.random() < self.p_aspersor
        
        if lluvia and aspersor:
            pasto_humedo = random.random() < 0.99
        elif lluvia or aspersor:
            pasto_humedo = random.random() < 0.8
        else:
            pasto_humedo = random.random() < 0.1
            
        return {
            'Lluvia': lluvia,
            'Aspersor': aspersor,
            'Pasto_húmedo': pasto_humedo
        }
    
    def es_consistente(self, muestra, evidencia):
        """Verifica si la muestra es consistente con la evidencia"""
        for variable, valor in evidencia.items():
            if muestra[variable] != valor:
                return False
        return True
    
    def estimar_probabilidad(self, n, variable, valor, evidencia):
        """
        Estima P(variable=valor | evidencia)
        Genera n muestras y rechaza las inconsistentes
        """
        muestras_aceptadas = []
        intentos = 0
        
        # Generar muestras hasta obtener suficientes aceptadas
        while len(muestras_aceptadas) < n and intentos < n * 100:
            muestra = self.generar_muestra()
            intentos += 1
            
            if self.es_consistente(muestra, evidencia):
                muestras_aceptadas.append(muestra)
        
        if not muestras_aceptadas:
            print("Advertencia: No se generaron muestras consistentes")
            return 0.0
        
        # Contar cuántas muestras aceptadas cumplen la condición
        contador = sum(1 for m in muestras_aceptadas 
                      if m[variable] == valor)
        
        prob = contador / len(muestras_aceptadas)
        tasa_aceptacion = len(muestras_aceptadas) / intentos
        
        return prob, tasa_aceptacion, len(muestras_aceptadas)


# ============================================================
# PRUEBAS Y COMPARACIONES
# ============================================================

def main():
    print("=" * 70)
    print("MUESTREO DIRECTO Y POR RECHAZO - RAZONAMIENTO PROBABILÍSTICO")
    print("=" * 70)
    
    # 1. MUESTREO DIRECTO
    print("\n1. MUESTREO DIRECTO")
    print("-" * 70)
    print("Estimando P(Lluvia=Verdadero) con 10,000 muestras\n")
    
    md = MuestreoDirecto()
    prob_lluvia = md.estimar_probabilidad(10000, 'Lluvia', True)
    print(f"P(Lluvia=True) ≈ {prob_lluvia:.4f} (valor real: 0.3000)")
    
    prob_aspersor = md.estimar_probabilidad(10000, 'Aspersor', True)
    print(f"P(Aspersor=True) ≈ {prob_aspersor:.4f} (valor real: 0.4000)")
    
    prob_pasto = md.estimar_probabilidad(10000, 'Pasto_húmedo', True)
    print(f"P(Pasto_húmedo=True) ≈ {prob_pasto:.4f}")
    
    # 2. MUESTREO POR RECHAZO SIN EVIDENCIA
    print("\n2. MUESTREO POR RECHAZO - Sin Evidencia")
    print("-" * 70)
    
    mr = MuestreoRechazo()
    evidencia_vacia = {}
    prob, tasa, num = mr.estimar_probabilidad(
        1000, 'Lluvia', True, evidencia_vacia
    )
    print(f"P(Lluvia=True | ∅) ≈ {prob:.4f}")
    print(f"Tasa de aceptación: {tasa:.4f} (muestras aceptadas: {num})")
    
    # 3. MUESTREO POR RECHAZO CON EVIDENCIA
    print("\n3. MUESTREO POR RECHAZO - Con Evidencia")
    print("-" * 70)
    
    # Evidencia: Pasto está húmedo
    print("\nEvidencia: Pasto_húmedo = True")
    print("Queremos estimar: P(Lluvia=True | Pasto_húmedo=True)\n")
    
    evidencia = {'Pasto_húmedo': True}
    prob, tasa, num = mr.estimar_probabilidad(
        1000, 'Lluvia', True, evidencia
    )
    print(f"P(Lluvia=True | Pasto_húmedo=True) ≈ {prob:.4f}")
    print(f"Muestras aceptadas: {num} de ~{1000 * 100}")
    print(f"Tasa de aceptación: {tasa:.4f}")
    
    # 4. COMPARACIÓN: Diferentes evidencias
    print("\n4. COMPARACIÓN - Diferentes Evidencias")
    print("-" * 70)
    
    evidencias = [
        ('Aspersor=True', {'Aspersor': True}),
        ('Aspersor=False', {'Aspersor': False}),
        ('Lluvia=True', {'Lluvia': True}),
    ]
    
    for desc, evidencia in evidencias:
        prob, tasa, num = mr.estimar_probabilidad(
            500, 'Pasto_húmedo', True, evidencia
        )
        print(f"\nEvidencia: {desc}")
        print(f"  P(Pasto_húmedo=True | {desc}) ≈ {prob:.4f}")
        print(f"  Muestras aceptadas: {num}")
    
    # 5. Análisis de eficiencia
    print("\n5. ANÁLISIS DE EFICIENCIA - Tasa de Rechazo")
    print("-" * 70)
    
    print("\nEvidencia: Pasto_húmedo=True Y Lluvia=True")
    evidencia_compleja = {'Pasto_húmedo': True, 'Lluvia': True}
    prob, tasa, num = mr.estimar_probabilidad(
        500, 'Aspersor', True, evidencia_compleja
    )
    print(f"Tasa de aceptación: {tasa:.4f}")
    print(f"Tasa de rechazo: {1-tasa:.4f}")
    print(f"Muestras aceptadas: {num}")
    print("Nota: Evidencia más restrictiva = menor tasa de aceptación")


if __name__ == "__main__":
    main()
