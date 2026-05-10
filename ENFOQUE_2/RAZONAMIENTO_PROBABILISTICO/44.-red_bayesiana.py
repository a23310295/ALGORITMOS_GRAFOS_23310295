import numpy as np
from itertools import product

class NodoRed:
    """Representa un nodo en la red Bayesiana"""
    def __init__(self, nombre, padres=None):
        self.nombre = nombre
        self.padres = padres if padres else []
        self.tabla_probabilidad = {}
    
    def set_probabilidad(self, config, prob):
        """Establece la probabilidad condicional para una configuración"""
        self.tabla_probabilidad[config] = prob
    
    def get_probabilidad(self, config):
        """Obtiene la probabilidad condicional para una configuración"""
        return self.tabla_probabilidad.get(config, 0)


class RedBayesiana:
    """Implementa una Red Bayesiana para razonamiento probabilístico"""
    def __init__(self):
        self.nodos = {}
    
    def agregar_nodo(self, nombre, padres=None):
        """Agrega un nodo a la red"""
        nodo = NodoRed(nombre, padres)
        self.nodos[nombre] = nodo
        return nodo
    
    def probabilidad_conjunta(self, configuracion):
        """
        Calcula la probabilidad conjunta de una configuración
        configuracion: dict con {nodo: valor}
        """
        prob = 1.0
        for nombre_nodo, nodo in self.nodos.items():
            valor = configuracion.get(nombre_nodo)
            if valor is None:
                continue
            
            # Construir configuración de padres
            config_padres = tuple(configuracion.get(padre) for padre in nodo.padres)
            config_completa = config_padres + (valor,)
            
            prob *= nodo.get_probabilidad(config_completa)
        
        return prob
    
    def inferencia_por_enumeracion(self, variable_consulta, variables_evidencia, valores_posibles):
        """
        Realiza inferencia por enumeración
        variable_consulta: variable que queremos consultar
        variables_evidencia: dict con evidencia observada
        valores_posibles: dict con valores posibles para cada variable
        """
        # Obtener todas las variables no observadas
        variables_ocultas = [v for v in self.nodos.keys() 
                            if v not in variables_evidencia and v != variable_consulta]
        
        prob_total = {}
        
        # Para cada valor posible de la variable de consulta
        for valor_consulta in valores_posibles[variable_consulta]:
            config_base = variables_evidencia.copy()
            config_base[variable_consulta] = valor_consulta
            
            # Enumerar todas las combinaciones de variables ocultas
            if variables_ocultas:
                valores_ocultos = [valores_posibles[v] for v in variables_ocultas]
                suma = 0
                
                for combinacion in product(*valores_ocultos):
                    config_completa = config_base.copy()
                    for var_oculta, valor in zip(variables_ocultas, combinacion):
                        config_completa[var_oculta] = valor
                    
                    suma += self.probabilidad_conjunta(config_completa)
                
                prob_total[valor_consulta] = suma
            else:
                prob_total[valor_consulta] = self.probabilidad_conjunta(config_base)
        
        # Normalizar
        suma_total = sum(prob_total.values())
        if suma_total > 0:
            prob_total = {k: v/suma_total for k, v in prob_total.items()}
        
        return prob_total


# Ejemplo: Red Bayesiana para diagnóstico de enfermedades
def ejemplo_diagnostico():
    print("=" * 60)
    print("RED BAYESIANA - RAZONAMIENTO PROBABILÍSTICO")
    print("=" * 60)
    print("\nEjemplo: Diagnóstico médico\n")
    
    # Crear red bayesiana
    red = RedBayesiana()
    
    # Agregar nodos
    enfermedad = red.agregar_nodo("Enfermedad")
    sintoma = red.agregar_nodo("Síntoma", padres=["Enfermedad"])
    
    # Probabilidades: P(Enfermedad)
    enfermedad.set_probabilidad((True,), 0.1)
    enfermedad.set_probabilidad((False,), 0.9)
    
    # Probabilidades: P(Síntoma | Enfermedad)
    sintoma.set_probabilidad((True, True), 0.9)    # P(Síntoma=T | Enfermedad=T)
    sintoma.set_probabilidad((True, False), 0.1)   # P(Síntoma=F | Enfermedad=T)
    sintoma.set_probabilidad((False, True), 0.2)   # P(Síntoma=T | Enfermedad=F)
    sintoma.set_probabilidad((False, False), 0.8)  # P(Síntoma=F | Enfermedad=F)
    
    print("Estructura de la red:")
    print("  Enfermedad -> Síntoma\n")
    
    # Caso 1: Probabilidad a priori
    print("1. PROBABILIDAD A PRIORI:")
    print("   P(Enfermedad=Sí) =", enfermedad.get_probabilidad((True,)))
    print("   P(Enfermedad=No) =", enfermedad.get_probabilidad((False,)))
    
    # Caso 2: Probabilidad conjunta
    print("\n2. PROBABILIDAD CONJUNTA:")
    config1 = {"Enfermedad": True, "Síntoma": True}
    config2 = {"Enfermedad": False, "Síntoma": True}
    print(f"   P(Enfermedad=Sí, Síntoma=Sí) = {red.probabilidad_conjunta(config1):.4f}")
    print(f"   P(Enfermedad=No, Síntoma=Sí) = {red.probabilidad_conjunta(config2):.4f}")
    
    # Caso 3: Inferencia (Teorema de Bayes)
    print("\n3. INFERENCIA POSTERIOR (con observación de síntoma):")
    valores_posibles = {
        "Enfermedad": [True, False],
        "Síntoma": [True, False]
    }
    evidencia = {"Síntoma": True}
    
    resultado = red.inferencia_por_enumeracion("Enfermedad", evidencia, valores_posibles)
    print("   Dado que el paciente tiene síntoma:")
    print(f"   P(Enfermedad=Sí | Síntoma=Sí) = {resultado[True]:.4f}")
    print(f"   P(Enfermedad=No | Síntoma=Sí) = {resultado[False]:.4f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    ejemplo_diagnostico()
