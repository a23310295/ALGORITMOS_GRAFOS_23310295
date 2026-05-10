import random
from collections import defaultdict

class NodoMarkov:
    """Nodo en una red bayesiana con manto de Markov"""
    def __init__(self, nombre):
        self.nombre = nombre
        self.padres = set()
        self.hijos = set()
        self.hermanos = set()
        self.probabilidad_condicional = {}
    
    def calcular_manto_markov(self):
        """Calcula el manto de Markov del nodo"""
        manto = set()
        # Padres
        manto.update(self.padres)
        # Hijos
        manto.update(self.hijos)
        # Padres de los hijos (copadres)
        for hijo in self.hijos:
            manto.update(hijo.padres - {self})
        return manto
    
    def es_independiente_de(self, otro_nodo):
        """Verifica si este nodo es independiente de otro dado su manto de Markov"""
        manto = self.calcular_manto_markov()
        return otro_nodo not in manto and otro_nodo not in {self}


class RedBayesiana:
    """Red Bayesiana para razonamiento probabilístico"""
    def __init__(self):
        self.nodos = {}
        self.evidencia = {}
    
    def agregar_nodo(self, nombre):
        """Agrega un nodo a la red"""
        if nombre not in self.nodos:
            self.nodos[nombre] = NodoMarkov(nombre)
        return self.nodos[nombre]
    
    def agregar_relacion(self, padre, hijo):
        """Agrega una relación padre-hijo"""
        nodo_padre = self.agregar_nodo(padre)
        nodo_hijo = self.agregar_nodo(hijo)
        
        nodo_padre.hijos.add(nodo_hijo)
        nodo_hijo.padres.add(nodo_padre)
    
    def establecer_probabilidad(self, nodo, config_padres, probabilidad):
        """Establece la probabilidad condicional de un nodo"""
        nodo_obj = self.nodos[nodo]
        nodo_obj.probabilidad_condicional[config_padres] = probabilidad
    
    def mostrar_manto_markov(self, nombre_nodo):
        """Muestra el manto de Markov de un nodo"""
        nodo = self.nodos[nombre_nodo]
        manto = nodo.calcular_manto_markov()
        
        print(f"\n{'='*50}")
        print(f"Manto de Markov para: {nombre_nodo}")
        print(f"{'='*50}")
        print(f"Padres: {[n.nombre for n in nodo.padres] if nodo.padres else 'Ninguno'}")
        print(f"Hijos: {[n.nombre for n in nodo.hijos] if nodo.hijos else 'Ninguno'}")
        print(f"Copadres (padres de hijos): {[n.nombre for n in manto - nodo.padres - nodo.hijos] if manto - nodo.padres - nodo.hijos else 'Ninguno'}")
        print(f"\nManto de Markov completo: {[n.nombre for n in manto]}")
        print(f"Total de nodos en el manto: {len(manto)}")
    
    def verificar_independencias(self, nodo1, nodo2):
        """Verifica independencias entre nodos"""
        nodo = self.nodos[nodo1]
        otro = self.nodos[nodo2]
        
        print(f"\n{'='*50}")
        print(f"Análisis de Independencia")
        print(f"{'='*50}")
        print(f"¿{nodo1} es independiente de {nodo2}?")
        
        if nodo.es_independiente_de(otro):
            print(f"SÍ: {nodo1} es marginalmente independiente de {nodo2}")
        else:
            print(f"NO: {nodo1} depende de {nodo2}")
            manto = nodo.calcular_manto_markov()
            if otro in manto:
                print(f"  Razón: {nodo2} está en el manto de Markov de {nodo1}")


# Ejemplo de uso: Red Bayesiana para diagnóstico médico
print("RAZONAMIENTO PROBABILÍSTICO: MANTO DE MARKOV")
print("=" * 60)
print("Ejemplo: Red Bayesiana para Diagnóstico Médico")
print("=" * 60)

# Crear red bayesiana
red = RedBayesiana()

# Agregar nodos y relaciones
# Estructura: Factores → Enfermedad → Síntomas
red.agregar_relacion("Genética", "Diabetes")
red.agregar_relacion("Dieta", "Diabetes")
red.agregar_relacion("Ejercicio", "Diabetes")
red.agregar_relacion("Diabetes", "AltaGlucosa")
red.agregar_relacion("Diabetes", "Sed")
red.agregar_relacion("AltaGlucosa", "Fatiga")
red.agregar_relacion("Sed", "Fatiga")

# Mostrar manto de Markov para diferentes nodos
red.mostrar_manto_markov("Diabetes")
red.mostrar_manto_markov("AltaGlucosa")
red.mostrar_manto_markov("Fatiga")
red.mostrar_manto_markov("Dieta")

# Verificar independencias
red.verificar_independencias("Genética", "Dieta")
red.verificar_independencias("AltaGlucosa", "Sed")
red.verificar_independencias("Genética", "Fatiga")

print("\n" + "=" * 60)
print("EXPLICACIÓN DEL MANTO DE MARKOV:")
print("=" * 60)
print("""
El Manto de Markov de un nodo N consiste en:
1. Sus padres (causas directas)
2. Sus hijos (consecuencias directas)
3. Los padres de sus hijos (copadres)

Un nodo es condicionalmente independiente de todos los
demás nodos de la red, dado su Manto de Markov.

Esto reduce significativamente la complejidad computacional
en redes bayesianas grandes.
""")
