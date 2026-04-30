# Búsqueda por Profundidad Limitada (Depth Limited Search - DLS)

class Nodo:
    def __init__(self, valor, hijos=None):
        self.valor = valor
        self.hijos = hijos if hijos else []

def busqueda_profundidad_limitada(nodo, objetivo, limite):
    """
    Busca un nodo objetivo usando profundidad limitada.
    
    Args:
        nodo: Nodo raíz del árbol
        objetivo: Valor a buscar
        limite: Profundidad máxima permitida
    
    Returns:
        True si encuentra el objetivo, False en caso contrario
    """
    return dls_recursivo(nodo, objetivo, limite)

def dls_recursivo(nodo, objetivo, limite):
    """
    Búsqueda recursiva con límite de profundidad.
    """
    if nodo.valor == objetivo:
        return True
    
    if limite == 0:
        return False
    
    for hijo in nodo.hijos:
        if dls_recursivo(hijo, objetivo, limite - 1):
            return True
    
    return False

# Ejemplo de uso
if __name__ == "__main__":
    # Crear árbol de ejemplo
    #       1
    #      / \
    #     2   3
    #    / \   \
    #   4   5   6
    
    nodo4 = Nodo(4)
    nodo5 = Nodo(5)
    nodo6 = Nodo(6)
    nodo2 = Nodo(2, [nodo4, nodo5])
    nodo3 = Nodo(3, [nodo6])
    nodo1 = Nodo(1, [nodo2, nodo3])
    
    # Búsquedas
    print("Buscando 5 con límite 2:", busqueda_profundidad_limitada(nodo1, 5, 2))
    print("Buscando 6 con límite 2:", busqueda_profundidad_limitada(nodo1, 6, 2))
    print("Buscando 5 con límite 1:", busqueda_profundidad_limitada(nodo1, 5, 1))
