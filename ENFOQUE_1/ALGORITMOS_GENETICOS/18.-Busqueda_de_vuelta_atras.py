"""
Búsqueda de Vuelta Atrás (Backtracking) - Satisfacción de Restricciones
Problema: N-Reinas
"""

def es_seguro(tablero, fila, col, n):
    """Verifica si es seguro colocar una reina en (fila, col)"""
    # Verificar columna
    for i in range(fila):
        if tablero[i] == col:
            return False
    
    # Verificar diagonales
    for i in range(fila):
        if abs(tablero[i] - col) == abs(i - fila):
            return False
    
    return True

def resolver_nreinas(tablero, fila, n, soluciones):
    """Backtracking para resolver N-Reinas"""
    if fila == n:
        soluciones.append(tablero[:])
        return
    
    for col in range(n):
        if es_seguro(tablero, fila, col, n):
            tablero[fila] = col
            resolver_nreinas(tablero, fila + 1, n, soluciones)
            tablero[fila] = -1  # Backtrack

def n_reinas(n):
    """Encuentra todas las soluciones para N-Reinas"""
    tablero = [-1] * n
    soluciones = []
    resolver_nreinas(tablero, 0, n, soluciones)
    return soluciones

def imprimir_solucion(solucion):
    """Imprime una solución en forma visual"""
    n = len(solucion)
    for i in range(n):
        fila = ""
        for j in range(n):
            fila += "R " if solucion[i] == j else "· "
        print(fila)
    print()

# Ejemplo de uso
if __name__ == "__main__":
    n = 4
    soluciones = n_reinas(n)
    
    print(f"Soluciones para {n}-Reinas: {len(soluciones)}\n")
    
    for idx, solucion in enumerate(soluciones, 1):
        print(f"Solución {idx}:")
        imprimir_solucion(solucion)
