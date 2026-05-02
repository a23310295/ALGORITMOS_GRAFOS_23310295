import random

def is_conflict(board, row, col):
    """
    Verifica si colocar una reina en (row, col) causa conflictos.
    """
    n = len(board)
    for i in range(n):
        if i != row:
            # Misma columna
            if board[i] == col:
                return True
            # Misma diagonal
            if abs(i - row) == abs(board[i] - col):
                return True
    return False

def min_conflicts(board, max_steps=1000):
    """
    Algoritmo de búsqueda local de mínimos conflictos para resolver N-Reinas.
    """
    n = len(board)
    for _ in range(max_steps):
        # Calcular conflictos para cada fila
        conflicts = [sum(1 for j in range(n) if is_conflict(board, i, board[i])) for i in range(n)]
        # Si no hay conflictos, solución encontrada
        if max(conflicts) == 0:
            return board
        # Elegir una fila con el máximo número de conflictos al azar
        max_conf = max(conflicts)
        row = random.choice([i for i, c in enumerate(conflicts) if c == max_conf])
        # Encontrar la columna con el mínimo número de conflictos para esa fila
        min_conf = min(sum(1 for j in range(n) if is_conflict(board, row, col)) for col in range(n))
        # Elegir una columna al azar entre las que minimizan conflictos
        board[row] = random.choice([col for col in range(n) if sum(1 for j in range(n) if is_conflict(board, row, col)) == min_conf])
    return None  # No se encontró solución en max_steps

def solve_n_queens(n):
    """
    Resuelve el problema de N-Reinas usando mínimos conflictos.
    """
    # Asignación inicial aleatoria
    board = [random.randint(0, n-1) for _ in range(n)]
    return min_conflicts(board)

# Ejemplo de uso
if __name__ == "__main__":
    n = 8
    solution = solve_n_queens(n)
    if solution:
        print("Solución encontrada:")
        for row in range(n):
            print(" ".join("Q" if col == solution[row] else "." for col in range(n)))
    else:
        print("No se encontró solución.")