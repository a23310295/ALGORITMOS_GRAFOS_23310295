import random

# Funcion objetivo a maximizar: ejemplo simple
def funcion_objetivo(x):
    return -(x - 3) ** 2 + 10

# Vecinos: valores cercanos a x
def generar_vecinos(x, paso=1, limite=(-10, 10)):
    vecinos = []
    for d in (-paso, paso):
        nuevo = x + d
        if limite[0] <= nuevo <= limite[1]:
            vecinos.append(nuevo)
    return vecinos

# Busqueda ascension de colinas
def ascension_colinas(inicial, iteraciones=20):
    actual = inicial
    mejor = actual
    for _ in range(iteraciones):
        vecinos = generar_vecinos(actual)
        mejor_vecino = max(vecinos, key=funcion_objetivo, default=actual)
        if funcion_objetivo(mejor_vecino) > funcion_objetivo(mejor):
            mejor = mejor_vecino
            actual = mejor_vecino
        else:
            break
    return mejor

if __name__ == '__main__':
    inicio = random.randint(-10, 10)
    resultado = ascension_colinas(inicio)
    print(f'Inicio: {inicio}')
    print(f'Mejor solucion: {resultado}')
    print(f'Valor objetivo: {funcion_objetivo(resultado)}')
