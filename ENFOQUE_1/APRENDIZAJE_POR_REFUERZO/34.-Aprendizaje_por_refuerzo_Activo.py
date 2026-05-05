"""
Aprendizaje por Refuerzo Activo - Q-Learning
GridWorld 3x3: S=inicio, G=meta(+1), #=obstáculo(-1), .=paso(-0.1)
  S . .
  . # .
  . . G
El agente ELIGE sus acciones (ε-greedy) y aprende la política óptima.
"""
import numpy as np

# Entorno
GRID = {(0,0):"S",(0,1):".",(0,2):".",
        (1,0):".",(1,1):"#",(1,2):".",
        (2,0):".",(2,1):".",(2,2):"G"}
R    = {"G": 1.0, "#": -1.0, ".": -0.1, "S": -0.1}
MOVS = [(-1,0),(1,0),(0,-1),(0,1)]

def mover(s, a):
    ns = (s[0]+a[0], s[1]+a[1])
    ns = ns if ns in GRID else s
    return ns, R[GRID[ns]]

# Q-Learning: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
Q     = {s: [0.0]*4 for s in GRID}
alpha, gamma, epsilon = 0.1, 0.9, 0.3

for ep in range(500):
    s = (0, 0)
    while GRID[s] not in ("G", "#"):
        # Política ε-greedy: explorar o explotar
        a_idx = np.random.randint(4) if np.random.rand() < epsilon else np.argmax(Q[s])
        ns, r = mover(s, MOVS[a_idx])

        # Actualización Q-Learning
        Q[s][a_idx] += alpha * (r + gamma * max(Q[ns]) - Q[s][a_idx])
        s = ns

    epsilon = max(0.01, epsilon * 0.995)  # reducir exploración

# Mostrar política óptima aprendida
FLECHAS = ["↑","↓","←","→"]
print("Política óptima aprendida:\n")
for r in range(3):
    fila = ""
    for c in range(3):
        s = (r, c)
        fila += f" {GRID[s] if GRID[s] in ('G','#') else FLECHAS[np.argmax(Q[s])]}"
    print(fila)

print("\nValores Q máximos por estado:\n")
for r in range(3):
    print(" ".join(f"{max(Q[(r,c)]):+.2f}" for c in range(3)))