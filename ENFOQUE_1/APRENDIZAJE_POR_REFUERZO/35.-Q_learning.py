"""
Q-Learning - Aprendizaje por Refuerzo
GridWorld 4x4: S=inicio, G=meta(+1), #=obstáculo(-1), .=paso(-0.1)
  S . . .
  . # . .
  . . . .
  . . . G

Muestra la tabla Q y la evolución del aprendizaje.
"""
import numpy as np

# ── Entorno ──────────────────────────────────────────
GRID = {
    (0,0):"S",(0,1):".",(0,2):".",(0,3):".",
    (1,0):".",(1,1):"#",(1,2):".",(1,3):".",
    (2,0):".",(2,1):".",(2,2):".",(2,3):".",
    (3,0):".",(3,1):".",(3,2):".",(3,3):"G",
}
R    = {"G":+1.0, "#":-1.0, ".": -0.1, "S":-0.1}
MOVS = [(-1,0),(1,0),(0,-1),(0,1)]
FLECHAS = ["↑","↓","←","→"]

def mover(s, a):
    ns = (s[0]+MOVS[a][0], s[1]+MOVS[a][1])
    ns = ns if ns in GRID else s
    return ns, R[GRID[ns]]

# ── Tabla Q ──────────────────────────────────────────
Q = {s: np.zeros(4) for s in GRID}
alpha, gamma = 0.2, 0.9
epsilon = 1.0          # exploración inicial máxima

# ── Entrenamiento ────────────────────────────────────
print("Entrenando...\n")
for ep in range(1, 601):
    s = (0, 0)
    pasos, recompensa_total = 0, 0

    while GRID[s] not in ("G", "#") and pasos < 50:
        # ε-greedy
        a = np.random.randint(4) if np.random.rand() < epsilon else np.argmax(Q[s])
        ns, r = mover(s, a)

        # Actualización Q-Learning
        Q[s][a] += alpha * (r + gamma * np.max(Q[ns]) - Q[s][a])

        recompensa_total += r
        s, pasos = ns, pasos + 1

    epsilon = max(0.05, epsilon * 0.99)   # decaimiento de exploración

    if ep % 200 == 0:
        print(f"  Episodio {ep:>4} | ε={epsilon:.3f} | Recompensa: {recompensa_total:+.2f}")

# ── Resultados ───────────────────────────────────────
print("\n=== TABLA Q (mejor acción por estado) ===\n")
print("     C0   C1   C2   C3")
for r in range(4):
    fila = f"F{r} "
    for c in range(4):
        s = (r, c)
        celda = GRID[s] if GRID[s] in ("G","#") else FLECHAS[np.argmax(Q[s])]
        fila += f"  {celda:^3}"
    print(fila)

print("\n=== POLÍTICA ÓPTIMA (valores Q máximos) ===\n")
print("       C0      C1      C2      C3")
for r in range(4):
    fila = f"F{r} "
    for c in range(4):
        fila += f"  {max(Q[(r,c)]):+.3f}"
    print(fila)

print("\n=== RUTA DEL AGENTE (política greedy) ===")
s, ruta, total = (0,0), [(0,0)], 0
while GRID[s] not in ("G","#") and len(ruta) < 20:
    a = np.argmax(Q[s])
    s, r = mover(s, a)
    ruta.append(s); total += r
print(f"  {'→'.join(str(x) for x in ruta)}")
print(f"  Resultado: {'META ✓' if GRID[s]=='G' else 'OBSTÁCULO ✗'} | Recompensa: {total:+.2f}")