"""
Aprendizaje por Refuerzo Pasivo - TD(0)
GridWorld 3x3: S=inicio, G=meta(+1), #=obstáculo(-1), .=paso(-0.1)
  S . .
  . # .
  . . G
"""
import numpy as np

# Entorno
GRID  = {(0,0):"S",(0,1):".",(0,2):".",
         (1,0):".",(1,1):"#",(1,2):".",
         (2,0):".",(2,1):".",(2,2):"G"}
R     = {"G": 1.0, "#": -1.0, ".": -0.1, "S": -0.1}
MOVS  = [(-1,0),(1,0),(0,-1),(0,1)]

def mover(s, a):
    ns = (s[0]+a[0], s[1]+a[1])
    ns = ns if ns in GRID else s
    return ns, R[GRID[ns]]

def politica(s):
    """Política fija: moverse hacia (2,2)."""
    opciones = [(1,0) if s[0]<2 else (-1,0), (0,1) if s[1]<2 else (0,-1)]
    return opciones[np.random.randint(2)]

# TD(0): V(s) ← V(s) + α[r + γV(s') - V(s)]
V = {s: 0.0 for s in GRID}
alpha, gamma = 0.1, 0.9

for ep in range(300):
    s = (0, 0)
    while GRID[s] not in ("G", "#"):
        a       = politica(s)
        ns, r   = mover(s, a)
        V[s]   += alpha * (r + gamma * V[ns] - V[s])
        s       = ns

# Mostrar resultados
print("Valores aprendidos V(s):\n")
for r in range(3):
    print(" ".join(f"{V[(r,c)]:+.2f}" for c in range(3)))