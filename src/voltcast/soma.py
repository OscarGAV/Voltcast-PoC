"""SOMA (Self-Organizing Migrating Algorithm), estrategia All-To-One, para optimizar hiperparámetros (Propuesta A).

Cada individuo es un vector en [0, 1]^d; ``decode`` lo traduce a hiperparámetros. En cada migración, todos los
individuos "viajan" hacia el líder (el mejor) por pasos ``step`` hasta ``path_length``, perturbando solo las
dimensiones elegidas al azar con probabilidad ``prt``, y se quedan con la mejor posición encontrada.
"""
from __future__ import annotations

import time

import numpy as np


def soma(objective, decode, dim: int, pop: int = 6, migrations: int = 3, path_length: float = 2.0,
         step: float = 0.4, prt: float = 0.3, seed: int = 42, log=print) -> dict:
    """Minimiza ``objective(decode(x))``. Devuelve el mejor hiperparámetro, su valor y el historial de evaluaciones."""
    rng = np.random.default_rng(seed)
    hist = []

    def evaluar(x, migracion):
        params = decode(x)
        t = time.time()
        f = float(objective(params))
        hist.append({"migracion": migracion, **params, "objetivo": f, "segundos": time.time() - t})
        if log:
            log(f"  mig {migracion} | {params} → {f:.5f} ({time.time() - t:.0f} s)")
        return f

    X = rng.random((pop, dim))
    F = np.array([evaluar(x, 0) for x in X])
    for m in range(1, migrations + 1):
        lider = int(np.argmin(F))
        for i in range(pop):
            if i == lider:
                continue
            mejor_x, mejor_f = X[i].copy(), F[i]
            for t in np.arange(step, path_length + 1e-9, step):
                perturb = rng.random(dim) < prt
                if not perturb.any():
                    perturb[rng.integers(dim)] = True
                x = np.clip(X[i] + (X[lider] - X[i]) * t * perturb, 0, 1)
                f = evaluar(x, m)
                if f < mejor_f:
                    mejor_x, mejor_f = x, f
            X[i], F[i] = mejor_x, mejor_f
    b = int(np.argmin(F))
    return {"mejor": decode(X[b]), "objetivo": float(F[b]), "historial": hist, "evaluaciones": len(hist)}
