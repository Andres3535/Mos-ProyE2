from __future__ import annotations
from typing import List, Dict, Tuple
import random

from .cvrp_data import InstanciaCVRP
from .cvrp_eval import evaluar_rutas


def permutacion_a_rutas(instancia: InstanciaCVRP, perm: List[str]) -> List[List[str]]:
    """
    Dada una permutación de clientes, construye rutas respetando la capacidad Q
    de manera voraz (greedy): llenamos un vehículo hasta que no quepa más.
    """
    rutas: List[List[str]] = []
    ruta_actual: List[str] = []
    carga_actual = 0.0

    for c in perm:
        demanda = instancia.demandas[c]
        # Si la demanda excede la capacidad restante, abrimos nueva ruta
        if ruta_actual and carga_actual + demanda > instancia.Q + 1e-6:
            rutas.append(ruta_actual)
            ruta_actual = [c]
            carga_actual = demanda
        else:
            ruta_actual.append(c)
            carga_actual += demanda

    if ruta_actual:
        rutas.append(ruta_actual)

    return rutas


def fitness(instancia: InstanciaCVRP, perm: List[str]) -> Tuple[float, Dict[str, float]]:
    """
    Fitness = costo total de las rutas generadas.
    Se penalizan violaciones de capacidad si existieran (safety check).
    """
    rutas = permutacion_a_rutas(instancia, perm)
    stats = evaluar_rutas(instancia, rutas)

    # Usamos las llaves en español definidas en cvrp_eval
    penalidad = 1e6 * stats["violacion_capacidad_max"]
    costo = stats["costo_total"] + penalidad
    return costo, stats

# Operadores GA

def iniciar_poblacion(instancia: InstanciaCVRP, tamano_pob: int, rng: random.Random):
    perm_base = instancia.ids_clientes[:]  # Copia de la lista de clientes
    poblacion = []

    for _ in range(tamano_pob):
        perm = perm_base[:]
        rng.shuffle(perm)
        costo, stats = fitness(instancia, perm)
        poblacion.append({"perm": perm, "costo": costo, "stats": stats})

    mejor = min(poblacion, key=lambda ind: ind["costo"])
    return poblacion, mejor


def seleccion_torneo(poblacion, rng: random.Random, k: int = 3):
    candidatos = rng.sample(poblacion, k)
    return min(candidatos, key=lambda ind: ind["costo"])


def cruce_ox(p1: List[str], p2: List[str], rng: random.Random) -> List[str]:
    n = len(p1)
    a, b = sorted(rng.sample(range(n), 2))
    hijo = [None] * n

    # Copiar segmento de p1
    hijo[a:b+1] = p1[a:b+1]

    # Rellenar con genes de p2 en el orden original
    relleno = [c for c in p2 if c not in hijo]
    idx = 0
    for i in range(n):
        if hijo[i] is None:
            hijo[i] = relleno[idx]
            idx += 1

    return hijo


def mutar(perm: List[str], rng: random.Random,
            prob_intercambio: float = 0.2, prob_inversion: float = 0.1) -> List[str]:
    perm = perm[:]  # copia

    # Swap
    if rng.random() < prob_intercambio:
        i, j = rng.sample(range(len(perm)), 2)
        perm[i], perm[j] = perm[j], perm[i]

    # Reverse
    if rng.random() < prob_inversion and len(perm) >= 3:
        a, b = sorted(rng.sample(range(len(perm)), 2))
        perm[a:b+1] = reversed(perm[a:b+1])

    return perm
# Main GA


def ejecutar_ga(
    instancia: InstanciaCVRP,
    semilla: int = 0,
    tamano_pob: int = 80,
    n_generaciones: int = 400,
    prob_cruce: float = 0.9,
    prob_mutacion: float = 0.3,
    torneo_k: int = 3,
    verbose: bool = True,
):
    rng = random.Random(semilla)

    poblacion, mejor = iniciar_poblacion(instancia, tamano_pob, rng)
    historial_mejor = [mejor["costo"]]

    if verbose:
        print(f"[GA] Inicial: mejor costo = {mejor['costo']:.2f}")

    for gen in range(1, n_generaciones + 1):
        nueva_pob = []

        while len(nueva_pob) < tamano_pob:
            padre1 = seleccion_torneo(poblacion, rng, k=torneo_k)
            padre2 = seleccion_torneo(poblacion, rng, k=torneo_k)

            # Crossover
            if rng.random() < prob_cruce:
                perm_hijo = cruce_ox(padre1["perm"], padre2["perm"], rng)
            else:
                perm_hijo = padre1["perm"][:]

            # Mutación
            if rng.random() < prob_mutacion:
                perm_hijo = mutar(perm_hijo, rng)

            costo, stats = fitness(instancia, perm_hijo)
            nueva_pob.append({"perm": perm_hijo, "costo": costo, "stats": stats})

        poblacion = nueva_pob
        mejor_gen = min(poblacion, key=lambda ind: ind["costo"])

        if mejor_gen["costo"] < mejor["costo"]:
            mejor = mejor_gen

        historial_mejor.append(mejor["costo"])

        if verbose and gen % 20 == 0:
            print(f"[GA] Gen {gen:4d}: mejor costo = {mejor['costo']:.2f}")

    # Construir rutas finales
    mejores_rutas = permutacion_a_rutas(instancia, mejor["perm"])
    return mejor, mejores_rutas, historial_mejor