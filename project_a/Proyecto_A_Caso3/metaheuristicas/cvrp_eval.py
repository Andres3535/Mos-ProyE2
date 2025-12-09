from typing import List, Dict
from .cvrp_data import InstanciaCVRP
import math

def evaluar_rutas(instancia: InstanciaCVRP, rutas: List[List[str]]):
    id_deposito = instancia.id_deposito
    indices = instancia.indice_nodo
    matriz_dist = instancia.matriz_dist

    distancia_total = 0.0
    tiempo_total_horas = 0.0
    costo_combustible_total = 0.0
    costo_dist_total = 0.0
    costo_tiempo_total = 0.0
    costo_fijo_total = 0.0
    violacion_max = 0.0

    for clientes_ruta in rutas:
        if not clientes_ruta:
            continue

        # Calcular carga y verificar violaciones de capacidad
        carga = sum(instancia.demandas[c] for c in clientes_ruta)
        violacion = max(0.0, carga - instancia.Q)
        violacion_max = max(violacion_max, violacion)

        # Calcular distancia
        secuencia = [id_deposito] + clientes_ruta + [id_deposito]
        dist_ruta = 0.0
        for i in range(len(secuencia) - 1):
            nodo_a = secuencia[i]
            nodo_b = secuencia[i + 1]
            dist_ruta += matriz_dist[indices[nodo_a], indices[nodo_b]]

        tiempo_ruta = dist_ruta / instancia.velocidad_kmh

        # Cálculo de costos individuales
        consumo_litros = dist_ruta / instancia.efic_km_litro
        costo_combustible = consumo_litros * instancia.precio_combustible

        costo_dist = instancia.C_dist * dist_ruta
        costo_tiempo = instancia.C_tiempo * tiempo_ruta
        costo_fijo = instancia.C_fijo  # Costo por usar el vehículo

        # Acumular totales
        distancia_total += dist_ruta
        tiempo_total_horas += tiempo_ruta
        costo_combustible_total += costo_combustible
        costo_dist_total += costo_dist
        costo_tiempo_total += costo_tiempo
        costo_fijo_total += costo_fijo

    costo_total = (costo_fijo_total + costo_dist_total + costo_tiempo_total + costo_combustible_total)

    return {
        "distancia_total_km": distancia_total,
        "tiempo_total_horas": tiempo_total_horas,
        "costo_combustible_total": costo_combustible_total,
        "costo_fijo_total": costo_fijo_total,
        "costo_dist_total": costo_dist_total,
        "costo_tiempo_total": costo_tiempo_total,
        "costo_total": costo_total,
        "violacion_capacidad_max": violacion_max, # 0 si es factible
    }