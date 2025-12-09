from dataclasses import dataclass
from typing import Dict, List, Tuple
import os
import math
import pandas as pd
import numpy as np

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

@dataclass
class InstanciaCVRP:
    nombre: str
    id_deposito: str
    ids_clientes: List[str]
    ids_nodos: List[str]            
    coords: Dict[str, Tuple[float, float]]  
    demandas: Dict[str, float]     
    Q: float                        
    C_fijo: float
    C_dist: float
    C_tiempo: float
    precio_combustible: float
    efic_km_litro: float
    velocidad_kmh: float
    matriz_dist: np.ndarray      
    indice_nodo: Dict[str, int]      

def cargar_parametros(ruta_params: str):
    df = pd.read_csv(ruta_params)
    mapa_params = dict(zip(df["Parameter"], df["Value"]))

    C_fijo = float(mapa_params["C_fixed"])
    C_dist = float(mapa_params["C_dist"])
    C_tiempo = float(mapa_params["C_time"])
    precio_combustible = float(mapa_params["fuel_price"])

    # Promedio de eficiencias si existen varias definidas
    llaves_efic = [k for k in mapa_params.keys() if "fuel_efficiency" in k]
    if llaves_efic:
        valores_efic = [mapa_params[k] for k in llaves_efic]
        efic_km_litro = float(sum(valores_efic) / len(valores_efic))
    else:
        efic_km_litro = 8.0 # Valor por defecto

    return {
        "C_fijo": C_fijo,
        "C_dist": C_dist,
        "C_tiempo": C_tiempo,
        "precio_combustible": precio_combustible,
        "efic_km_litro": efic_km_litro,
    }

def elegir_capacidad_homogenea(df_vehiculos: pd.DataFrame):
    caps = df_vehiculos["Capacity"].values
    if len(caps) == 0:
        raise ValueError("No hay vehículos en el archivo CSV.")

    vals, counts = np.unique(caps, return_counts=True)
    cap_moda = vals[np.argmax(counts)]
    return float(cap_moda)

def cargar_instancia_cvrp_caso3(raiz_proyecto: str = ".",dir_caso_base: str = "Proyecto_Caso_Base",dir_caso3: str = "Proyecto_A_Caso3",velocidad_kmh: float = 25.0):
    """
    Carga la instancia CVRP simplificada para el Caso 3, combinando:
    Depósito del Caso Base.
    Clientes, vehículos y parámetros del Caso 3.
    """
    ruta_base = os.path.join(raiz_proyecto, dir_caso_base)
    ruta_caso3 = os.path.join(raiz_proyecto, dir_caso3)

    # Cargar deposito
    df_depositos = pd.read_csv(os.path.join(ruta_base, "depots.csv"))
    fila_deposito = df_depositos.iloc[0] # Tomamos el primero
    id_deposito = fila_deposito["StandardizedID"]
    lat_dep = fila_deposito["Latitude"]
    lon_dep = fila_deposito["Longitude"]

    # Cargar clientes y vehiculos
    df_clientes = pd.read_csv(os.path.join(ruta_caso3, "clients.csv"))
    df_vehiculos = pd.read_csv(os.path.join(ruta_caso3, "vehicles.csv"))
    ruta_params = os.path.join(ruta_caso3, "parameters_urban.csv")

    df_clientes = df_clientes.sort_values("StandardizedID")
    ids_clientes = df_clientes["StandardizedID"].tolist()
    demandas = dict(zip(df_clientes["StandardizedID"], df_clientes["Demand"]))

    # Construcción de coordenadas
    coords: Dict[str, Tuple[float, float]] = {}
    coords[id_deposito] = (float(lat_dep), float(lon_dep))
    for _, fila in df_clientes.iterrows():
        coords[fila["StandardizedID"]] = (float(fila["Latitude"]), float(fila["Longitude"]))

    # Definir capacidad Q y parámetros
    Q = elegir_capacidad_homogenea(df_vehiculos)
    p = cargar_parametros(ruta_params)

    # Matriz de distancia
    ids_nodos = [id_deposito] + ids_clientes
    n = len(ids_nodos)
    indice_nodo = {nid: i for i, nid in enumerate(ids_nodos)}
    matriz_dist = np.zeros((n, n), dtype=float)

    for i, nid_i in enumerate(ids_nodos):
        lat_i, lon_i = coords[nid_i]
        for j, nid_j in enumerate(ids_nodos):
            if i == j:
                matriz_dist[i, j] = 0.0
            else:
                lat_j, lon_j = coords[nid_j]
                matriz_dist[i, j] = haversine_km(lat_i, lon_i, lat_j, lon_j)

    instancia = InstanciaCVRP(
        nombre="Proyecto_A_Caso3_CVRP_simplificado",
        id_deposito=id_deposito,
        ids_clientes=ids_clientes,
        ids_nodos=ids_nodos,
        coords=coords,
        demandas=demandas,
        Q=Q,
        C_fijo=p["C_fijo"],
        C_dist=p["C_dist"],
        C_tiempo=p["C_tiempo"],
        precio_combustible=p["precio_combustible"],
        efic_km_litro=p["efic_km_litro"],
        velocidad_kmh=velocidad_kmh,
        matriz_dist=matriz_dist,
        indice_nodo=indice_nodo,
    )

    return instancia