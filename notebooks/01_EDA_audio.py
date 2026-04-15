"""
SelvaSonic — Exploración del dataset Xeno-canto
Objetivo: Encontrar qué especies amazónicas colombianas tienen
suficientes grabaciones de calidad para entrenar nuestro clasificador.
"""

import requests
import json
import time
from collections import Counter

# Configuración de la API Xeno-canto v3
BASE_URL = "https://xeno-canto.org/api/3/recordings"
API_KEY = "d8010b4e9ec72963e12c5f534d94739299da55ba"

print("✅ Imports listos")

# ============================================
#        Función de consulta (API v3)
# ============================================

def consultar_xenocanto(query: str, page: int = 1) -> dict:
    """
    Hace una consulta a la API v3 de Xeno-canto.
    
    Parámetros:
    - query: string con tags de búsqueda (ej: 'cnt:Colombia')
    - page: número de página de resultados (por defecto 1)
    """
    params = {
        "query": query,
        "key": API_KEY,
        "page": page
    }
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"  ❌ Error {response.status_code}: {response.text}")
        return None

# --- Primera consulta: ¿cuántas grabaciones hay en Colombia? ---
resultado = consultar_xenocanto("cnt:Colombia")

if resultado:
    print(f"🇨🇴 Total de grabaciones de aves en Colombia: {resultado['numRecordings']}")
    print(f"🐦 Total de especies registradas: {resultado['numSpecies']}")
    print(f"📄 Páginas de resultados: {resultado['numPages']}")

# ============================================
# Buscar por departamentos amazónicos usando loc:
# ============================================

departamentos_amazonicos = [
    "Amazonas",
    "Caqueta",
    "Putumayo",
    "Vaupes",
    "Guainia",
    "Guaviare",
]

print("\n🌿 Búsqueda por departamentos amazónicos (tag loc:)")
print("=" * 75)

contador_total = Counter()

for depto in departamentos_amazonicos:
    time.sleep(1)
    query = f'cnt:Colombia loc:{depto}'
    resultado = consultar_xenocanto(query)
    
    if resultado:
        n = int(resultado['numRecordings'])
        print(f"  📍 {depto:15s} → {n:5d} grabaciones")
        
        # Si hay grabaciones, descargamos las primeras páginas para contar especies
        if n > 0:
            paginas = min(int(resultado['numPages']), 5)
            for pag in range(1, paginas + 1):
                time.sleep(1)
                res = consultar_xenocanto(query, page=pag)
                if res and 'recordings' in res:
                    for rec in res['recordings']:
                        nombre = f"{rec['gen']} {rec['sp']}"
                        contador_total[nombre] += 1

print(f"\n📊 Total especies encontradas en la Amazonía: {len(contador_total)}")
print("=" * 75)

print("\n🏆 TOP 30 ESPECIES MÁS GRABADAS EN LA AMAZONÍA COLOMBIANA:")
print(f"   {'#':>3}  {'Especie':40s}  {'Grabaciones':>12}")
print("   " + "-" * 60)

for i, (especie, count) in enumerate(contador_total.most_common(30), 1):
    indicador = "✅" if count >= 10 else "⚠️"
    print(f"   {i:3d}. {indicador} {especie:40s}  {count:>8d}")

# Resumen para decidir
print("\n" + "=" * 75)
aprobadas = [(e, c) for e, c in contador_total.most_common() if c >= 10]
print(f"✅ Especies con ≥10 grabaciones en la Amazonía: {len(aprobadas)}")
print(f"   (Si son pocas, podemos ampliar a toda Colombia)")

# ============================================
# PASO 4 — Especies seleccionadas para SelvaSonic
# ============================================

ESPECIES_SELVASONIC = [
    {"gen": "Crypturellus",  "sp": "cinereus",      "comun": "Tinamú cenizo",           "familia": "Tinamidae"},
    {"gen": "Ramphastos",    "sp": "tucanus",        "comun": "Tucán goliblanco",        "familia": "Ramphastidae"},
    {"gen": "Celeus",        "sp": "grammicus",      "comun": "Carpintero pechipunteado","familia": "Picidae"},
    {"gen": "Glaucidium",    "sp": "brasilianum",    "comun": "Buhíto ferruginoso",      "familia": "Strigidae"},
    {"gen": "Chordeiles",    "sp": "pusillus",       "comun": "Chotacabras menudo",      "familia": "Caprimulgidae"},
    {"gen": "Rupornis",      "sp": "magnirostris",   "comun": "Gavilán caminero",        "familia": "Accipitridae"},
    {"gen": "Frederickena",  "sp": "fulva",          "comun": "Hormiguero crestado",     "familia": "Thamnophilidae"},
    {"gen": "Crypturellus",  "sp": "undulatus",      "comun": "Tinamú ondulado",         "familia": "Tinamidae"},
    {"gen": "Lipaugus",      "sp": "vociferans",     "comun": "Piha gritona",            "familia": "Cotingidae"},
    {"gen": "Trogon",        "sp": "viridis",        "comun": "Trogón coliblanco",       "familia": "Trogonidae"},
]

print("\n\n" + "=" * 75)
print("🦜 ESPECIES SELECCIONADAS PARA SELVASONIC (10 especies, 8 familias)")
print("=" * 75)

for i, esp in enumerate(ESPECIES_SELVASONIC, 1):
    print(f"  {i:2d}. {esp['gen']} {esp['sp']:20s} | {esp['comun']:30s} | {esp['familia']}")

# ============================================
# PASO 5 — Descargar metadatos completos de cada especie
# Ahora sí descargamos TODAS las grabaciones (no solo Amazonía)
# de estas 10 especies en Colombia para maximizar datos
# ============================================

print("\n\n🔽 DESCARGANDO METADATOS COMPLETOS DE LAS 10 ESPECIES EN COLOMBIA...")
print("   (Primero buscamos en Amazonía, si no hay suficientes ampliamos a Colombia)")
print("=" * 75)

import csv
import os

# Crear carpeta para guardar metadatos
os.makedirs("data/metadata", exist_ok=True)

todas_las_grabaciones = []
resumen_por_especie = []

departamentos_amazonicos = ["Amazonas", "Caqueta", "Putumayo", "Vaupes", "Guainia", "Guaviare"]

for esp in ESPECIES_SELVASONIC:
    nombre = f"{esp['gen']} {esp['sp']}"
    print(f"\n  🐦 {nombre} ({esp['comun']})...")
    
    grabaciones_especie = []
    
    # Buscar en cada departamento amazónico
    for depto in departamentos_amazonicos:
        time.sleep(1)
        query = f"gen:{esp['gen']} sp:{esp['sp']} cnt:Colombia loc:{depto}"
        resultado = consultar_xenocanto(query)
        
        if resultado and int(resultado['numRecordings']) > 0:
            # Descargar todas las páginas de esta especie en este depto
            total_paginas = int(resultado['numPages'])
            for pag in range(1, total_paginas + 1):
                if pag > 1:
                    time.sleep(1)
                    resultado = consultar_xenocanto(query, page=pag)
                
                if resultado and 'recordings' in resultado:
                    for rec in resultado['recordings']:
                        grabaciones_especie.append({
                            'id': rec['id'],
                            'gen': rec['gen'],
                            'sp': rec['sp'],
                            'en': rec.get('en', ''),
                            'cnt': rec['cnt'],
                            'loc': rec.get('loc', ''),
                            'lat': rec.get('lat', ''),
                            'lng': rec.get('lng', ''),
                            'type': rec.get('type', ''),
                            'q': rec.get('q', ''),
                            'length': rec.get('length', ''),
                            'file': rec.get('file', ''),
                            'date': rec.get('date', ''),
                            'familia': esp['familia'],
                            'nombre_comun': esp['comun'],
                            'region': 'Amazonia',
                        })
    
    n_amazonia = len(grabaciones_especie)
    
    # Si hay menos de 20 grabaciones en Amazonía, ampliamos a toda Colombia
    if n_amazonia < 20:
        print(f"     ⚠️ Solo {n_amazonia} en Amazonía, ampliando a toda Colombia...")
        time.sleep(1)
        query = f"gen:{esp['gen']} sp:{esp['sp']} cnt:Colombia"
        resultado = consultar_xenocanto(query)
        
        if resultado and int(resultado['numRecordings']) > 0:
            ids_existentes = {r['id'] for r in grabaciones_especie}
            total_paginas = min(int(resultado['numPages']), 10)
            
            for pag in range(1, total_paginas + 1):
                if pag > 1:
                    time.sleep(1)
                    resultado = consultar_xenocanto(query, page=pag)
                
                if resultado and 'recordings' in resultado:
                    for rec in resultado['recordings']:
                        if rec['id'] not in ids_existentes:
                            grabaciones_especie.append({
                                'id': rec['id'],
                                'gen': rec['gen'],
                                'sp': rec['sp'],
                                'en': rec.get('en', ''),
                                'cnt': rec['cnt'],
                                'loc': rec.get('loc', ''),
                                'lat': rec.get('lat', ''),
                                'lng': rec.get('lng', ''),
                                'type': rec.get('type', ''),
                                'q': rec.get('q', ''),
                                'length': rec.get('length', ''),
                                'file': rec.get('file', ''),
                                'date': rec.get('date', ''),
                                'familia': esp['familia'],
                                'nombre_comun': esp['comun'],
                                'region': 'Colombia_otro',
                            })
    
    n_total = len(grabaciones_especie)
    print(f"     ✅ Total: {n_total} grabaciones ({n_amazonia} Amazonía + {n_total - n_amazonia} resto Colombia)")
    
    resumen_por_especie.append({
        'especie': nombre,
        'comun': esp['comun'],
        'familia': esp['familia'],
        'amazonia': n_amazonia,
        'colombia_total': n_total,
    })
    
    todas_las_grabaciones.extend(grabaciones_especie)

# ============================================
# Guardar metadata.csv
# ============================================

metadata_path = "data/metadata.csv"
if todas_las_grabaciones:
    campos = todas_las_grabaciones[0].keys()
    with open(metadata_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(todas_las_grabaciones)
    print(f"\n💾 Metadatos guardados en: {metadata_path}")
    print(f"   Total grabaciones: {len(todas_las_grabaciones)}")

# ============================================
# RESUMEN FINAL
# ============================================

print("\n" + "=" * 75)
print("📊 RESUMEN FINAL DEL DATASET SELVASONIC")
print("=" * 75)
print(f"   {'Especie':35s} {'Familia':20s} {'Amazonía':>10} {'Total':>8}")
print("   " + "-" * 75)
for r in resumen_por_especie:
    print(f"   {r['especie']:35s} {r['familia']:20s} {r['amazonia']:>10} {r['colombia_total']:>8}")
print("   " + "-" * 75)
print(f"   {'TOTAL':35s} {'':20s} {sum(r['amazonia'] for r in resumen_por_especie):>10} {sum(r['colombia_total'] for r in resumen_por_especie):>8}")