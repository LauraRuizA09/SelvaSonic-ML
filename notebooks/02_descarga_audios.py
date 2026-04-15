"""
SelvaSonic — Paso 6: Descarga de audios desde Xeno-canto
Lee el archivo metadata.csv y descarga cada MP3 en carpetas organizadas
por especie dentro de data/raw/
"""

import os
import csv
import time
import requests

# ============================================
# CONFIGURACIÓN
# ============================================

METADATA_PATH = "data/metadata.csv"
RAW_DIR = "data/raw"
DESCARGA_PAUSA = 1  # segundos entre descargas para no saturar el servidor

# ============================================
# PASO 1 — Leer el metadata.csv
# ============================================

print("📂 Leyendo metadata.csv...")

grabaciones = []
with open(METADATA_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        grabaciones.append(row)

print(f"   Encontradas {len(grabaciones)} grabaciones para descargar")

# Contar por especie
from collections import Counter
conteo = Counter(f"{r['gen']} {r['sp']}" for r in grabaciones)
print("\n   Distribución por especie:")
for especie, n in conteo.most_common():
    print(f"     {especie:35s} → {n:4d} audios")

# ============================================
# PASO 2 — Crear carpetas por especie
# ============================================

print(f"\n📁 Creando estructura de carpetas en {RAW_DIR}/...")

especies_carpetas = set()
for rec in grabaciones:
    nombre_carpeta = f"{rec['gen']}_{rec['sp']}"
    carpeta = os.path.join(RAW_DIR, nombre_carpeta)
    os.makedirs(carpeta, exist_ok=True)
    especies_carpetas.add(nombre_carpeta)

print(f"   Creadas {len(especies_carpetas)} carpetas:")
for c in sorted(especies_carpetas):
    print(f"     📂 {RAW_DIR}/{c}/")

# ============================================
# PASO 3 — Descargar audios MP3
# ============================================

print(f"\n🔽 INICIANDO DESCARGA DE {len(grabaciones)} AUDIOS...")
print("   (Esto puede tomar varios minutos según tu conexión)")
print("=" * 60)

descargados = 0
errores = 0
ya_existentes = 0

for i, rec in enumerate(grabaciones, 1):
    nombre_carpeta = f"{rec['gen']}_{rec['sp']}"
    nombre_archivo = f"XC{rec['id']}.mp3"
    ruta_destino = os.path.join(RAW_DIR, nombre_carpeta, nombre_archivo)
    
    # Si ya existe, no lo descargamos de nuevo
    if os.path.exists(ruta_destino):
        ya_existentes += 1
        continue
    
    # Construir URL de descarga
    url = rec['file']
    if url.startswith("//"):
        url = "https:" + url
    
    # Progreso
    especie_corta = f"{rec['gen']} {rec['sp']}"
    print(f"  [{i:3d}/{len(grabaciones)}] {especie_corta:30s} → {nombre_archivo}...", end=" ")
    
    try:
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            with open(ruta_destino, 'wb') as f:
                f.write(response.content)
            
            # Tamaño del archivo en KB
            tamano_kb = len(response.content) / 1024
            print(f"✅ ({tamano_kb:.0f} KB)")
            descargados += 1
        else:
            print(f"❌ Error HTTP {response.status_code}")
            errores += 1
    except requests.exceptions.Timeout:
        print("❌ Timeout")
        errores += 1
    except Exception as e:
        print(f"❌ {str(e)[:50]}")
        errores += 1
    
    # Pausa entre descargas
    time.sleep(DESCARGA_PAUSA)

# ============================================
# RESUMEN DE DESCARGA
# ============================================

print("\n" + "=" * 60)
print("📊 RESUMEN DE DESCARGA")
print("=" * 60)
print(f"  ✅ Descargados exitosamente: {descargados}")
print(f"  ⏭️  Ya existían (omitidos):   {ya_existentes}")
print(f"  ❌ Errores:                   {errores}")
print(f"  📁 Ubicación: {RAW_DIR}/")

# Mostrar contenido final de cada carpeta
print("\n📂 CONTENIDO FINAL:")
for carpeta in sorted(especies_carpetas):
    ruta = os.path.join(RAW_DIR, carpeta)
    archivos = [f for f in os.listdir(ruta) if f.endswith('.mp3')]
    print(f"  {carpeta:35s} → {len(archivos):4d} archivos MP3")

total_mp3 = sum(
    len([f for f in os.listdir(os.path.join(RAW_DIR, c)) if f.endswith('.mp3')])
    for c in especies_carpetas
)
print(f"\n🎵 Total archivos MP3 descargados: {total_mp3}")
print("🦜 ¡Dataset listo para preprocesamiento!")