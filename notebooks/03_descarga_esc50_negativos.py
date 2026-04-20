"""
03_descarga_esc50_negativos.py — SelvaSonic: Descarga de dataset ESC-50 (clase negativa)

Descarga el dataset ESC-50 (Environmental Sound Classification) y filtra las categorías
que NO contienen sonidos de animales, para usarlas como clase negativa ("No ave") en el
clasificador bioacústico de especies amazónicas.

¿POR QUÉ NECESITAMOS NEGATIVOS?
─────────────────────────────────
Sin una clase negativa, el modelo SIEMPRE clasificará cualquier audio como alguna de
las 10 especies de aves — incluso si es lluvia, un motor o música. La clase negativa
le enseña al modelo la frontera: "esto NO es una vocalización de ave amazónica".

¿POR QUÉ ESC-50?
──────────────────
- 2000 clips de 5 segundos en formato WAV (44.1 kHz, mono)
- 50 categorías bien etiquetadas con 40 ejemplos cada una
- Ya está segmentado en clips de 5s (mismo formato que usaremos para Xeno-canto)
- Licencia Creative Commons Attribution Non-Commercial

¿QUÉ CATEGORÍAS USAMOS?
─────────────────────────
Excluimos la categoría "Animals" (sonidos de animales como perros, gallos, ranas)
porque podrían confundir al modelo al ser sonidos bioacústicos similares a los de aves.
Usamos las otras 4 categorías:
  - Natural soundscapes & water sounds (lluvia, mar, viento, etc.)
  - Human non-speech sounds (tos, estornudo, aplausos, etc.)
  - Interior/domestic sounds (reloj, puerta, teclado, etc.)
  - Exterior/urban noises (helicóptero, sirena, motor, etc.)

"""

# ──────────────────────────────────────────────────────────────────────
#                                Imports
# ──────────────────────────────────────────────────────────────────────
import os
import sys
import zipfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# Para descarga
import urllib.request

# ──────────────────────────────────────────────────────────────────────
#                              Configuración
# ──────────────────────────────────────────────────────────────────────

# URL oficial del dataset ESC-50 (release en GitHub)
ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"

# Directorio base del proyecto (ajustar según tu máquina)
PROJECT_ROOT = Path(".")  # Cambiar a la ruta de tu repo si es necesario

# Directorios de datos
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ESC50_DIR = RAW_DIR / "ESC-50"             # Audios negativos filtrados
ESC50_DOWNLOAD_DIR = DATA_DIR / "_temp_esc50"  # Temporal para descarga

# Archivo de metadata
METADATA_NEGATIVOS = DATA_DIR / "metadata_negativos.csv"

# ──────────────────────────────────────────────────────────────────────
# Categorías de ESC-50
# ──────────────────────────────────────────────────────────────────────
# ESC-50 organiza sus 50 clases en 5 categorías principales.
# Cada clase tiene un ID numérico (0-49) y pertenece a una categoría.
#
# Categoría 1: Animals (IDs 0-9)        ← EXCLUIR (son sonidos de animales)
# Categoría 2: Natural soundscapes (IDs 10-19) ← INCLUIR
# Categoría 3: Human non-speech (IDs 20-29)    ← INCLUIR
# Categoría 4: Interior/domestic (IDs 30-39)   ← INCLUIR
# Categoría 5: Exterior/urban (IDs 40-49)      ← INCLUIR

# Clases de ANIMALES que EXCLUIMOS (categoría 1)
ANIMAL_CLASSES = {
    "dog", "rooster", "pig", "cow", "frog",
    "cat", "hen", "insects", "sheep", "crow"
}

# Número máximo de clips negativos a usar (para balancear con los positivos)
# Tenemos ~334 clips positivos (aves), así que no queremos un desbalance extremo.
# Estrategia: mantener todos los negativos disponibles (~1600) y después
# aplicar class weights o undersampling durante el entrenamiento.
MAX_NEGATIVE_CLIPS: Optional[int] = None  # None = usar todos


def descargar_esc50() -> Path:
    """
    Descarga el dataset ESC-50 desde GitHub como archivo ZIP.
    
    Returns:
        Path al directorio descomprimido con los datos.
    """
    zip_path = ESC50_DOWNLOAD_DIR / "ESC-50-master.zip"
    extracted_dir = ESC50_DOWNLOAD_DIR / "ESC-50-master"
    
    # Si ya existe el directorio extraído, no volver a descargar
    if extracted_dir.exists():
        print(f"✅ ESC-50 ya descargado en: {extracted_dir}")
        return extracted_dir
    
    # Crear directorio temporal
    ESC50_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    print("📥 Descargando ESC-50 (~600 MB)...")
    print(f"   URL: {ESC50_URL}")
    print("   Esto puede tomar unos minutos dependiendo de tu conexión...\n")
    
    # Descarga con barra de progreso simple
    def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            bar_len = 40
            filled = int(bar_len * pct / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r   [{bar}] {pct:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(ESC50_URL, str(zip_path), reporthook=_progress_hook)
        print("\n\n✅ Descarga completada!")
    except Exception as e:
        print(f"\n❌ Error en la descarga: {e}")
        print("   Alternativa: descarga manualmente desde https://github.com/karolpiczak/ESC-50")
        print("   y coloca el ZIP en:", zip_path)
        sys.exit(1)
    
    # Descomprimir
    print("📦 Descomprimiendo...")
    with zipfile.ZipFile(str(zip_path), 'r') as zf:
        zf.extractall(str(ESC50_DOWNLOAD_DIR))
    
    print(f"✅ Descomprimido en: {extracted_dir}")
    
    # Eliminar el ZIP para ahorrar espacio
    zip_path.unlink()
    print("🗑️  ZIP eliminado para ahorrar espacio.\n")
    
    return extracted_dir


def cargar_metadata_esc50(esc50_dir: Path) -> pd.DataFrame:
    """
    Carga y muestra el archivo de metadata de ESC-50.
    
    El archivo esc50.csv tiene las siguientes columnas:
    - filename: nombre del archivo de audio (e.g., "1-100032-A-0.ogg")
    - fold: fold de cross-validation (1-5)
    - target: ID numérico de la clase (0-49)
    - category: nombre de la clase (e.g., "dog", "rain", "siren")
    - esc10: si pertenece al subconjunto ESC-10 (True/False)
    - src_file: ID del archivo fuente original en Freesound
    - take: toma de la grabación
    
    Returns:
        DataFrame con la metadata completa.
    """
    meta_path = esc50_dir / "meta" / "esc50.csv"
    
    if not meta_path.exists():
        print(f"❌ No se encontró el archivo de metadata en: {meta_path}")
        sys.exit(1)
    
    df = pd.read_csv(meta_path)
    
    print("=" * 65)
    print("📊 METADATA ESC-50 — RESUMEN")
    print("=" * 65)
    print(f"   Total de clips: {len(df)}")
    print(f"   Total de categorías: {df['category'].nunique()}")
    print(f"   Clips por categoría: {df.groupby('category').size().iloc[0]}")
    print(f"   Folds de cross-validation: {df['fold'].nunique()}")
    print(f"   Columnas: {list(df.columns)}")
    print()
    
    # Mostrar todas las categorías organizadas por grupo
    print("📋 CATEGORÍAS DE ESC-50 (50 clases en 5 grupos):")
    print("-" * 65)
    
    # Definir grupos manualmente (ESC-50 los organiza por rango de target ID)
    grupos = {
        "🐾 Animals (IDs 0-9) — EXCLUIR": df[df['target'].between(0, 9)],
        "🌊 Natural soundscapes (IDs 10-19) — INCLUIR": df[df['target'].between(10, 19)],
        "🗣️ Human non-speech (IDs 20-29) — INCLUIR": df[df['target'].between(20, 29)],
        "🏠 Interior/domestic (IDs 30-39) — INCLUIR": df[df['target'].between(30, 39)],
        "🏙️ Exterior/urban (IDs 40-49) — INCLUIR": df[df['target'].between(40, 49)],
    }
    
    for grupo_nombre, grupo_df in grupos.items():
        categorias = sorted(grupo_df['category'].unique())
        n_clips = len(grupo_df)
        print(f"\n   {grupo_nombre}")
        print(f"   Categorías ({len(categorias)}): {', '.join(categorias)}")
        print(f"   Clips: {n_clips}")
    
    print()
    return df


def filtrar_negativos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra el DataFrame para quedarnos SOLO con las categorías que NO son animales.
    
    Lógica de filtrado:
    ──────────────────
    - Las clases con target ID 0-9 son sonidos de animales → EXCLUIR
    - Las clases con target ID 10-49 son sonidos no-animales → INCLUIR
    
    Alternativa equivalente: excluir por nombre de categoría (más explícito).
    
    Returns:
        DataFrame filtrado con solo los negativos.
    """
    print("=" * 65)
    print("🔍 FILTRADO DE NEGATIVOS")
    print("=" * 65)
    
    # Método 1: Filtrar por nombre de categoría (más legible)
    df_negativos = df[~df['category'].isin(ANIMAL_CLASSES)].copy()
    
    # Verificación: confirmar que no quedan animales
    categorias_restantes = set(df_negativos['category'].unique())
    animales_filtrados = categorias_restantes.intersection(ANIMAL_CLASSES)
    
    if animales_filtrados:
        print(f"⚠️  ADVERTENCIA: Aún quedan categorías de animales: {animales_filtrados}")
    else:
        print("✅ Todas las categorías de animales han sido excluidas correctamente.")
    
    print(f"\n   Clips originales (ESC-50 completo): {len(df)}")
    print(f"   Clips excluidos (animales):          {len(df) - len(df_negativos)}")
    print(f"   Clips negativos resultantes:         {len(df_negativos)}")
    print(f"   Categorías negativas:                {df_negativos['category'].nunique()}")
    
    # Limitar si se configuró MAX_NEGATIVE_CLIPS
    if MAX_NEGATIVE_CLIPS is not None and len(df_negativos) > MAX_NEGATIVE_CLIPS:
        print(f"\n   ⚠️  Limitando a {MAX_NEGATIVE_CLIPS} clips (configuración)")
        # Muestreo estratificado para mantener diversidad de categorías
        df_negativos = df_negativos.groupby('category', group_keys=False).apply(
            lambda x: x.sample(
                n=min(len(x), MAX_NEGATIVE_CLIPS // df_negativos['category'].nunique()),
                random_state=42
            )
        )
        print(f"   Clips después del muestreo: {len(df_negativos)}")
    
    print(f"\n📋 Categorías incluidas como negativos:")
    for cat in sorted(df_negativos['category'].unique()):
        n = len(df_negativos[df_negativos['category'] == cat])
        print(f"      • {cat}: {n} clips")
    
    print()
    return df_negativos


def copiar_audios_negativos(
    df_negativos: pd.DataFrame,
    esc50_dir: Path
) -> pd.DataFrame:
    """
    Copia los archivos de audio filtrados al directorio del proyecto,
    organizados en la carpeta data/raw/ESC-50/.
    
    Estructura resultante:
        data/raw/ESC-50/
        ├── rain/
        │   ├── 1-17367-A-10.wav
        │   └── ...
        ├── siren/
        │   ├── 1-24074-A-43.wav
        │   └── ...
        └── ...
    
    Returns:
        DataFrame con la metadata actualizada (rutas locales añadidas).
    """
    audio_src_dir = esc50_dir / "audio"
    
    if not audio_src_dir.exists():
        print(f"❌ No se encontró el directorio de audio en: {audio_src_dir}")
        sys.exit(1)
    
    print("=" * 65)
    print("📁 COPIANDO AUDIOS NEGATIVOS AL PROYECTO")
    print("=" * 65)
    
    # Crear directorio destino
    ESC50_DIR.mkdir(parents=True, exist_ok=True)
    
    rutas_locales = []
    copiados = 0
    errores = 0
    
    for _, row in df_negativos.iterrows():
        categoria = row['category']
        filename = row['filename']
        
        # Crear subdirectorio por categoría
        cat_dir = ESC50_DIR / categoria
        cat_dir.mkdir(parents=True, exist_ok=True)
        
        # Archivo fuente (ESC-50 usa .wav en la carpeta audio/)
        # Nota: el metadata dice .ogg pero los archivos descargados son .wav
        src_file = audio_src_dir / filename
        
        # Si el archivo no existe con la extensión del metadata, buscar con .wav
        if not src_file.exists():
            # Intentar con .wav en lugar de .ogg
            wav_name = filename.replace('.ogg', '.wav')
            src_file = audio_src_dir / wav_name
        
        if not src_file.exists():
            errores += 1
            rutas_locales.append(None)
            continue
        
        # Copiar al directorio del proyecto
        dst_file = cat_dir / src_file.name
        if not dst_file.exists():
            shutil.copy2(str(src_file), str(dst_file))
        
        rutas_locales.append(str(dst_file.relative_to(PROJECT_ROOT)))
        copiados += 1
    
    # Agregar columna con ruta local al DataFrame
    df_negativos = df_negativos.copy()
    df_negativos['local_path'] = rutas_locales
    
    print(f"   ✅ Archivos copiados: {copiados}")
    if errores > 0:
        print(f"   ⚠️  Archivos no encontrados: {errores}")
    print(f"   📂 Directorio destino: {ESC50_DIR}")
    
    # Mostrar estructura de carpetas
    print(f"\n   Estructura creada:")
    for cat_dir in sorted(ESC50_DIR.iterdir()):
        if cat_dir.is_dir():
            n_files = len(list(cat_dir.glob("*")))
            print(f"      {cat_dir.name}/  ({n_files} archivos)")
    
    print()
    return df_negativos


def generar_metadata_negativos(df_negativos: pd.DataFrame) -> None:
    """
    Genera el archivo metadata_negativos.csv compatible con el formato
    del proyecto SelvaSonic.
    
    Columnas del CSV:
    - filename: nombre del archivo de audio
    - category: categoría ESC-50 (rain, siren, etc.)
    - esc50_target: ID numérico original en ESC-50
    - selvasonic_label: etiqueta en SelvaSonic (siempre "no_ave")
    - selvasonic_class_id: ID de clase en SelvaSonic (siempre 0)
    - local_path: ruta relativa al archivo de audio
    - fold: fold de cross-validation original de ESC-50
    - source: siempre "ESC-50"
    """
    print("=" * 65)
    print("📝 GENERANDO METADATA DE NEGATIVOS")
    print("=" * 65)
    
    # Crear DataFrame limpio para el proyecto
    metadata = pd.DataFrame({
        'filename': df_negativos['filename'].values,
        'category': df_negativos['category'].values,
        'esc50_target': df_negativos['target'].values,
        'selvasonic_label': 'no_ave',
        'selvasonic_class_id': 0,
        'local_path': df_negativos['local_path'].values,
        'fold': df_negativos['fold'].values,
        'source': 'ESC-50'
    })
    
    # Eliminar filas sin ruta local (archivos no encontrados)
    n_antes = len(metadata)
    metadata = metadata.dropna(subset=['local_path'])
    n_despues = len(metadata)
    
    if n_antes != n_despues:
        print(f"   ⚠️  Se eliminaron {n_antes - n_despues} filas sin archivo de audio")
    
    # Guardar CSV
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(METADATA_NEGATIVOS, index=False, encoding='utf-8')
    
    print(f"   ✅ Metadata guardada en: {METADATA_NEGATIVOS}")
    print(f"   Total de clips negativos: {len(metadata)}")
    print(f"\n   Primeras 5 filas:")
    print(metadata.head().to_string(index=False))
    
    print()
    return metadata


def resumen_final(metadata_neg: pd.DataFrame) -> None:
    """
    Muestra un resumen comparativo entre los datos positivos (aves) y
    los negativos (ESC-50) para tener una vista completa del dataset.
    """
    print("=" * 65)
    print("📊 RESUMEN FINAL — DATASET SELVASONIC")
    print("=" * 65)
    
    # Intentar cargar metadata de positivos si existe
    metadata_positivos_path = DATA_DIR / "metadata.csv"
    
    print("\n   CLASE NEGATIVA (ESC-50):")
    print(f"   ├── Clips totales:      {len(metadata_neg)}")
    print(f"   ├── Categorías:         {metadata_neg['category'].nunique()}")
    print(f"   ├── Formato:            WAV 44.1 kHz, mono, 5 segundos")
    print(f"   ├── Label SelvaSonic:   'no_ave' (class_id = 0)")
    print(f"   └── Fuente:             ESC-50 (Piczak, 2015)")
    
    if metadata_positivos_path.exists():
        df_pos = pd.read_csv(metadata_positivos_path)
        n_especies = df_pos['species'].nunique() if 'species' in df_pos.columns else '?'
        print(f"\n   CLASES POSITIVAS (Xeno-canto):")
        print(f"   ├── Clips totales:      {len(df_pos)}")
        print(f"   ├── Especies:           {n_especies}")
        print(f"   └── Labels SelvaSonic:  class_id 1..{n_especies}")
        
        ratio = len(metadata_neg) / len(df_pos)
        print(f"\n   BALANCE:")
        print(f"   ├── Ratio negativos/positivos: {ratio:.2f}x")
        if ratio > 3:
            print(f"   └── ⚠️  Desbalance significativo → usar class_weights en la loss")
        else:
            print(f"   └── ✅ Balance razonable")
    else:
        print(f"\n   ℹ️  No se encontró metadata de positivos en {metadata_positivos_path}")
        print(f"       Ejecuta primero 02_descarga_audios.py para generar los positivos.")
    
    print(f"\n   PRÓXIMOS PASOS:")
    print(f"   1. Unificar metadata_negativos.csv con metadata.csv (positivos)")
    print(f"   2. Convertir todos los audios a WAV 22050 Hz (si no lo están)")
    print(f"   3. Segmentar clips largos en ventanas de 5 segundos")
    print(f"   4. Extraer espectrogramas Mel → tensores PyTorch")
    print(f"   5. Crear DataLoaders con split estratificado (train/val/test)")
    print()
    print("🦜 SelvaSonic — Escuchando la Amazonía con Machine Learning")
    print("=" * 65)


def limpiar_temporales() -> None:
    """Elimina el directorio temporal de descarga para ahorrar espacio."""
    if ESC50_DOWNLOAD_DIR.exists():
        print("🗑️  Limpiando archivos temporales de descarga...")
        shutil.rmtree(str(ESC50_DOWNLOAD_DIR))
        print("   ✅ Temporales eliminados.\n")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print()
    print("🦜" + "=" * 63)
    print("   SELVASONIC — Descarga de Negativos (ESC-50)")
    print("   Actividad Semana 1: Dataset de clase negativa")
    print("=" * 65)
    print()
    
    # Paso 1: Descargar ESC-50
    esc50_dir = descargar_esc50()
    
    # Paso 2: Cargar y explorar metadata
    df = cargar_metadata_esc50(esc50_dir)
    
    # Paso 3: Filtrar solo los negativos (excluir animales)
    df_negativos = filtrar_negativos(df)
    
    # Paso 4: Copiar audios al directorio del proyecto
    df_negativos = copiar_audios_negativos(df_negativos, esc50_dir)
    
    # Paso 5: Generar archivo de metadata para el proyecto
    metadata_neg = generar_metadata_negativos(df_negativos)
    
    # Paso 6: Limpiar temporales (opcional — descomentar si quieres)
    # limpiar_temporales()
    
    # Paso 7: Resumen final
    resumen_final(metadata_neg)