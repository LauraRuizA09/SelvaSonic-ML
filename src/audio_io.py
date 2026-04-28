"""
src/audio_io.py
===============

Módulo de entrada/salida de audio para SelvaSonic.

Este módulo centraliza la carga de archivos de audio desde disco hacia
arrays de NumPy, unificando formatos heterogéneos (MP3 de Xeno-canto,
WAV de ESC-50) en una representación homogénea para el resto del pipeline.

Responsabilidades
-----------------
- Leer audio de cualquier formato soportado por librosa (MP3, WAV, FLAC, OGG).
- Resamplear a la frecuencia de muestreo unificada del proyecto (22,050 Hz).
- Convertir a mono (promediando canales si el audio viniera estéreo).
- Retornar metadatos útiles: sample rate, duración, forma del array.
- Manejar errores de I/O de forma explícita (archivos corruptos, rutas
  inválidas, audios vacíos).

Este módulo NO hace:
- Segmentación en clips (eso va en transforms.py).
- Extracción de espectrogramas (eso va en transforms.py).
- Data augmentation (eso va en transforms.py).

Autoras(es)
-----------
Laura Ruiz Arango & Jose Aldair Molina Méndez
Universidad Nacional de Colombia - Sede Medellín
Aprendizaje Automático (Prof. Alcides Montoya) - 2026
"""

from __future__ import annotations

# --- Librerías estándar ---
from pathlib import Path
from typing import NamedTuple

# --- Librerías de terceros ---
import numpy as np
import librosa

# --- Configuración del proyecto ---
# Importamos las constantes desde config.py (fuente única de verdad).
# NO duplicamos definiciones aquí — si necesitas cambiar SAMPLE_RATE,
# modifica src/config.py y todo el proyecto se actualiza automáticamente.
from src.config import SAMPLE_RATE, MONO, SUPPORTED_EXTENSIONS


# =============================================================================
# TIPOS DE RETORNO
# =============================================================================

class AudioData(NamedTuple):
    """Contenedor de un audio cargado y sus metadatos.

    Attributes
    ----------
    waveform : np.ndarray
        Array 1D de float32 con la forma de onda en mono, normalizada
        en el rango [-1.0, 1.0]. Shape: (n_samples,).
    sample_rate : int
        Frecuencia de muestreo del audio cargado, en Hz.
        Siempre será igual a SAMPLE_RATE tras el resampling.
    duration_s : float
        Duración del audio en segundos (= len(waveform) / sample_rate).
    source_path : str
        Ruta original del archivo de audio (útil para logging y debugging).
    """
    waveform: np.ndarray
    sample_rate: int
    duration_s: float
    source_path: str


# =============================================================================
# EXCEPCIONES PERSONALIZADAS
# =============================================================================

class AudioLoadError(Exception):
    """Error al cargar o procesar un archivo de audio."""
    pass


# =============================================================================
# FUNCIONES PÚBLICAS
# =============================================================================

def load_audio(
    path: str | Path,
    *,
    target_sr: int = SAMPLE_RATE,
    mono: bool = MONO,
    offset: float = 0.0,
    duration: float | None = None,
) -> AudioData:
    """Carga un archivo de audio desde disco y devuelve un array de NumPy.

    Esta es la función central de I/O del proyecto. Todo módulo que
    necesite leer un audio debe llamar a esta función, nunca a
    librosa.load() directamente, para garantizar consistencia en el
    sample rate, número de canales y manejo de errores.

    Parameters
    ----------
    path : str | Path
        Ruta al archivo de audio. Formatos soportados: MP3, WAV, FLAC, OGG.
    target_sr : int, optional
        Frecuencia de muestreo destino en Hz. Por defecto, SAMPLE_RATE
        (definido en config.py = 22,050 Hz).
    mono : bool, optional
        Si es True, convierte estéreo a mono promediando los canales.
        Por defecto, MONO (definido en config.py = True).
    offset : float, optional
        Segundo desde el cual empezar a leer. Útil para segmentación
        eficiente sin cargar todo el audio a memoria. Por defecto, 0.0.
    duration : float | None, optional
        Duración en segundos a leer. Si es None, lee hasta el final.
        Por defecto, None.

    Returns
    -------
    AudioData
        NamedTuple con (waveform, sample_rate, duration_s, source_path).

    Raises
    ------
    AudioLoadError
        Si el archivo no existe, tiene una extensión no soportada, está
        corrupto, o resulta en un audio vacío.

    Examples
    --------
    >>> audio = load_audio("data/raw/Ramphastos_tucanus/XC694038.mp3")
    >>> audio.waveform.shape
    (551250,)  # ~25 segundos a 22,050 Hz

    >>> # Cargar solo los primeros 5 segundos (eficiente en disco)
    >>> clip = load_audio("audio.mp3", duration=5.0)
    >>> clip.duration_s
    5.0
    """
    # 1. Normalizar el path
    path = Path(path)

    # 2. Validar que el archivo existe
    if not path.exists():
        raise AudioLoadError(f"El archivo no existe: {path}")
    if not path.is_file():
        raise AudioLoadError(f"La ruta no apunta a un archivo: {path}")

    # 3. Validar extensión soportada
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise AudioLoadError(
            f"Extensión '{extension}' no soportada. "
            f"Formatos válidos: {SUPPORTED_EXTENSIONS}"
        )

    # 4. Cargar el audio con librosa
    # librosa.load() hace internamente:
    #   a) Decodifica el archivo (soundfile para WAV/FLAC, audioread para MP3).
    #   b) Convierte a float32 en el rango [-1.0, 1.0].
    #   c) Si sr != sr_original, aplica filtro anti-aliasing y resamplea.
    #   d) Si mono=True y el audio es estéreo, promedia canales.
    try:
        waveform, sr = librosa.load(
            str(path),
            sr=target_sr,
            mono=mono,
            offset=offset,
            duration=duration,
        )
    except Exception as e:
        raise AudioLoadError(
            f"Error al decodificar el audio '{path}': {type(e).__name__}: {e}"
        ) from e

    # 5. Validar el resultado
    if waveform.size == 0:
        raise AudioLoadError(
            f"El audio cargado está vacío (0 samples): {path}. "
            f"Posibles causas: archivo corrupto, offset > duración total, "
            f"o archivo de 0 bytes."
        )

    # Sanity check de dtype (librosa siempre retorna float32, pero validamos)
    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32)

    # 6. Construir y retornar el AudioData
    duration_s = len(waveform) / sr

    return AudioData(
        waveform=waveform,
        sample_rate=sr,
        duration_s=duration_s,
        source_path=str(path),
    )


# =============================================================================
# UTILIDADES
# =============================================================================

def summarize(audio: AudioData) -> str:
    """Genera un resumen legible de un AudioData, útil para logging."""
    filename = Path(audio.source_path).name
    return (
        f"AudioData(file='{filename}', "
        f"sr={audio.sample_rate} Hz, "
        f"duration={audio.duration_s:.2f} s, "
        f"samples={len(audio.waveform)})"
    )


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    import sys

    data_dir = Path(__file__).parent.parent / "data" / "raw"

    if not data_dir.exists():
        print(f"⚠️  No existe {data_dir}. Smoke test omitido.")
        sys.exit(0)

    test_files = [
        f for ext in SUPPORTED_EXTENSIONS
        for f in data_dir.rglob(f"*{ext}")
    ]

    if not test_files:
        print(f"⚠️  No hay audios en {data_dir}. Smoke test omitido.")
        sys.exit(0)

    test_path = test_files[0]
    print(f"🔍 Probando carga de: {test_path}")

    try:
        audio = load_audio(test_path)
        print(f"✅ {summarize(audio)}")
        print(f"   - dtype: {audio.waveform.dtype}")
        print(f"   - min/max: [{audio.waveform.min():.4f}, {audio.waveform.max():.4f}]")
        print(f"   - mean/std: {audio.waveform.mean():.4f} / {audio.waveform.std():.4f}")
    except AudioLoadError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
