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
- Segmentación en clips (eso va en transforms.py, Semana 2).
- Extracción de espectrogramas (eso va en transforms.py, Semana 2).
- Data augmentation (eso va en transforms.py, Semana 2).

"""

from __future__ import annotations  # permite type hints modernos en Python 3.10

# --- Librerías estándar ---
from pathlib import Path
from typing import NamedTuple

# --- Librerías de terceros ---
import numpy as np
import librosa


# =============================================================================
# CONSTANTES DEL PROYECTO
# =============================================================================
# Estas constantes son la "fuente única de verdad" para todo SelvaSonic.
# Cualquier módulo que necesite conocer el sample rate estándar debe
# importarlo desde aquí, NO hardcodearlo.

SAMPLE_RATE: int = 22_050

"""Frecuencia de muestreo unificada del proyecto, en Hz.

Justificación (teorema de Nyquist):
    Las vocalizaciones de aves amazónicas se concentran en el rango
    500 Hz - 10 kHz. Con sr = 22,050 Hz, la frecuencia máxima
    representable es sr/2 = 11,025 Hz, suficiente para capturar
    toda la información biológicamente relevante.

Trade-off:
    Usar 22,050 Hz en lugar de 44,100 Hz (calidad CD) reduce a la mitad
    el tamaño de los arrays y acelera el entrenamiento, sin perder
    información útil para clasificación de aves.
"""

MONO: bool = True

"""Todos los audios se convierten a mono (un solo canal).

Justificación:
    La identificación de especies por vocalización no depende de
    información espacial (estéreo). Trabajar en mono reduce el tamaño
    del tensor y simplifica el pipeline.
"""

# Formatos de audio soportados por librosa en este proyecto.
# librosa soporta más, pero estos son los únicos que aparecen en nuestras
# fuentes de datos (Xeno-canto = MP3, ESC-50 = WAV).

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".mp3", ".wav", ".flac", ".ogg")


# =============================================================================
# TIPOS DE RETORNO
# =============================================================================
# Usamos NamedTuple en lugar de retornar una tupla "desnuda" (waveform, sr).
# Ventajas:
#   - Auto-documentado: load_audio(...).sample_rate es más claro que [1].
#   - Inmutable: nadie puede modificarlo accidentalmente.
#   - Se puede ampliar en el futuro sin romper código existente.

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
# FUNCIONES PÚBLICAS
# =============================================================================
# (Se implementarán en los siguientes commits)

# TODO [Commit 2]: implementar load_audio(path) -> AudioData
# TODO [Commit 3]: añadir validación y manejo de errores
# TODO [Commit 4]: implementar load_dataset(metadata_csv) -> Iterator[AudioData]