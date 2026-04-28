"""
src/config.py
=============

Configuración centralizada de SelvaSonic.

Este módulo es la **fuente única de verdad** para todos los hiperparámetros
del pipeline de datos del proyecto. Cualquier módulo (audio_io, transforms,
dataset, etc.) que necesite conocer un parámetro debe importarlo desde aquí,
NUNCA hardcodearlo localmente.

Ventajas de centralizar:
- Un solo lugar para revisar/modificar configuración del proyecto.
- Reproducibilidad: el reporte final puede simplemente referenciar este archivo.
- Coherencia: imposible que dos módulos usen valores distintos de SAMPLE_RATE.
- Experimentación rápida: cambiar un hiperparámetro = cambiar un solo número.

Convención: TODAS las constantes en MAYÚSCULAS_CON_GUIÓN_BAJO (PEP 8).

Autoras(es)
-----------
Laura Ruiz Arango & Jose Aldair Molina Méndez
Universidad Nacional de Colombia - Sede Medellín
Aprendizaje Automático (Prof. Alcides Montoya) - 2026
"""

from __future__ import annotations


# =============================================================================
# 1. AUDIO I/O
# =============================================================================
# Constantes que controlan la carga de audio desde disco.
# Usadas por: src/audio_io.py

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

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".mp3", ".wav", ".flac", ".ogg")
"""Formatos de audio que el pipeline puede leer.

Xeno-canto entrega MP3, ESC-50 entrega WAV. Los otros (FLAC, OGG)
se incluyen por si en el futuro se añaden datasets adicionales.
"""


# =============================================================================
# 2. EXTRACCIÓN DE MEL-SPECTROGRAMAS
# =============================================================================
# Constantes que controlan el cálculo del Mel-spectrograma a partir
# del waveform. Usadas por: src/transforms.py

N_FFT: int = 2048
"""Tamaño de la ventana FFT (número de muestras por frame).

Justificación:
    - Resolución frecuencial: Δf = SAMPLE_RATE / N_FFT = 22050/2048 ≈ 10.8 Hz/bin
    - Suficiente para distinguir tonos de aves (cuyas variaciones son >> 10 Hz)
    - Estándar de facto en bioacústica con sr=22050.

Trade-off (principio de incertidumbre tiempo-frecuencia):
    - N_FFT más grande → mejor resolución frecuencial, peor temporal
    - N_FFT más pequeño → mejor resolución temporal, peor frecuencial
    - 2048 es el equilibrio establecido en el campo.
"""

HOP_LENGTH: int = 512
"""Paso (en muestras) entre ventanas FFT consecutivas.

Justificación:
    - Resolución temporal: Δt = HOP_LENGTH / SAMPLE_RATE ≈ 23 ms por frame
    - Captura trinos rápidos de aves (sílabas de ~50-100 ms tienen 2-4 frames)
    - Ratio N_FFT/HOP_LENGTH = 4 → overlap del 75% entre ventanas (estándar).
"""

N_MELS: int = 128
"""Número de bandas en el banco de filtros Mel.

Justificación:
    - 128 es el estándar moderno en clasificación de audio
      (YAMNet, AST, PANNs, BirdNET).
    - Da suficiente detalle para que el CNN aprenda patrones espectrales
      sin ser excesivo (más bandas = más cómputo, ganancia marginal).
    - Coincide con la arquitectura declarada en el README de SelvaSonic.
"""

FMIN: float = 50.0
"""Frecuencia mínima del banco de filtros Mel, en Hz.

Justificación:
    - Por debajo de 50 Hz hay principalmente ruido (cable, viento, motores).
    - Las aves no emiten contenido relevante en frecuencias tan bajas.
    - Filtrar este rango reduce ruido sin perder información de canto.
"""

FMAX: float = 11_025.0
"""Frecuencia máxima del banco de filtros Mel, en Hz.

Justificación:
    - Es el límite de Nyquist: SAMPLE_RATE / 2 = 11,025 Hz.
    - Es físicamente imposible representar frecuencias mayores con
      el sample rate del proyecto.
"""


# =============================================================================
# 3. SEGMENTACIÓN DE CLIPS
# =============================================================================
# Constantes que controlan el corte de audios largos en clips uniformes.
# Usadas por: src/segmentation.py, src/dataset.py

CLIP_DURATION_S: float = 5.0
"""Duración estándar de cada clip de entrenamiento, en segundos.

Justificación:
    - 5 segundos es el estándar en bioacústica computacional
      (BirdCLEF, BirdNET, AST-bird).
    - Suficiente para capturar al menos un canto completo de la mayoría
      de aves amazónicas.
    - Coincide con la duración nativa de ESC-50 (también 5s) → no requiere
      padding cuando se entrena con clases negativas.
"""

CLIP_NUM_SAMPLES: int = int(CLIP_DURATION_S * SAMPLE_RATE)
"""Número de muestras de un clip estándar (constante derivada).

Calculado automáticamente: CLIP_DURATION_S * SAMPLE_RATE = 5.0 * 22050 = 110250.
Usado para padding/truncate de clips que no midan exactamente 5s.
NO modificar manualmente — depende de CLIP_DURATION_S y SAMPLE_RATE.
"""

CLIP_OVERLAP_RATIO: float = 0.5
"""Solapamiento entre clips consecutivos, fracción en [0.0, 1.0).

Valores típicos:
    - 0.0 → clips adyacentes (sin solape)
    - 0.5 → 50% de solape (cada clip avanza 2.5s) — RECOMENDADO
    - 0.75 → 75% de solape (más datos, redundancia alta)

Justificación:
    Con solape 50%, un audio de 30s genera 11 clips en lugar de 6.
    Es data augmentation honesto: ventanas distintas del mismo evento
    acústico ayudan al modelo a aprender invarianza temporal.

IMPORTANTE: Solo aplicar solape en TRAIN, nunca en VAL/TEST.
    Si dos clips de validación se solapan al 50%, comparten el 50% de
    sus muestras → métricas infladas (data leakage entre clips).
"""

SHORT_AUDIO_STRATEGY: str = "wrap"
"""Estrategia para audios MÁS CORTOS que CLIP_DURATION_S.

Opciones:
    "wrap"  → repetición circular (loop) — preserva propiedades estadísticas
    "zero"  → padding con ceros — introduce silencio artificial (no recomendado)
    "drop"  → descartar el audio — pérdida de datos

Justificación de "wrap" como default:
    El padding con ceros mete silencio que la CNN puede aprender como
    "feature de borde" (sesgo). El padding circular conserva la firma
    espectral del canto, manteniendo el espectrograma coherente.
"""

LEFTOVER_STRATEGY: str = "wrap"
"""Estrategia para el SOBRANTE al final de un audio largo.

Ejemplo: audio de 12s con clips de 5s y solape 0 → 2 clips + 2s sobrantes.
¿Qué hacer con esos 2s?

Opciones:
    "wrap"  → completar con repetición desde el inicio del sobrante
    "drop"  → descartar el sobrante (más limpio, pierde poco)
    "zero"  → padding con ceros (no recomendado)

Solo se procesa el sobrante si dura ≥ MIN_LEFTOVER_SEC (ver abajo).
"""

MIN_LEFTOVER_SEC: float = 2.0
"""Mínima duración (en segundos) del sobrante para procesarlo.

Justificación:
    Un sobrante de 0.3s rellenado con repetición produciría un clip
    casi totalmente sintético — ruido para el modelo. A partir de 2s
    de audio "real" hay suficiente firma espectral para que el clip
    aporte información, incluso después del padding.
"""


# =============================================================================
# 4. RESERVADO PARA FUTURO (Semanas 3-6)
# =============================================================================
# Estos hiperparámetros se irán llenando conforme avance el proyecto.
# Se dejan documentados aquí para tener un "mapa" de qué falta configurar.

# --- Modelo (Semana 3) ---
# CNN_CHANNELS: list[int] = [32, 64, 128, 256]
# ATTENTION_HEADS: int = 4
# DROPOUT: float = 0.3
# NUM_CLASSES: int = 11  # 10 especies + clase "no_ave"

# --- Entrenamiento (Semana 4) ---
# BATCH_SIZE: int = 32
# LEARNING_RATE: float = 1e-3
# NUM_EPOCHS: int = 50
# WEIGHT_DECAY: float = 1e-4

# --- Inferencia (Semana 5-6) ---
# CONFIDENCE_THRESHOLD: float = 0.6  # Por debajo → "No identificado"
