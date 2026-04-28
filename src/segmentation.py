"""
src/segmentation.py
====================

Segmentación de audios largos en clips de duración fija para alimentar
la CNN del clasificador SelvaSonic.

¿Por qué este módulo existe?
----------------------------
Los audios de Xeno-canto tienen duraciones muy variables (desde 3s hasta
varios minutos). Una CNN necesita inputs de tamaño fijo. La solución es
trocear cada audio en clips de duración constante (CLIP_DURATION_S).

Decisiones de diseño (documentadas en config.py):
- Duración fija: 5 segundos (~ una vocalización completa de ave)
- Solapamiento configurable: 50% por defecto (más datos sin grabar más)
- Audios cortos: padding circular (loop) para preservar estadísticas
- Sobrantes ≥ 2s: padding por repetición; < 2s: descartados

Estructura del módulo:
- AudioClip: NamedTuple con metadata completa de cada clip
- SegmentationError: excepción específica del módulo
- segment_audio(): función pública principal
- Funciones privadas (_pad_*, _compute_*): helpers reutilizables
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

# Importamos de config.py — fuente única de verdad para constantes
from src.config import (
    CLIP_DURATION_S,
    CLIP_NUM_SAMPLES,
    CLIP_OVERLAP_RATIO,
    LEFTOVER_STRATEGY,
    MIN_LEFTOVER_SEC,
    SHORT_AUDIO_STRATEGY,
)


# ============================================================================
# Estructuras de datos
# ============================================================================

class AudioClip(NamedTuple):
    """
    Representa un clip individual de audio segmentado.

    Atributos
    ---------
    samples : np.ndarray
        Array 1D de muestras de audio, con longitud EXACTA de
        CLIP_NUM_SAMPLES (110250 muestras para 5s a 22050 Hz).
    sample_rate : int
        Frecuencia de muestreo en Hz.
    source_file : str
        Path o nombre del archivo de audio original (trazabilidad).
    clip_index : int
        Índice del clip dentro del audio original (0-based).
        Útil para reconstruir el audio o reportar qué segmento confundió
        al modelo.
    start_time : float
        Segundo de inicio del clip dentro del audio original.
    end_time : float
        Segundo final del clip dentro del audio original. Si el clip
        tuvo padding, end_time refleja el tiempo "real" antes del padding.
    is_padded : bool
        True si el clip tuvo que rellenarse (audio corto o sobrante).
        Útil para análisis: ¿el modelo se equivoca más con clips padded?

    ¿Por qué NamedTuple y no @dataclass?
    ------------------------------------
    - Inmutable (no se puede modificar accidentalmente)
    - Más liviano en memoria (importante con miles de clips)
    - Compatible con tuple unpacking: samples, sr, *_ = clip
    - Coherente con AudioData de audio_io.py (mismo estilo)
    """
    samples: np.ndarray
    sample_rate: int
    source_file: str
    clip_index: int
    start_time: float
    end_time: float
    is_padded: bool


class SegmentationError(Exception):
    """
    Error específico de segmentación.

    Se lanza cuando:
    - El audio de entrada está vacío o tiene shape inválido.
    - Los parámetros de segmentación son inconsistentes.
    - La estrategia configurada no es reconocida.
    """


# ============================================================================
# Funciones privadas (helpers)
# ============================================================================

def _pad_circular(samples: np.ndarray, target_length: int) -> np.ndarray:
    """
    Padding circular: repite el audio desde el inicio hasta llegar a
    target_length muestras.

    Ejemplo: audio de 3s, target 5s → 3s + 2s_iniciales = 5s

    ¿Por qué circular y no con ceros?
    ---------------------------------
    El padding con ceros introduce silencio artificial. La CNN podría
    aprender que "audios cortos = silencio en el final" como un sesgo.
    El padding circular preserva las propiedades espectrales y temporales
    de la señal original — el espectrograma sigue siendo coherente.

    Parámetros
    ----------
    samples : np.ndarray
        Array 1D más corto que target_length.
    target_length : int
        Longitud deseada en muestras.

    Retorna
    -------
    np.ndarray
        Array 1D con longitud exactamente target_length.
    """
    if samples.shape[0] >= target_length:
        return samples[:target_length]

    # np.tile repite el array N veces. Calculamos cuántas repeticiones
    # mínimas necesitamos para superar target_length, luego truncamos.
    repetitions_needed = int(np.ceil(target_length / samples.shape[0]))
    tiled = np.tile(samples, repetitions_needed)
    return tiled[:target_length]


def _pad_zero(samples: np.ndarray, target_length: int) -> np.ndarray:
    """
    Padding con ceros al final. Implementado por completitud, NO recomendado.
    """
    if samples.shape[0] >= target_length:
        return samples[:target_length]

    pad_amount = target_length - samples.shape[0]
    return np.pad(samples, (0, pad_amount), mode="constant", constant_values=0.0)


def _apply_strategy(
    samples: np.ndarray,
    target_length: int,
    strategy: str,
) -> np.ndarray:
    """
    Aplica la estrategia de padding configurada.

    Parámetros
    ----------
    samples : np.ndarray
        Audio original más corto que target_length.
    target_length : int
        Longitud deseada en muestras.
    strategy : str
        Una de: "wrap", "zero". "drop" se maneja antes (devuelve None
        en el caller).

    Retorna
    -------
    np.ndarray
        Audio con longitud exactamente target_length.

    Lanza
    -----
    SegmentationError
        Si strategy no es reconocida.
    """
    if strategy == "wrap":
        return _pad_circular(samples, target_length)
    if strategy == "zero":
        return _pad_zero(samples, target_length)
    raise SegmentationError(
        f"Estrategia de padding no reconocida: '{strategy}'. "
        f"Opciones válidas: 'wrap', 'zero'."
    )


def _compute_n_clips(
    audio_length: int,
    clip_length: int,
    hop_length: int,
) -> int:
    """
    Calcula cuántos clips COMPLETOS caben en el audio.

    Aplica la fórmula:  N = floor((L - T_c) / h) + 1

    Donde:
    - L = audio_length (muestras)
    - T_c = clip_length (muestras por clip)
    - h = hop_length (avance entre clips)

    Si audio_length < clip_length, retorna 0 (el audio se manejará como
    "audio corto" con SHORT_AUDIO_STRATEGY).

    Ejemplo: L=15s, T_c=5s, h=2.5s (50% solape)
        N = floor((15-5)/2.5) + 1 = floor(4) + 1 = 5 clips
    """
    if audio_length < clip_length:
        return 0
    return (audio_length - clip_length) // hop_length + 1


# ============================================================================
# Función pública principal
# ============================================================================

def segment_audio(
    samples: np.ndarray,
    *,
    sample_rate: int,
    source_file: str = "<unknown>",
    clip_duration_sec: float = CLIP_DURATION_S,
    overlap_ratio: float = CLIP_OVERLAP_RATIO,
    short_audio_strategy: str = SHORT_AUDIO_STRATEGY,
    leftover_strategy: str = LEFTOVER_STRATEGY,
    min_leftover_sec: float = MIN_LEFTOVER_SEC,
) -> list[AudioClip]:
    """
    Segmenta un audio en clips de duración fija.

    Parámetros (keyword-only excepto samples)
    -----------------------------------------
    samples : np.ndarray
        Array 1D del audio cargado (output de load_audio()).
    sample_rate : int
        Frecuencia de muestreo en Hz.
    source_file : str, opcional
        Ruta o nombre del archivo original (para trazabilidad).
    clip_duration_sec : float
        Duración de cada clip en segundos. Por defecto, CLIP_DURATION_S.
    overlap_ratio : float
        Solapamiento entre clips consecutivos en [0.0, 1.0).
    short_audio_strategy : str
        Cómo manejar audios más cortos que clip_duration_sec.
        Opciones: "wrap", "zero", "drop".
    leftover_strategy : str
        Cómo manejar el sobrante final.
        Opciones: "wrap", "zero", "drop".
    min_leftover_sec : float
        Mínima duración del sobrante para procesarlo (segundos).

    Retorna
    -------
    list[AudioClip]
        Lista de clips segmentados. Puede estar vacía si el audio era
        corto y short_audio_strategy="drop".

    Lanza
    -----
    SegmentationError
        Si los parámetros son inválidos o el audio está vacío.

    Ejemplos
    --------
    >>> import numpy as np
    >>> sr = 22050
    >>> audio = np.random.randn(15 * sr).astype(np.float32)  # 15 segundos
    >>> clips = segment_audio(audio, sample_rate=sr)
    >>> len(clips)  # Con 50% solape: 5 clips
    5
    >>> clips[0].samples.shape
    (110250,)
    """
    # ------------------------------------------------------------------
    # Validaciones de entrada
    # ------------------------------------------------------------------
    if samples.ndim != 1:
        raise SegmentationError(
            f"samples debe ser 1D, recibido shape {samples.shape}"
        )
    if samples.size == 0:
        raise SegmentationError("samples está vacío.")
    if not (0.0 <= overlap_ratio < 1.0):
        raise SegmentationError(
            f"overlap_ratio debe estar en [0.0, 1.0), recibido {overlap_ratio}"
        )
    if clip_duration_sec <= 0:
        raise SegmentationError(
            f"clip_duration_sec debe ser > 0, recibido {clip_duration_sec}"
        )

    # ------------------------------------------------------------------
    # Conversión a muestras
    # ------------------------------------------------------------------
    # Trabajamos en SAMPLES, no en segundos, para evitar errores de
    # redondeo. La librería estándar de DSP siempre opera sobre muestras.
    clip_length = int(clip_duration_sec * sample_rate)
    hop_length = int(clip_length * (1.0 - overlap_ratio))
    min_leftover_samples = int(min_leftover_sec * sample_rate)
    audio_length = samples.shape[0]

    # Edge case: hop_length=0 si overlap_ratio≈1.0 (loop infinito)
    # Ya validamos overlap_ratio < 1.0, pero por robustez:
    if hop_length <= 0:
        raise SegmentationError(
            f"hop_length calculado es {hop_length}. "
            f"Revisa overlap_ratio={overlap_ratio}."
        )

    clips: list[AudioClip] = []

    # ------------------------------------------------------------------
    # Caso 1: Audio MÁS CORTO que clip_duration → estrategia especial
    # ------------------------------------------------------------------
    if audio_length < clip_length:
        if short_audio_strategy == "drop":
            return []  # No generar ningún clip
        padded = _apply_strategy(samples, clip_length, short_audio_strategy)
        clips.append(AudioClip(
            samples=padded,
            sample_rate=sample_rate,
            source_file=source_file,
            clip_index=0,
            start_time=0.0,
            end_time=audio_length / sample_rate,  # tiempo "real" sin padding
            is_padded=True,
        ))
        return clips

    # ------------------------------------------------------------------
    # Caso 2: Audio LARGO → ventanas deslizantes
    # ------------------------------------------------------------------
    n_clips = _compute_n_clips(audio_length, clip_length, hop_length)

    for i in range(n_clips):
        start_sample = i * hop_length
        end_sample = start_sample + clip_length
        clip_samples = samples[start_sample:end_sample]

        clips.append(AudioClip(
            samples=clip_samples,
            sample_rate=sample_rate,
            source_file=source_file,
            clip_index=i,
            start_time=start_sample / sample_rate,
            end_time=end_sample / sample_rate,
            is_padded=False,
        ))

    # ------------------------------------------------------------------
    # Caso 3: Sobrante al final
    # ------------------------------------------------------------------
    # Después del último clip, ¿queda audio sin procesar?
    last_processed_sample = (n_clips - 1) * hop_length + clip_length
    leftover_samples = audio_length - last_processed_sample

    if leftover_samples >= min_leftover_samples:
        if leftover_strategy == "drop":
            pass  # No agregar el sobrante
        else:
            leftover_audio = samples[last_processed_sample:]
            padded_leftover = _apply_strategy(
                leftover_audio, clip_length, leftover_strategy
            )
            clips.append(AudioClip(
                samples=padded_leftover,
                sample_rate=sample_rate,
                source_file=source_file,
                clip_index=n_clips,  # índice siguiente al último
                start_time=last_processed_sample / sample_rate,
                end_time=audio_length / sample_rate,  # tiempo real
                is_padded=True,
            ))

    return clips


# ============================================================================
# Smoke test — verifica que el módulo funciona end-to-end
# ============================================================================
# Se ejecuta SOLO si corres `python -m src.segmentation` directamente.
# Importar el módulo (en otros archivos) no dispara este código.

if __name__ == "__main__":
    print("=" * 70)
    print("SMOKE TEST: src/segmentation.py")
    print("=" * 70)
    print(f"\nUsando constantes de config.py:")
    print(f"  CLIP_DURATION_S    = {CLIP_DURATION_S}")
    print(f"  CLIP_NUM_SAMPLES   = {CLIP_NUM_SAMPLES}")
    print(f"  CLIP_OVERLAP_RATIO = {CLIP_OVERLAP_RATIO}")
    print(f"  SHORT_AUDIO_STRATEGY = '{SHORT_AUDIO_STRATEGY}'")
    print(f"  LEFTOVER_STRATEGY    = '{LEFTOVER_STRATEGY}'")
    print(f"  MIN_LEFTOVER_SEC     = {MIN_LEFTOVER_SEC}")

    SR = 22050

    # ---- Test 1: Audio largo (15 segundos) sin solape ----
    print("\n[Test 1] Audio largo de 15s, sin solape:")
    audio_long = np.random.randn(15 * SR).astype(np.float32)
    clips = segment_audio(
        audio_long,
        sample_rate=SR,
        source_file="test_largo.wav",
        overlap_ratio=0.0,
    )
    print(f"  Audio entrada: {audio_long.shape[0]} muestras "
          f"({audio_long.shape[0] / SR:.1f}s)")
    print(f"  Clips generados: {len(clips)}")
    for c in clips:
        print(f"    clip {c.clip_index}: "
              f"[{c.start_time:.2f}s, {c.end_time:.2f}s], "
              f"shape={c.samples.shape}, padded={c.is_padded}")
    assert len(clips) == 3, f"Esperaba 3 clips, obtuve {len(clips)}"
    assert all(c.samples.shape == (5 * SR,) for c in clips)
    print("  PASS")

    # ---- Test 2: Audio largo con sobrante (12s, 50% solape) ----
    print("\n[Test 2] Audio de 12s con 50% de solape:")
    audio_med = np.random.randn(12 * SR).astype(np.float32)
    clips = segment_audio(
        audio_med,
        sample_rate=SR,
        source_file="test_medio.wav",
        overlap_ratio=0.5,
    )
    print(f"  Clips generados: {len(clips)}")
    for c in clips:
        print(f"    clip {c.clip_index}: "
              f"[{c.start_time:.2f}s, {c.end_time:.2f}s], padded={c.is_padded}")
    # Con L=12, T_c=5, h=2.5: N = floor(7/2.5)+1 = 3 clips
    # Sobrante = 12 - (2*2.5+5) = 2s >= MIN_LEFTOVER_SEC=2s -> 1 clip mas
    assert len(clips) >= 3, f"Esperaba >=3 clips, obtuve {len(clips)}"
    print("  PASS")

    # ---- Test 3: Audio corto (3s) con padding circular ----
    print("\n[Test 3] Audio corto de 3s (padding circular):")
    audio_short = np.random.randn(3 * SR).astype(np.float32)
    clips = segment_audio(
        audio_short,
        sample_rate=SR,
        source_file="test_corto.wav",
        short_audio_strategy="wrap",
    )
    assert len(clips) == 1
    assert clips[0].samples.shape == (5 * SR,)
    assert clips[0].is_padded is True
    print(f"  Audio entrada: 3s -> 1 clip de 5s (padded)")
    print(f"  is_padded: {clips[0].is_padded}")
    print("  PASS")

    # ---- Test 4: Audio corto con strategy="drop" ----
    print("\n[Test 4] Audio corto con strategy='drop':")
    clips = segment_audio(
        audio_short,
        sample_rate=SR,
        short_audio_strategy="drop",
    )
    assert clips == []
    print(f"  Resultado: {len(clips)} clips (esperado: 0)")
    print("  PASS")

    # ---- Test 5: Validación de errores ----
    print("\n[Test 5] Validación de errores:")
    try:
        segment_audio(np.array([]), sample_rate=SR)
        print("  FAIL: debio lanzar SegmentationError")
    except SegmentationError as e:
        print(f"  PASS: array vacio detectado -> {e}")

    try:
        segment_audio(audio_long, sample_rate=SR, overlap_ratio=1.0)
        print("  FAIL: debio lanzar SegmentationError")
    except SegmentationError as e:
        print(f"  PASS: overlap invalido detectado -> {e}")

    print("\n" + "=" * 70)
    print("TODOS LOS TESTS PASARON")
    print("=" * 70)
