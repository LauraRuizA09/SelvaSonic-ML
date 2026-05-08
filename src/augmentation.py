"""
src/augmentation.py
====================

Data augmentation para audio: técnicas que generan variaciones sintéticas
de los clips de entrenamiento para reducir overfitting y mejorar la
generalización del modelo a condiciones reales de campo.

¿Por qué augmentation?
----------------------
Tenemos ~334 audios de Xeno-canto. Tras segmentar (con solape 50%) tendremos
~1500-3000 clips. Para una CNN moderna eso sigue siendo poco. Una red mal
entrenada en pocos datos memoriza el training set (overfitting): accuracy
alta en train, baja en val/test.

Augmentation crea "audios nuevos" modificando los existentes de forma que
SIGUEN SIENDO CANTOS VÁLIDOS de la misma especie. Es como tomar fotos
desde ángulos distintos del mismo animal.

Tres técnicas (las del cronograma de Semana 2):
- time_stretch: cambia velocidad sin alterar tono (phase vocoder)
- pitch_shift:  cambia tono sin alterar velocidad (resampling + stretch)
- add_noise:    suma ruido gaussiano con SNR controlado en decibelios

REGLA DE ORO
------------
Aplicar SOLO en TRAIN. Aplicar en VAL/TEST contaminaría las métricas con
aleatoriedad y haría imposible comparar modelos.

Reproducibilidad
----------------
Cada función acepta un parámetro opcional `random_state`:
- random_state=None  -> aleatoriedad pura (training real, distinto cada vez)
- random_state=42    -> reproducible (experimentos académicos comparables)

Es el patrón de scikit-learn: flexible y profesional.
"""

from __future__ import annotations

from typing import Optional, Union

import librosa
import numpy as np

# Constantes desde config.py (fuente única de verdad)
from src.config import (
    AUG_PROB_ADD_NOISE,
    AUG_PROB_PITCH_SHIFT,
    AUG_PROB_TIME_STRETCH,
    CLIP_NUM_SAMPLES,
    PITCH_SHIFT_RANGE,
    SNR_DB_RANGE,
    TIME_STRETCH_RANGE,
)

# Tipo para random_state: int (semilla), Generator (numpy moderno) o None
RandomState = Optional[Union[int, np.random.Generator]]


# ============================================================================
# Excepciones
# ============================================================================

class AugmentationError(Exception):
    """Error específico de las funciones de augmentation."""


# ============================================================================
# Helpers privados
# ============================================================================

def _resolve_rng(random_state: RandomState) -> np.random.Generator:
    """
    Convierte random_state al objeto Generator moderno de numpy.

    Acepta:
        - None        -> nuevo Generator con seed aleatoria del sistema
        - int         -> Generator con seed fija (reproducible)
        - Generator   -> se devuelve tal cual (encadenamiento)

    Por qué np.random.Generator y NO np.random.seed():
        - np.random.seed() es global: contamina el resto del programa.
        - Generator es local: solo afecta a esta función.
        - Es el método recomendado desde NumPy 1.17 (2019).
    """
    if random_state is None:
        return np.random.default_rng()
    if isinstance(random_state, int):
        return np.random.default_rng(random_state)
    if isinstance(random_state, np.random.Generator):
        return random_state
    raise AugmentationError(
        f"random_state debe ser None, int, o np.random.Generator. "
        f"Recibido: {type(random_state).__name__}"
    )


def _validate_samples(samples: np.ndarray, function_name: str) -> None:
    """Validaciones comunes para arrays de entrada."""
    if not isinstance(samples, np.ndarray):
        raise AugmentationError(
            f"{function_name}: samples debe ser np.ndarray, "
            f"recibido {type(samples).__name__}"
        )
    if samples.ndim != 1:
        raise AugmentationError(
            f"{function_name}: samples debe ser 1D, "
            f"shape recibido: {samples.shape}"
        )
    if samples.size == 0:
        raise AugmentationError(f"{function_name}: samples está vacío")


def _ensure_length(samples: np.ndarray, target_length: int) -> np.ndarray:
    """
    Garantiza que el array tenga exactamente target_length muestras.

    Por qué es necesario:
        time_stretch CAMBIA la longitud del array (más rápido = más corto).
        Pero la CNN espera inputs de tamaño fijo (CLIP_NUM_SAMPLES).
        Después de cualquier augmentation que cambie longitud, debemos
        normalizar de vuelta al tamaño esperado.

    Estrategia:
        - Si el resultado es más LARGO -> truncar al inicio.
        - Si el resultado es más CORTO -> repetir circularmente (wrap).
          Coherente con la estrategia de segmentation.py.
    """
    n = samples.shape[0]
    if n == target_length:
        return samples
    if n > target_length:
        return samples[:target_length]
    repetitions = int(np.ceil(target_length / n))
    return np.tile(samples, repetitions)[:target_length]


# ============================================================================
# Augmentation 1: Time Stretch (cambio de velocidad sin tono)
# ============================================================================

def time_stretch(
    samples: np.ndarray,
    *,
    sample_rate: int,
    rate: Optional[float] = None,
    rate_range: tuple[float, float] = TIME_STRETCH_RANGE,
    random_state: RandomState = None,
    target_length: int = CLIP_NUM_SAMPLES,
) -> np.ndarray:
    """
    Aplica time stretch al audio (cambio de velocidad sin alterar el tono).

    Algoritmo subyacente: phase vocoder
        1. STFT del audio
        2. Reescalar el eje temporal en factor 1/rate
        3. Ajustar fases para mantener continuidad
        4. ISTFT para reconstruir

    Implementación: librosa.effects.time_stretch

    Parámetros
    ----------
    samples : np.ndarray
        Audio 1D de entrada (típicamente CLIP_NUM_SAMPLES muestras).
    sample_rate : int
        Frecuencia de muestreo (no usada por librosa para esta función,
        pero se incluye por consistencia con la API del proyecto).
    rate : float, opcional
        Factor de stretch específico:
            < 1.0 -> audio más LARGO (canto lento)
            = 1.0 -> sin cambio
            > 1.0 -> audio más CORTO (canto rápido)
        Si es None, se elige aleatoriamente del rate_range.
    rate_range : tuple[float, float]
        Rango (min, max) del cual samplear si rate es None.
    random_state : None | int | np.random.Generator
        Control de aleatoriedad. None = aleatorio puro.
    target_length : int
        Longitud final del array (en muestras).

    Retorna
    -------
    np.ndarray
        Audio aumentado con shape (target_length,).
    """
    _validate_samples(samples, "time_stretch")

    rng = _resolve_rng(random_state)

    if rate is None:
        rate = float(rng.uniform(rate_range[0], rate_range[1]))

    if rate <= 0:
        raise AugmentationError(
            f"time_stretch: rate debe ser > 0, recibido {rate}"
        )
    if rate < 0.5 or rate > 2.0:
        raise AugmentationError(
            f"time_stretch: rate={rate} está fuera del rango seguro [0.5, 2.0]. "
            f"Valores extremos producen artefactos audibles."
        )

    stretched = librosa.effects.time_stretch(samples.astype(np.float32), rate=rate)
    return _ensure_length(stretched, target_length)


# ============================================================================
# Augmentation 2: Pitch Shift (cambio de tono sin velocidad)
# ============================================================================

def pitch_shift(
    samples: np.ndarray,
    *,
    sample_rate: int,
    n_semitones: Optional[float] = None,
    semitones_range: tuple[int, int] = PITCH_SHIFT_RANGE,
    random_state: RandomState = None,
    target_length: int = CLIP_NUM_SAMPLES,
) -> np.ndarray:
    """
    Aplica pitch shift al audio (cambio de tono sin alterar duración).

    Algoritmo:
        1. Time stretch por factor 2^(-n/12)
        2. Resample por factor 2^(n/12)
        3. Resultado: misma duración, frecuencias multiplicadas por 2^(n/12)

    Donde n es el número de semitones (12 semitones = 1 octava).

    Parámetros
    ----------
    samples : np.ndarray
        Audio 1D de entrada.
    sample_rate : int
        Frecuencia de muestreo.
    n_semitones : float, opcional
        Número de semitones a desplazar:
            < 0 -> más GRAVE
            = 0 -> sin cambio
            > 0 -> más AGUDO
        Si es None, se elige aleatoriamente del semitones_range.
    semitones_range : tuple[int, int]
        Rango (min, max) del cual samplear si n_semitones es None.
    random_state : None | int | np.random.Generator
        Control de aleatoriedad.
    target_length : int
        Longitud final del array.
    """
    _validate_samples(samples, "pitch_shift")

    rng = _resolve_rng(random_state)

    if n_semitones is None:
        n_semitones = float(
            rng.uniform(semitones_range[0], semitones_range[1])
        )

    if abs(n_semitones) > 12:
        raise AugmentationError(
            f"pitch_shift: n_semitones={n_semitones} excede ±12 (una octava). "
            f"Valores tan grandes destruyen la firma espectral del ave."
        )

    shifted = librosa.effects.pitch_shift(
        samples.astype(np.float32),
        sr=sample_rate,
        n_steps=n_semitones,
    )

    return _ensure_length(shifted, target_length)


# ============================================================================
# Augmentation 3: Add Noise (ruido gaussiano con SNR controlado)
# ============================================================================

def add_noise(
    samples: np.ndarray,
    *,
    snr_db: Optional[float] = None,
    snr_db_range: tuple[float, float] = SNR_DB_RANGE,
    random_state: RandomState = None,
) -> np.ndarray:
    """
    Suma ruido gaussiano al audio con un SNR controlado en decibelios.

    Matemática:
        SNR_dB = 10 * log10(P_señal / P_ruido)
        P = (1/N) * sum(x[i]^2)   [potencia = energía media]

    Para inyectar ruido a un SNR específico:
        P_ruido_objetivo = P_señal / 10^(SNR/10)
        ruido ~ N(0, sqrt(P_ruido_objetivo))

    Por qué SNR y no "ruido aleatorio sin más":
        Sumar ruido sin medir puede no tener efecto (ruido tan bajo que
        nada cambia) o destruir la señal (ruido más fuerte que la señal).
        El SNR garantiza una proporción CONOCIDA y reproducible.

    Parámetros
    ----------
    samples : np.ndarray
        Audio 1D de entrada (señal limpia).
    snr_db : float, opcional
        SNR objetivo en decibelios:
            30 dB -> ruido apenas audible
            20 dB -> ruido moderado
            10 dB -> ruido fuerte (límite reconocible)
        Si es None, se elige aleatoriamente del snr_db_range.
    snr_db_range : tuple[float, float]
        Rango si snr_db es None.
    random_state : None | int | np.random.Generator
        Control de aleatoriedad.
    """
    _validate_samples(samples, "add_noise")

    rng = _resolve_rng(random_state)

    if snr_db is None:
        snr_db = float(rng.uniform(snr_db_range[0], snr_db_range[1]))

    # Potencia de la señal
    signal_power = float(np.mean(samples ** 2))

    # Edge case: señal silenciosa
    if signal_power == 0:
        return samples.copy()

    # Potencia objetivo del ruido a partir del SNR
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear

    # Generar ruido gaussiano: para N(0, sigma), E[x^2] = sigma^2
    noise_std = np.sqrt(noise_power)
    noise = rng.normal(loc=0.0, scale=noise_std, size=samples.shape).astype(samples.dtype)

    return samples + noise


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SMOKE TEST: src/augmentation.py")
    print("=" * 70)
    print(f"\nUsando constantes de config.py:")
    print(f"  TIME_STRETCH_RANGE    = {TIME_STRETCH_RANGE}")
    print(f"  PITCH_SHIFT_RANGE     = {PITCH_SHIFT_RANGE}")
    print(f"  SNR_DB_RANGE          = {SNR_DB_RANGE}")
    print(f"  AUG_PROB_TIME_STRETCH = {AUG_PROB_TIME_STRETCH}")
    print(f"  AUG_PROB_PITCH_SHIFT  = {AUG_PROB_PITCH_SHIFT}")
    print(f"  AUG_PROB_ADD_NOISE    = {AUG_PROB_ADD_NOISE}")

    SR = 22050
    # Audio sintético: tono puro 440 Hz por 5 segundos (LA musical)
    t = np.arange(CLIP_NUM_SAMPLES) / SR
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    print(f"\nAudio de prueba: tono 440Hz, {CLIP_NUM_SAMPLES} muestras "
          f"({CLIP_NUM_SAMPLES/SR:.1f}s)")

    # ---- Test 1: time_stretch con rate fijo ----
    print("\n[Test 1] time_stretch rate=1.2 (mas rapido):")
    aug1 = time_stretch(audio, sample_rate=SR, rate=1.2)
    print(f"  Input shape:  {audio.shape}")
    print(f"  Output shape: {aug1.shape}")
    assert aug1.shape == (CLIP_NUM_SAMPLES,)
    print("  PASS")

    # ---- Test 2: time_stretch aleatorio ----
    print("\n[Test 2] time_stretch aleatorio (rango por defecto):")
    aug2 = time_stretch(audio, sample_rate=SR)
    assert aug2.shape == (CLIP_NUM_SAMPLES,)
    print(f"  Output shape: {aug2.shape}")
    print("  PASS")

    # ---- Test 3: pitch_shift +2 semitones ----
    print("\n[Test 3] pitch_shift n_semitones=+2 (mas agudo):")
    aug3 = pitch_shift(audio, sample_rate=SR, n_semitones=2)
    assert aug3.shape == (CLIP_NUM_SAMPLES,)
    expected_freq = 440 * 2 ** (2/12)
    print(f"  Frecuencia esperada: 440Hz -> ~{expected_freq:.1f} Hz")
    print("  PASS")

    # ---- Test 4: pitch_shift -2 semitones ----
    print("\n[Test 4] pitch_shift n_semitones=-2 (mas grave):")
    aug4 = pitch_shift(audio, sample_rate=SR, n_semitones=-2)
    expected_freq = 440 * 2 ** (-2/12)
    print(f"  Frecuencia esperada: 440Hz -> ~{expected_freq:.1f} Hz")
    print("  PASS")

    # ---- Test 5: add_noise SNR=20dB (verificacion matematica) ----
    print("\n[Test 5] add_noise snr_db=20 (verificacion matematica):")
    noisy = add_noise(audio, snr_db=20.0, random_state=42)
    signal_power = np.mean(audio ** 2)
    noise = noisy - audio
    noise_power = np.mean(noise ** 2)
    measured_snr = 10 * np.log10(signal_power / noise_power)
    print(f"  SNR objetivo: 20.00 dB")
    print(f"  SNR medido:   {measured_snr:.2f} dB")
    assert abs(measured_snr - 20.0) < 0.5
    print("  PASS")

    # ---- Test 6: add_noise SNR=10dB (mucho ruido) ----
    print("\n[Test 6] add_noise snr_db=10 (ruido fuerte):")
    noisy_strong = add_noise(audio, snr_db=10.0, random_state=42)
    noise = noisy_strong - audio
    measured_snr = 10 * np.log10(signal_power / np.mean(noise ** 2))
    print(f"  SNR objetivo: 10.00 dB, medido: {measured_snr:.2f} dB")
    assert abs(measured_snr - 10.0) < 0.5
    print("  PASS")

    # ---- Test 7: Reproducibilidad ----
    print("\n[Test 7] Reproducibilidad (seed=42 -> mismo output):")
    a = time_stretch(audio, sample_rate=SR, random_state=42)
    b = time_stretch(audio, sample_rate=SR, random_state=42)
    assert np.array_equal(a, b)
    print("  time_stretch: 2 corridas con seed=42 -> identicas")

    c = pitch_shift(audio, sample_rate=SR, random_state=42)
    d = pitch_shift(audio, sample_rate=SR, random_state=42)
    assert np.array_equal(c, d)
    print("  pitch_shift:  2 corridas con seed=42 -> identicas")

    e = add_noise(audio, random_state=42)
    f = add_noise(audio, random_state=42)
    assert np.array_equal(e, f)
    print("  add_noise:    2 corridas con seed=42 -> identicas")
    print("  PASS")

    # ---- Test 8: Aleatoriedad real (sin seed) ----
    print("\n[Test 8] Aleatoriedad real (sin seed -> diferente cada vez):")
    x = time_stretch(audio, sample_rate=SR)
    y = time_stretch(audio, sample_rate=SR)
    assert not np.array_equal(x, y)
    print("  PASS (cada llamada genera una variacion distinta)")

    # ---- Test 9: Pipeline combinado (como sera en el Dataset) ----
    print("\n[Test 9] Pipeline combinado (como funcionara en el Dataset):")
    rng = np.random.default_rng(seed=123)
    clip = audio.copy()

    if rng.random() < AUG_PROB_TIME_STRETCH:
        clip = time_stretch(clip, sample_rate=SR, random_state=rng)
        print("  -> time_stretch aplicado")
    if rng.random() < AUG_PROB_PITCH_SHIFT:
        clip = pitch_shift(clip, sample_rate=SR, random_state=rng)
        print("  -> pitch_shift aplicado")
    if rng.random() < AUG_PROB_ADD_NOISE:
        clip = add_noise(clip, random_state=rng)
        print("  -> add_noise aplicado")

    assert clip.shape == (CLIP_NUM_SAMPLES,)
    print(f"  Shape final: {clip.shape}")
    print("  PASS")

    # ---- Test 10: Validacion de errores ----
    print("\n[Test 10] Validacion de errores:")
    try:
        time_stretch(np.array([]), sample_rate=SR)
        print("  FAIL")
    except AugmentationError as e:
        print(f"  PASS: array vacio -> {str(e)[:60]}")

    try:
        time_stretch(audio, sample_rate=SR, rate=10.0)
        print("  FAIL")
    except AugmentationError as e:
        print(f"  PASS: rate fuera de rango -> {str(e)[:60]}")

    try:
        pitch_shift(audio, sample_rate=SR, n_semitones=24)
        print("  FAIL")
    except AugmentationError as e:
        print(f"  PASS: semitones fuera de rango -> {str(e)[:60]}")

    print("\n" + "=" * 70)
    print("TODOS LOS TESTS PASARON")
    print("=" * 70)
