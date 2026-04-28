"""
src/transforms.py
=================

Módulo de transformaciones y feature extraction para SelvaSonic.

Convierte señales de audio (waveform 1D) en representaciones que el
modelo CNN + Attention puede consumir directamente como tensores PyTorch.

Pipeline conceptual
-------------------
::

    waveform (np.ndarray, 1D)
        │
        ▼  compute_mel_spectrogram()
        │
    log-Mel spectrogram (np.ndarray, 2D, dB)
        │
        ▼  normalize_spectrogram()
        │
    Mel normalizado (np.ndarray, 2D, mean=0 std=1)
        │
        ▼  to_tensor()
        │
    torch.Tensor [1, 128, T]  ← LISTO PARA EL CNN

La función `waveform_to_mel_tensor()` ejecuta toda la cadena de una vez.

API pública
-----------
- compute_mel_spectrogram : extracción del Mel-spectrograma en dB.
- normalize_spectrogram   : z-score per-instance para que mean=0, std=1.
- to_tensor               : np.ndarray 2D → torch.Tensor 3D [1, H, W].
- waveform_to_mel_tensor  : pipeline completo en una llamada (atajo).

Responsabilidades
-----------------
- Calcular Mel-spectrogramas en dB con los hiperparámetros del proyecto.
- Normalizar espectrogramas (z-score per-instance).
- Convertir np.ndarray → torch.Tensor con la forma correcta para el CNN.

Este módulo NO hace:
- Carga de audio desde disco (eso va en audio_io.py).
- Segmentación en clips (eso va en dataset.py).
- Data augmentation (se añadirá más adelante en este mismo archivo).

Autoras(es)
-----------
Laura Ruiz Arango & Jose Aldair Molina Méndez
Universidad Nacional de Colombia - Sede Medellín
Aprendizaje Automático (Prof. Alcides Montoya) - 2026
"""

from __future__ import annotations

# --- Librerías estándar ---
# (ninguna por ahora)

# --- Librerías de terceros ---
import numpy as np
import librosa
import torch

# --- Configuración del proyecto ---
from src.config import (
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    FMIN,
    FMAX,
)


# =============================================================================
# CONSTANTES INTERNAS DE ESTE MÓDULO
# =============================================================================

_TOP_DB: float = 80.0
"""Rango dinámico máximo del Mel-spectrograma en dB."""

_POWER: float = 2.0
"""Potencia aplicada antes del filterbank Mel (2.0 = power, 1.0 = magnitude)."""

_NORM_EPSILON: float = 1e-8
"""Constante para evitar división por cero en z-score normalization."""


# =============================================================================
# 1. EXTRACCIÓN DE MEL-SPECTROGRAMA
# =============================================================================

def compute_mel_spectrogram(
    waveform: np.ndarray,
    *,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> np.ndarray:
    """Calcula el Mel-spectrograma en dB de una señal de audio.

    Pipeline interno:
        1. STFT con ventana Hann (n_fft, hop_length)
        2. Magnitud al cuadrado → power spectrogram
        3. Multiplicación por banco de filtros Mel (n_mels filtros)
        4. Conversión a escala logarítmica (dB) con truncamiento a -80 dB

    Parameters
    ----------
    waveform : np.ndarray
        Señal de audio 1D, mono, float32, en el rango [-1, 1].
    sr, n_fft, hop_length, n_mels, fmin, fmax : opcionales
        Hiperparámetros. Los defaults provienen de src/config.py.

    Returns
    -------
    np.ndarray
        Mel-spectrograma en dB. Shape: (n_mels, n_frames). Rango: [-80, 0].

    Raises
    ------
    ValueError
        Si el waveform está vacío o no es 1D.
    """
    # 1. Validación de entrada
    if waveform.ndim != 1:
        raise ValueError(
            f"compute_mel_spectrogram espera waveform 1D, "
            f"pero recibió shape {waveform.shape} (ndim={waveform.ndim}). "
            f"Si tu audio es estéreo, conviértelo a mono primero."
        )

    if waveform.size == 0:
        raise ValueError(
            "compute_mel_spectrogram recibió un waveform vacío (0 muestras)."
        )

    # 2. Mel-spectrograma en escala de potencia
    mel_power = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=_POWER,
    )

    # 3. Potencia → dB
    mel_db = librosa.power_to_db(
        mel_power,
        ref=np.max,
        top_db=_TOP_DB,
    )

    # 4. Asegurar float32 (PyTorch espera float32, no float64)
    if mel_db.dtype != np.float32:
        mel_db = mel_db.astype(np.float32)

    return mel_db


# =============================================================================
# 2. NORMALIZACIÓN DEL ESPECTROGRAMA
# =============================================================================

def normalize_spectrogram(
    spectrogram: np.ndarray,
    *,
    epsilon: float = _NORM_EPSILON,
) -> np.ndarray:
    """Normaliza un espectrograma a media 0 y desviación estándar 1.

    Aplica z-score normalization PER-INSTANCE (cada espectrograma con sus
    propias estadísticas), no global. Esto elimina variabilidad espuria
    de las grabaciones (distancia al micrófono, ganancia, ruido base).

    Fórmula:
        S_norm = (S - mean(S)) / (std(S) + epsilon)

    Parameters
    ----------
    spectrogram : np.ndarray
        Espectrograma 2D, típicamente shape (n_mels, n_frames).
    epsilon : float, optional
        Estabilidad numérica (evita división por 0). Por defecto, 1e-8.

    Returns
    -------
    np.ndarray
        Espectrograma normalizado, mismo shape. mean ≈ 0, std ≈ 1.

    Raises
    ------
    ValueError
        Si el espectrograma está vacío o no es 2D.
    """
    if spectrogram.ndim != 2:
        raise ValueError(
            f"normalize_spectrogram espera espectrograma 2D, "
            f"pero recibió shape {spectrogram.shape} (ndim={spectrogram.ndim})."
        )

    if spectrogram.size == 0:
        raise ValueError("normalize_spectrogram recibió un espectrograma vacío.")

    mean = spectrogram.mean()
    std = spectrogram.std()

    normalized = (spectrogram - mean) / (std + epsilon)

    if normalized.dtype != np.float32:
        normalized = normalized.astype(np.float32)

    return normalized


# =============================================================================
# 3. CONVERSIÓN A TENSOR PYTORCH
# =============================================================================

def to_tensor(spectrogram: np.ndarray) -> torch.Tensor:
    """Convierte un espectrograma 2D (numpy) a tensor 3D (PyTorch).

    Añade una dimensión de canal al inicio para cumplir con la convención
    [C, H, W] que esperan las CNNs en PyTorch:
        Input  shape: (n_mels, n_frames)        ej: (128, 216)
        Output shape: (1, n_mels, n_frames)     ej: (1, 128, 216)

    Parameters
    ----------
    spectrogram : np.ndarray
        Espectrograma 2D, dtype float32.

    Returns
    -------
    torch.Tensor
        Tensor 3D con shape (1, n_mels, n_frames), dtype float32.

    Raises
    ------
    ValueError
        Si el espectrograma no es 2D o está vacío.

    Notes
    -----
    Usamos `torch.from_numpy()` (zero-copy) por velocidad: no copia los
    datos, solo crea una vista del array de numpy como tensor.
    """
    if spectrogram.ndim != 2:
        raise ValueError(
            f"to_tensor espera espectrograma 2D, "
            f"pero recibió shape {spectrogram.shape} (ndim={spectrogram.ndim})."
        )

    if spectrogram.size == 0:
        raise ValueError("to_tensor recibió un espectrograma vacío.")

    if spectrogram.dtype != np.float32:
        spectrogram = spectrogram.astype(np.float32)

    # numpy 2D → torch 2D (sin copia)
    tensor_2d = torch.from_numpy(spectrogram)

    # Añadir dim de canal: (H, W) → (1, H, W)
    tensor_3d = tensor_2d.unsqueeze(0)

    return tensor_3d


# =============================================================================
# 4. PIPELINE COMPLETO (FACHADA)
# =============================================================================

def waveform_to_mel_tensor(
    waveform: np.ndarray,
    *,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
    fmin: float = FMIN,
    fmax: float = FMAX,
    normalize: bool = True,
) -> torch.Tensor:
    """Pipeline completo: waveform de audio → tensor PyTorch listo para CNN.

    Esta es la función "fachada" que encapsula los 3 pasos del pipeline
    en una sola llamada. Es el atajo que usaremos en el Dataset
    (Semana 2 más adelante) para alimentar el modelo durante entrenamiento.

    Pipeline ejecutado::

        waveform (1D)
            │
            ▼  compute_mel_spectrogram(...)
            │
        Mel-dB (2D, n_mels × n_frames)
            │
            ▼  normalize_spectrogram(...)   [opcional, default=True]
            │
        Mel normalizado (2D, mean=0 std=1)
            │
            ▼  to_tensor(...)
            │
        Tensor PyTorch (3D, [1, n_mels, n_frames])

    Parameters
    ----------
    waveform : np.ndarray
        Señal de audio 1D, mono, float32. Típicamente proviene de
        `audio_io.load_audio(...).waveform`.
    sr, n_fft, hop_length, n_mels, fmin, fmax : opcionales
        Hiperparámetros del Mel-spectrograma. Defaults desde config.py.
    normalize : bool, optional
        Si es True (default), aplica z-score normalization per-instance.
        Útil para alternar durante experimentos:
            - True: input ideal para CNN entrenable.
            - False: tensor con valores en dB crudos (útil para debugging).

    Returns
    -------
    torch.Tensor
        Tensor 3D con shape (1, n_mels, n_frames), dtype float32,
        listo para alimentar a `torch.nn.Conv2d`.

    Raises
    ------
    ValueError
        Si el waveform es vacío o no es 1D (heredado de las funciones internas).

    Examples
    --------
    >>> from src.audio_io import load_audio
    >>> from src.transforms import waveform_to_mel_tensor
    >>>
    >>> audio = load_audio("data/raw/Lipaugus_vociferans/XC...mp3")
    >>> tensor = waveform_to_mel_tensor(audio.waveform)
    >>> tensor.shape
    torch.Size([1, 128, 502])
    >>> tensor.dtype
    torch.float32

    Notes
    -----
    Esta función es el atajo conveniente para el caso común (Dataset).
    Si necesitas hacer pasos intermedios (e.g., visualizar el Mel antes
    de normalizar, o aplicar augmentation entre pasos), llama a las
    funciones internas (`compute_mel_spectrogram`, etc.) directamente.

    Es el patrón de diseño "facade" — interfaz simple para el caso común,
    interfaz detallada disponible para casos avanzados.
    """
    # Paso 1: extraer Mel-spectrograma en dB
    mel_db = compute_mel_spectrogram(
        waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )

    # Paso 2: normalizar (opcional, default True)
    if normalize:
        mel_db = normalize_spectrogram(mel_db)

    # Paso 3: convertir a tensor PyTorch con shape [1, H, W]
    tensor = to_tensor(mel_db)

    return tensor


# =============================================================================
# SMOKE TEST
# =============================================================================
# Ejecutable con: python -m src.transforms

if __name__ == "__main__":
    import sys
    from pathlib import Path

    from src.audio_io import load_audio, summarize, SUPPORTED_EXTENSIONS

    print("=" * 70)
    print("SMOKE TEST — pipeline completo de transforms")
    print("=" * 70)

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
    print(f"\n📂 Audio de prueba: {test_path.name}")
    print(f"   Especie: {test_path.parent.name}")

    # ─── Paso a paso (uso "manual") ─────────────────────────────────────
    print("\n" + "─" * 70)
    print("MODO 1 — uso paso a paso (3 funciones)")
    print("─" * 70)

    audio = load_audio(test_path)
    print(f"\n🎵 [1/4] Audio cargado:")
    print(f"   {summarize(audio)}")

    mel = compute_mel_spectrogram(audio.waveform)
    print(f"\n🎼 [2/4] Mel-spectrograma calculado:")
    print(f"   shape: {mel.shape}  dtype: {mel.dtype}")
    print(f"   min: {mel.min():.2f} dB   max: {mel.max():.2f} dB")
    print(f"   mean: {mel.mean():.2f} dB   std: {mel.std():.2f} dB")

    mel_norm = normalize_spectrogram(mel)
    print(f"\n📊 [3/4] Espectrograma normalizado (z-score per-instance):")
    print(f"   shape: {mel_norm.shape}  dtype: {mel_norm.dtype}")
    print(f"   min: {mel_norm.min():.4f}   max: {mel_norm.max():.4f}")
    print(f"   mean: {mel_norm.mean():.4f}   std: {mel_norm.std():.4f}")
    print(f"   ✅ mean ≈ 0 y std ≈ 1 → input ideal para CNN")

    tensor = to_tensor(mel_norm)
    print(f"\n🔥 [4/4] Tensor PyTorch:")
    print(f"   shape: {tuple(tensor.shape)}  dtype: {tensor.dtype}")
    print(f"   tipo: {type(tensor).__name__}")
    print(f"   device: {tensor.device}")

    # ─── Pipeline completo (uso "atajo") ────────────────────────────────
    print("\n" + "─" * 70)
    print("MODO 2 — pipeline completo en 1 línea (waveform_to_mel_tensor)")
    print("─" * 70)

    tensor_atajo = waveform_to_mel_tensor(audio.waveform)
    print(f"\n⚡ Una sola llamada produce:")
    print(f"   shape: {tuple(tensor_atajo.shape)}")
    print(f"   dtype: {tensor_atajo.dtype}")
    print(f"   mean:  {tensor_atajo.mean().item():.4f}")
    print(f"   std:   {tensor_atajo.std().item():.4f}")

    # Verificar que MODO 1 y MODO 2 producen el mismo resultado
    coinciden = torch.allclose(tensor, tensor_atajo, atol=1e-6)
    print(f"\n🧪 ¿Modo manual y modo atajo producen el mismo tensor?")
    print(f"   {'✅ SÍ — son idénticos' if coinciden else '❌ NO — hay discrepancia'}")

    # ─── Test del flag normalize=False ──────────────────────────────────
    print("\n" + "─" * 70)
    print("MODO 3 — pipeline SIN normalización (para debugging)")
    print("─" * 70)

    tensor_raw = waveform_to_mel_tensor(audio.waveform, normalize=False)
    print(f"\n🔍 Con normalize=False:")
    print(f"   shape: {tuple(tensor_raw.shape)}")
    print(f"   mean:  {tensor_raw.mean().item():.2f} dB  (≠ 0, en dB crudo)")
    print(f"   std:   {tensor_raw.std().item():.2f} dB  (≠ 1, escala original)")

    # ─── Verificación final del shape ───────────────────────────────────
    expected_shape = (1, N_MELS, mel.shape[1])
    print(f"\n🧪 Verificación del shape final:")
    print(f"   Esperado: {expected_shape}  (1 canal × {N_MELS} bandas × {mel.shape[1]} frames)")
    print(f"   Obtenido: {tuple(tensor.shape)}")
    if tuple(tensor.shape) == expected_shape:
        print(f"   ✅ Coinciden — listo para Conv2D")
    else:
        print(f"   ❌ Difieren — revisar pipeline")

    print("\n✅ Smoke test completado")
    print("\n💡 Resumen: ahora puedes usar tanto las 3 funciones por separado como")
    print("   waveform_to_mel_tensor() para el pipeline completo en una sola llamada.")
