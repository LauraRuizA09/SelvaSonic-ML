"""
src/inference.py
================

Motor de inferencia de SelvaSonic: recibe un archivo de audio y devuelve
la(s) especie(s) predicha(s) con confianza calibrada por umbral.

Flujo de uso típico (API):
    >>> from src.inference import predict_audio
    >>> result = predict_audio(
    ...     "grabacion.wav",
    ...     model_path="results/runs/attention_S4_v1_20260601/best.pth",
    ... )
    >>> print(result.aggregated_species, f"{result.aggregated_confidence:.0%}")
    Ramphastos_tucanus 91%

Flujo de uso típico (CLI):
    python -m src.inference --audio grabacion.wav --model best.pth
    python -m src.inference --batch carpeta/ --model best.pth --threshold 0.7

Pipeline interno (por clip de CLIP_DURATION_S segundos):
    1. load_audio()              → waveform mono a SAMPLE_RATE Hz
    2. segment_audio()           → clips sin solapamiento (overlap=0 en inferencia)
    3. compute_mel_spectrogram() → Mel en dB (N_MELS × T)
    4. normalize_spectrogram()   → z-score per-instance (mismo que val/test)
    5. to_tensor()               → (1, N_MELS, T)
    6. model.forward()           → logits (num_classes,)
    7. F.softmax()               → probabilidades en [0, 1]
    8. umbral de confianza       → especie o 'no_identificado'
    9. agregación por moda       → predicción final del audio completo

Decisión de diseño — overlap=0 en inferencia:
    Durante el entrenamiento se usa overlap=50% para aumentar datos.
    En inferencia se usa overlap=0: cada segundo del audio se analiza
    exactamente una vez, la predicción de cada clip es independiente,
    y el resultado es fácil de interpretar sin redundancias.

Autoras(es)
-----------
Laura Ruiz Arango & Jose Aldair Molina Méndez
Universidad Nacional de Colombia - Sede Medellín
Aprendizaje Automático (Prof. Alcides Montoya) - 2026
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from src.audio_io import AudioLoadError, load_audio
from src.config import (
    CLIP_NUM_SAMPLES,
    SAMPLE_RATE,
    SUPPORTED_EXTENSIONS,
)
from src.model import SelvaSonicCNN, SelvaSonicCNNAttention
from src.segmentation import segment_audio
from src.transforms import compute_mel_spectrogram, normalize_spectrogram, to_tensor


# =============================================================================
# Constantes del módulo
# =============================================================================

DEFAULT_CONFIDENCE_THRESHOLD: float = 0.6
"""Confianza mínima para aceptar una predicción como válida.

Clips con softmax-max < threshold se reportan como 'no_identificado'.
Calibrado para el dataset de SelvaSonic (~330 audios, 11 clases):
    - Predicciones correctas suelen tener confianza > 0.7.
    - 0.6 deja un margen razonable sin disparar demasiados falsos positivos.

Nota: config.py tiene CONFIDENCE_THRESHOLD comentado como 'Semana 5-6'.
Se activa aquí como constante local con el valor planeado.
"""

_NO_ID_LABEL: str = "no_identificado"
"""Etiqueta devuelta cuando la confianza del clip cae bajo el threshold."""


# =============================================================================
# Estructuras de datos
# =============================================================================

class ClipPrediction(NamedTuple):
    """Predicción para un clip individual de CLIP_DURATION_S segundos.

    Attributes:
        clip_index: Índice 0-based del clip dentro del audio original.
        start_sec: Segundo de inicio del clip en el audio original.
        end_sec: Segundo de fin del clip en el audio original.
        species: Especie predicha, o 'no_identificado' si la confianza
            max del clip cayó por debajo del threshold.
        confidence: Probabilidad softmax del top-1 (en [0, 1]).
            Si species == 'no_identificado', es la confianza del top-1 raw
            que no superó el umbral (útil para saber qué tan cerca estuvo).
        all_probs: Probabilidad softmax de TODAS las clases. Suman 1.0.
            Útil para auditoría, debugging y segunda opinión del modelo.

    Example:
        >>> cp = ClipPrediction(
        ...     clip_index=0, start_sec=0.0, end_sec=5.0,
        ...     species="Ramphastos_tucanus", confidence=0.914,
        ...     all_probs={"no_ave": 0.021, "Ramphastos_tucanus": 0.914, ...},
        ... )
    """
    clip_index: int
    start_sec: float
    end_sec: float
    species: str
    confidence: float
    all_probs: dict[str, float]


class AudioPrediction(NamedTuple):
    """Predicción completa para un archivo de audio.

    Contiene las predicciones de cada clip de 5s y la predicción
    agregada (moda) del audio completo.

    Attributes:
        audio_path: Ruta absoluta del archivo de audio analizado.
        duration_sec: Duración total del audio en segundos.
        num_clips: Número de clips procesados.
        clip_predictions: Lista de ClipPrediction, uno por clip.
        aggregated_species: Especie más votada (moda) entre los clips
            identificados. 'no_identificado' si todos los clips quedaron
            bajo el threshold.
        aggregated_confidence: Confianza media de los clips que votaron
            por aggregated_species.
        model_name: Clase del modelo usada ("SelvaSonicCNN" o
            "SelvaSonicCNNAttention").
        threshold_used: Valor del umbral aplicado en esta predicción.

    Example:
        >>> result = predict_audio("grabacion.wav", model_path="best.pth")
        >>> result.aggregated_species
        'Ramphastos_tucanus'
    """
    audio_path: str
    duration_sec: float
    num_clips: int
    clip_predictions: list[ClipPrediction]
    aggregated_species: str
    aggregated_confidence: float
    model_name: str
    threshold_used: float


# =============================================================================
# Excepción personalizada
# =============================================================================

class InferenceError(Exception):
    """Error específico del pipeline de inferencia de SelvaSonic.

    Se lanza cuando:
    - El archivo de audio no existe o no se puede decodificar.
    - El checkpoint no existe, está corrupto o le faltan keys esperadas.
    - Los pesos son incompatibles con ambas arquitecturas conocidas.
    - El audio segmentado no produce ningún clip.
    """


# =============================================================================
# Helpers privados
# =============================================================================

def _resolve_device(device_str: str) -> torch.device:
    """Resuelve el string de device a torch.device.

    El valor "auto" detecta CUDA automáticamente. En el hardware del
    proyecto (laptop sin GPU) retornará CPU.

    Args:
        device_str: "auto", "cuda", "cpu" u otro string válido para
            torch.device.

    Returns:
        torch.device listo para usar en .to(device).
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _load_model_and_map(
    model_path: str | Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, int], str]:
    """Carga un checkpoint y devuelve (model, label_map, model_name).

    Estrategia de detección del tipo de modelo:
        1. Si ckpt['hparams']['model'] existe, se usa esa clase directamente.
           (los checkpoints de attention_S4_v1 incluyen esta key)
        2. Si no está declarado (baseline_S3 no la incluye), se prueba
           SelvaSonicCNNAttention primero (strict=True). Si los pesos dan
           RuntimeError por shape mismatch, se cae a SelvaSonicCNN.

    El orden en el fallback importa: Attention tiene más parámetros (714K
    vs 422K), un mismatch de Attention es más difícil de confundir con CNN.

    Args:
        model_path: Ruta al .pth generado por train.py.
        device: Device al que mover el modelo cargado.

    Returns:
        (model, label_map, model_name):
        - model: nn.Module en modo eval, listo para forward pass.
        - label_map: dict {nombre_especie: label_int} del checkpoint.
        - model_name: "SelvaSonicCNN" o "SelvaSonicCNNAttention".

    Raises:
        InferenceError: Si el archivo no existe, está corrupto, falta
            'label_map' o los pesos no coinciden con ninguna arquitectura.
    """
    path = Path(model_path)
    if not path.exists():
        raise InferenceError(f"Checkpoint no encontrado: {path}")

    try:
        ckpt = torch.load(str(path), map_location=device, weights_only=False)
    except Exception as e:
        raise InferenceError(
            f"No se pudo leer el checkpoint {path.name}: "
            f"{type(e).__name__}: {e}"
        ) from e

    for required_key in ("label_map", "model_state_dict"):
        if required_key not in ckpt:
            raise InferenceError(
                f"El checkpoint {path.name} no contiene '{required_key}'. "
                "Verifica que fue generado por train.py de SelvaSonic."
            )

    label_map: dict[str, int] = ckpt["label_map"]
    num_classes = len(label_map)
    declared: str | None = ckpt.get("hparams", {}).get("model", None)

    if declared in ("SelvaSonicCNN", "SelvaSonicCNNAttention"):
        # Tipo declarado explícitamente — no hay ambigüedad
        cls = SelvaSonicCNN if declared == "SelvaSonicCNN" else SelvaSonicCNNAttention
        model = cls(num_classes=num_classes)
        model_name: str = declared
        try:
            model.load_state_dict(ckpt["model_state_dict"])
        except RuntimeError as e:
            raise InferenceError(
                f"Los pesos del checkpoint no coinciden con {model_name}: {e}"
            ) from e
    else:
        # Tipo no declarado: probar Attention primero, CNN como fallback
        loaded_model: torch.nn.Module | None = None
        loaded_name: str = ""
        for cls, name in [
            (SelvaSonicCNNAttention, "SelvaSonicCNNAttention"),
            (SelvaSonicCNN, "SelvaSonicCNN"),
        ]:
            try:
                candidate = cls(num_classes=num_classes)
                candidate.load_state_dict(ckpt["model_state_dict"])
                loaded_model, loaded_name = candidate, name
                break
            except RuntimeError:
                continue

        if loaded_model is None:
            raise InferenceError(
                f"Los pesos de {path.name} son incompatibles con "
                "SelvaSonicCNNAttention y SelvaSonicCNN."
            )
        model, model_name = loaded_model, loaded_name

    model.to(device)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    best_acc = ckpt.get("best_val_acc", None)
    acc_str = f"{best_acc:.3f}" if isinstance(best_acc, float) else "?"
    print(
        f"  Modelo cargado: {model_name} | "
        f"epoch {epoch} | best_val_acc {acc_str} | device {device}"
    )

    return model, label_map, model_name


def _predict_clip(
    clip_samples: np.ndarray,
    *,
    model: torch.nn.Module,
    idx_to_name: dict[int, str],
    device: torch.device,
    sample_rate: int,
) -> tuple[str, float, dict[str, float]]:
    """Ejecuta un forward pass sobre muestras de un clip.

    Aplica exactamente el mismo preprocesamiento que el DataLoader en modo
    val/test: mel → normalize → tensor → unsqueeze batch. Sin augmentation.

    Args:
        clip_samples: Array 1D de muestras (shape ~= CLIP_NUM_SAMPLES).
        model: nn.Module en modo eval.
        idx_to_name: Mapa inverso {label_int: nombre_especie}.
        device: Device del modelo.
        sample_rate: Frecuencia de muestreo del clip.

    Returns:
        (species_raw, confidence, all_probs) donde species_raw es el top-1
        SIN aplicar threshold — el caller decide si reemplazar por
        'no_identificado'.
    """
    # Garantizar longitud exacta (segment_audio lo asegura, pero en
    # inferencia sobre audios externos podría llegar algo distinto)
    n = len(clip_samples)
    if n < CLIP_NUM_SAMPLES:
        reps = int(np.ceil(CLIP_NUM_SAMPLES / max(n, 1)))
        clip_samples = np.tile(clip_samples, reps)[:CLIP_NUM_SAMPLES]
    elif n > CLIP_NUM_SAMPLES:
        clip_samples = clip_samples[:CLIP_NUM_SAMPLES]

    mel = compute_mel_spectrogram(clip_samples, sr=sample_rate)
    mel_norm = normalize_spectrogram(mel)
    tensor = to_tensor(mel_norm)            # (1, N_MELS, T)
    batch = tensor.unsqueeze(0).to(device)  # (1, 1, N_MELS, T)

    with torch.no_grad():
        logits = model(batch)                       # (1, num_classes)
    probs = F.softmax(logits, dim=1).squeeze(0)     # (num_classes,)

    max_prob, max_idx = probs.max(dim=0)
    confidence = float(max_prob.item())
    species = idx_to_name[int(max_idx.item())]
    all_probs = {
        idx_to_name[i]: float(probs[i].item())
        for i in range(len(probs))
    }

    return species, confidence, all_probs


def _aggregate_clips(
    clip_preds: list[ClipPrediction],
) -> tuple[str, float]:
    """Calcula la predicción agregada por moda sobre los clips identificados.

    Algoritmo:
        1. Filtrar clips con species != 'no_identificado'.
        2. Si la lista queda vacía → ("no_identificado", 0.0).
        3. La especie ganadora es la moda (Counter.most_common(1)).
        4. La confianza agregada es la media de las confianzas de los clips
           que votaron por la moda.

    Justificación de la moda vs la media de probabilidades:
        La moda es más robusta a clips ruidosos que darían probabilidad
        fraccionada a varias clases. Un clip con alta confianza en Ramphastos
        importa igual que uno con baja confianza — lo que cuenta es el voto.
        La confianza media de los voters da una segunda señal cuantitativa.

    Args:
        clip_preds: ClipPredictions ya con threshold aplicado (species
            puede ser 'no_identificado').

    Returns:
        (aggregated_species, aggregated_confidence)
    """
    identified = [cp for cp in clip_preds if cp.species != _NO_ID_LABEL]

    if not identified:
        return _NO_ID_LABEL, 0.0

    vote_counts = Counter(cp.species for cp in identified)
    winner = vote_counts.most_common(1)[0][0]
    winner_confs = [cp.confidence for cp in identified if cp.species == winner]
    mean_conf = sum(winner_confs) / len(winner_confs)

    return winner, mean_conf


def _prediction_to_dict(result: AudioPrediction, *, timestamp: str) -> dict:
    """Convierte un AudioPrediction a dict JSON-serializable.

    Convierte las NamedTuples anidadas a dicts planos y agrega un
    timestamp ISO 8601 para trazabilidad del momento de la inferencia.

    Args:
        result: AudioPrediction a serializar.
        timestamp: String ISO 8601 de la hora de inferencia.

    Returns:
        dict con tipos primitivos (str, float, int, dict, list), listo
        para json.dumps().
    """
    return {
        "timestamp": timestamp,
        "audio_path": result.audio_path,
        "duration_sec": result.duration_sec,
        "num_clips": result.num_clips,
        "model_name": result.model_name,
        "threshold_used": result.threshold_used,
        "aggregated_species": result.aggregated_species,
        "aggregated_confidence": result.aggregated_confidence,
        "clip_predictions": [
            {
                "clip_index": cp.clip_index,
                "start_sec": cp.start_sec,
                "end_sec": cp.end_sec,
                "species": cp.species,
                "confidence": cp.confidence,
                "all_probs": cp.all_probs,
            }
            for cp in result.clip_predictions
        ],
    }


def _print_prediction(result: AudioPrediction) -> None:
    """Imprime un resumen legible del AudioPrediction a stdout."""
    audio_name = Path(result.audio_path).name
    sep = "=" * 55

    print(sep)
    print("SelvaSonic -- Prediccion")
    print(sep)
    print(f"Audio:     {audio_name}")
    print(f"Duracion:  {result.duration_sec:.1f} s")
    print(f"Modelo:    {result.model_name}")
    print(f"Threshold: {result.threshold_used}")
    print()
    print(f"Clips analizados: {result.num_clips}")
    print()

    # Ancho fijo para alinear columnas: etiqueta del clip + especie
    max_sp_len = max(
        (len(cp.species) for cp in result.clip_predictions),
        default=len(_NO_ID_LABEL),
    )
    for cp in result.clip_predictions:
        clip_tag = f"[clip {cp.clip_index + 1}: {cp.start_sec:.1f}-{cp.end_sec:.1f}s]"
        sp_padded = cp.species.ljust(max_sp_len)
        print(f"  {clip_tag:<30}  {sp_padded}  (conf {cp.confidence:.2f})")

    print()
    print(
        f"PREDICCION AGREGADA: {result.aggregated_species} "
        f"(confianza media {result.aggregated_confidence:.2f})"
    )
    print(sep)


# =============================================================================
# Función interna de predicción (reutilizada por predict_audio y predict_batch)
# =============================================================================

def _predict_single_audio(
    audio_path: str | Path,
    *,
    model: torch.nn.Module,
    label_map: dict[str, int],
    model_name: str,
    device: torch.device,
    confidence_threshold: float,
) -> AudioPrediction:
    """Inferencia sobre un audio dado un modelo ya cargado.

    Separa la carga del modelo de la predicción para que predict_batch
    pueda reutilizar el mismo modelo para N audios sin recargarlo.

    Args:
        audio_path: Ruta al archivo de audio.
        model: nn.Module en modo eval.
        label_map: dict {nombre: label_int} del checkpoint.
        model_name: "SelvaSonicCNN" o "SelvaSonicCNNAttention".
        device: Device del modelo.
        confidence_threshold: Umbral mínimo de confianza.

    Returns:
        AudioPrediction completo.

    Raises:
        InferenceError: Si el audio no se puede cargar o no genera clips.
    """
    audio_path = Path(audio_path)
    idx_to_name: dict[int, str] = {v: k for k, v in label_map.items()}

    # 1. Cargar audio completo desde disco
    try:
        audio = load_audio(audio_path)
    except AudioLoadError as e:
        raise InferenceError(
            f"No se pudo cargar {audio_path.name}: {e}"
        ) from e

    # 2. Segmentar en clips sin solapamiento (ver decisión de diseño en módulo)
    clips = segment_audio(
        audio.waveform,
        sample_rate=audio.sample_rate,
        source_file=str(audio_path),
        overlap_ratio=0.0,
    )

    if not clips:
        raise InferenceError(
            f"El audio {audio_path.name} no generó ningún clip. "
            "Verifica que tenga duración > 0."
        )

    # 3. Forward pass por clip
    clip_preds: list[ClipPrediction] = []
    for clip in clips:
        species_raw, conf, all_probs = _predict_clip(
            clip.samples,
            model=model,
            idx_to_name=idx_to_name,
            device=device,
            sample_rate=audio.sample_rate,
        )
        # Aplicar umbral de confianza
        species_final = species_raw if conf >= confidence_threshold else _NO_ID_LABEL

        clip_preds.append(ClipPrediction(
            clip_index=clip.clip_index,
            start_sec=clip.start_time,
            end_sec=clip.end_time,
            species=species_final,
            confidence=conf,
            all_probs=all_probs,
        ))

    # 4. Predicción agregada (moda de los clips identificados)
    agg_species, agg_conf = _aggregate_clips(clip_preds)

    return AudioPrediction(
        audio_path=str(audio_path.resolve()),
        duration_sec=audio.duration_s,
        num_clips=len(clips),
        clip_predictions=clip_preds,
        aggregated_species=agg_species,
        aggregated_confidence=agg_conf,
        model_name=model_name,
        threshold_used=confidence_threshold,
    )


# =============================================================================
# API pública
# =============================================================================

def predict_audio(
    audio_path: str | Path,
    *,
    model_path: str | Path,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    device: str = "auto",
) -> AudioPrediction:
    """Clasifica las vocalizaciones de un archivo de audio.

    Punto de entrada principal. Carga el modelo, segmenta el audio en
    clips de 5s, corre inferencia, aplica threshold y agrega por moda.

    Args:
        audio_path: Ruta al archivo de audio (.mp3, .wav, .flac, .ogg).
        model_path: Ruta al checkpoint .pth generado por train.py.
        confidence_threshold: Confianza mínima para aceptar una predicción.
            Default: 0.6.
        device: "auto" (detecta CUDA), "cuda" o "cpu".

    Returns:
        AudioPrediction con predicciones por clip y predicción agregada.

    Raises:
        InferenceError: Si el audio o el checkpoint fallan.

    Example:
        >>> result = predict_audio(
        ...     "data/raw/Ramphastos_tucanus/XC694038.mp3",
        ...     model_path="results/runs/attention_S4_v1_20260601/best.pth",
        ... )
        >>> result.aggregated_species
        'Ramphastos_tucanus'
        >>> len(result.clip_predictions)
        5
    """
    dev = _resolve_device(device)
    model, label_map, model_name = _load_model_and_map(model_path, device=dev)
    return _predict_single_audio(
        audio_path,
        model=model,
        label_map=label_map,
        model_name=model_name,
        device=dev,
        confidence_threshold=confidence_threshold,
    )


def predict_batch(
    audio_paths: list[str | Path],
    *,
    model_path: str | Path,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    device: str = "auto",
) -> list[AudioPrediction | None]:
    """Clasifica múltiples audios reutilizando un único modelo cargado.

    La clave de esta función es cargar el modelo UNA SOLA VEZ. Cargar
    un checkpoint implica leer pesos desde disco, instanciar la arquitectura
    y moverla al device — puede tardar 1-5 segundos. Con N=100 audios
    hacerlo N veces sería inadmisible.

    Los errores por audio individual se capturan (no abortan el batch):
    la posición correspondiente en la lista de retorno contendrá None.

    Args:
        audio_paths: Lista de rutas a archivos de audio.
        model_path: Ruta al checkpoint .pth.
        confidence_threshold: Umbral de confianza.
        device: "auto", "cuda" o "cpu".

    Returns:
        Lista de AudioPrediction (o None para audios que fallaron),
        en el mismo orden que audio_paths.

    Raises:
        InferenceError: Solo si el modelo no se puede cargar (aborta todo
            el batch — no tiene sentido continuar sin modelo).

    Example:
        >>> paths = sorted(Path("data/raw/Celeus_grammicus").glob("*.mp3"))
        >>> results = predict_batch(paths, model_path="best.pth")
        >>> valid = [r for r in results if r is not None]
        >>> print(f"{len(valid)}/{len(paths)} audios procesados")
    """
    dev = _resolve_device(device)

    # Carga única — la razón de existir de esta función
    model, label_map, model_name = _load_model_and_map(model_path, device=dev)

    results: list[AudioPrediction | None] = []
    n = len(audio_paths)
    for i, path in enumerate(audio_paths, start=1):
        print(f"  [{i}/{n}] {Path(path).name} ...", end=" ", flush=True)
        try:
            result = _predict_single_audio(
                path,
                model=model,
                label_map=label_map,
                model_name=model_name,
                device=dev,
                confidence_threshold=confidence_threshold,
            )
            print(f"-> {result.aggregated_species} (conf {result.aggregated_confidence:.2f})")
            results.append(result)
        except InferenceError as e:
            print(f"-> ERROR: {e}")
            results.append(None)

    return results


# =============================================================================
# CLI y smoke test
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos de la CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m src.inference",
        description=(
            "SelvaSonic -- Clasificacion bioacustica de aves amazonicas.\n"
            "Recibe un audio o carpeta de audios y predice la especie."
        ),
    )

    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--audio",
        metavar="ARCHIVO",
        help="Ruta a un archivo de audio (.mp3, .wav, .flac, .ogg)",
    )
    src_group.add_argument(
        "--batch",
        metavar="CARPETA",
        help="Carpeta con archivos de audio (procesa todos los soportados)",
    )

    parser.add_argument(
        "--model",
        required=True,
        metavar="CHECKPOINT",
        help="Ruta al checkpoint .pth generado por train.py (OBLIGATORIO)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        metavar="T",
        help=(
            f"Confianza minima para aceptar prediccion (default: "
            f"{DEFAULT_CONFIDENCE_THRESHOLD}). Clips por debajo "
            "se reportan como 'no_identificado'."
        ),
    )
    parser.add_argument(
        "--json",
        metavar="SALIDA.json",
        dest="json_out",
        help="Guardar resultado(s) en este archivo JSON",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device para inferencia: auto detecta CUDA (default: auto)",
    )

    return parser


def _run_cli() -> None:
    """Ejecuta la inferencia a partir de argumentos de línea de comandos."""
    parser = _build_parser()
    args = parser.parse_args()
    timestamp = datetime.now(tz=timezone.utc).isoformat()

    if args.audio:
        result = predict_audio(
            args.audio,
            model_path=args.model,
            confidence_threshold=args.threshold,
            device=args.device,
        )
        _print_prediction(result)

        if args.json_out:
            out_path = Path(args.json_out)
            out_path.write_text(
                json.dumps(
                    _prediction_to_dict(result, timestamp=timestamp),
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(f"\nResultado JSON guardado en: {out_path}")

    else:  # --batch
        folder = Path(args.batch)
        if not folder.is_dir():
            parser.error(f"'{args.batch}' no es una carpeta valida.")

        audio_files = sorted(
            p
            for ext in SUPPORTED_EXTENSIONS
            for p in folder.glob(f"*{ext}")
        )
        if not audio_files:
            print(f"No se encontraron audios en {folder}")
            raise SystemExit(0)

        print(f"Procesando {len(audio_files)} archivo(s) en {folder.name}/...")
        results = predict_batch(
            audio_files,
            model_path=args.model,
            confidence_threshold=args.threshold,
            device=args.device,
        )

        # Resumen de batch
        print()
        print("=" * 55)
        print(f"Resumen del batch ({len(results)} archivos)")
        print("=" * 55)
        for path, res in zip(audio_files, results):
            if res is None:
                status = "ERROR"
            else:
                status = f"{res.aggregated_species:<35} (conf {res.aggregated_confidence:.2f})"
            print(f"  {path.name:<40}  {status}")

        if args.json_out:
            valid = [r for r in results if r is not None]
            out_path = Path(args.json_out)
            out_path.write_text(
                json.dumps(
                    {
                        "timestamp": timestamp,
                        "batch": [
                            _prediction_to_dict(r, timestamp=timestamp)
                            for r in valid
                        ],
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(f"\nResultados JSON guardados en: {out_path}")


def _run_smoke_test() -> None:
    """Smoke test del pipeline completo con audio sintético.

    Busca el primer best.pth disponible en results/runs/. Si no hay
    ninguno, imprime un aviso y termina sin fallo. Si hay uno, genera
    un audio sintético (seno + ruido), corre predict_audio y verifica
    que el resultado tiene la estructura correcta.
    """
    import shutil
    import tempfile

    import soundfile as sf

    print("=" * 55)
    print(" SelvaSonic Inference -- Smoke Test ")
    print("=" * 55)

    # 1. Buscar checkpoint disponible
    runs_dir = Path(__file__).parent.parent / "results" / "runs"
    model_paths = sorted(runs_dir.glob("*/best.pth"))
    if not model_paths:
        print(f"\n[SKIP] No se encontro ningun best.pth en {runs_dir}")
        print("       Entrena un modelo primero para ejecutar este test.")
        print("       (Los modulos de audio, segmentation y transforms")
        print("        son correctos independientemente del test.)")
        return

    model_path = model_paths[0]
    print(f"\n  Checkpoint: {model_path.parent.name}")
    print(f"             -> {model_path}")

    # 2. Audio sintético: seno 440 Hz + ruido Gaussiano, 15s → 3 clips
    tmpdir = Path(tempfile.mkdtemp(prefix="selvasonic_infer_smoke_"))
    try:
        rng = np.random.default_rng(42)
        t = np.arange(15 * SAMPLE_RATE) / SAMPLE_RATE
        waveform = (
            0.5 * np.sin(2 * np.pi * 440.0 * t)
            + 0.05 * rng.standard_normal(len(t))
        ).astype(np.float32)
        audio_path = tmpdir / "smoke_audio.wav"
        sf.write(str(audio_path), waveform, SAMPLE_RATE)
        duration_s = len(waveform) / SAMPLE_RATE
        print(f"\n  Audio sintetico: {audio_path.name} ({duration_s:.1f} s)")

        # 3. Correr inferencia
        result = predict_audio(
            str(audio_path),
            model_path=str(model_path),
            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        )

        # 4. Verificar estructura del resultado
        assert isinstance(result, AudioPrediction), \
            "predict_audio debe devolver AudioPrediction"
        assert isinstance(result.clip_predictions, list), \
            "clip_predictions debe ser lista"
        assert len(result.clip_predictions) > 0, \
            "debe haber al menos un ClipPrediction"
        assert all(isinstance(cp, ClipPrediction) for cp in result.clip_predictions), \
            "cada elemento debe ser ClipPrediction"
        assert isinstance(result.aggregated_species, str), \
            "aggregated_species debe ser str"
        assert 0.0 <= result.aggregated_confidence <= 1.0, \
            f"aggregated_confidence fuera de [0,1]: {result.aggregated_confidence}"
        assert result.model_name in ("SelvaSonicCNN", "SelvaSonicCNNAttention"), \
            f"model_name inesperado: {result.model_name}"
        assert result.threshold_used == DEFAULT_CONFIDENCE_THRESHOLD, \
            "threshold_used no coincide con el valor pasado"

        # all_probs debe sumar 1 por clip
        for cp in result.clip_predictions:
            total = sum(cp.all_probs.values())
            assert abs(total - 1.0) < 1e-4, \
                f"all_probs no suma 1 en clip {cp.clip_index}: total={total:.6f}"

        # Verificar serialización JSON
        ts = datetime.now(tz=timezone.utc).isoformat()
        d = _prediction_to_dict(result, timestamp=ts)
        json_str = json.dumps(d)
        assert json_str, "la serialización JSON falló"

        # 5. Mostrar resultado y métricas del test
        print()
        _print_prediction(result)

        print()
        print(f"  num_clips verificado:       {result.num_clips}")
        print(f"  all_probs suman 1 por clip: OK")
        print(f"  JSON serializable:          OK")
        print()
        print("=" * 55)
        print(" Smoke test PASADO ")
        print("=" * 55)

    finally:
        shutil.rmtree(tmpdir)


# =============================================================================
# Punto de entrada
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        # Sin argumentos CLI → smoke test automático
        _run_smoke_test()
    else:
        _run_cli()
