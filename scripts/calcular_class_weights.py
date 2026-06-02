"""
scripts/calcular_class_weights.py
==================================

Computa los class weights del split TRAIN de SelvaSonic sin instanciar
el Dataset completo (evita el cuello de botella de I/O de build_index).

Estrategia:
    1. Parsea duraciones desde data/metadata.csv (sin cargar audio).
    2. Para ESC-50: asume 1 clip por archivo (todos son clips de 5s).
    3. Replica el split estratificado por archivo de dataset.py con la
       misma semilla, garantizando consistencia exacta con el Dataset real.
    4. Llama a compute_class_weights() del modulo ya existente.
    5. Imprime tabla, genera dos PNGs en results/, guarda tensor en data/.

Uso:
    python scripts/calcular_class_weights.py
    python scripts/calcular_class_weights.py --verify-against-dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# -- Ajustar sys.path para importar desde src/ sin instalar el paquete --
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.class_weights import ClassWeightsResult, compute_class_weights, verify_class_weights
from src.config import (
    CLIP_NUM_SAMPLES,
    CLIP_OVERLAP_RATIO,
    MIN_LEFTOVER_SEC,
    NUM_CLASSES,
    SAMPLE_RATE,
    SHORT_AUDIO_STRATEGY,
    LEFTOVER_STRATEGY,
    SUPPORTED_EXTENSIONS,
)

# =============================================================================
# Rutas y constantes del split (deben coincidir exactamente con dataset.py)
# =============================================================================

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
METADATA_CSV = DATA_DIR / "metadata.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_PT = DATA_DIR / "class_weights.pt"

NO_AVE_DIRNAME = "ESC-50"
NO_AVE_LABEL = 0
NO_AVE_LABEL_NAME = "no_ave"

# Estos valores deben ser identicos a los defaults de stratified_split_by_file
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
RANDOM_STATE = 42


# =============================================================================
# Estructuras de datos
# =============================================================================

class FileRecord(NamedTuple):
    """Metadata minima de un archivo: sin audio, solo lo que necesita el split."""
    file_id: str
    label: int
    label_name: str
    n_clips: int


# =============================================================================
# Funciones de soporte
# =============================================================================

def _list_audio_files(folder: Path) -> list[Path]:
    """Lista archivos de audio soportados en una carpeta (replica dataset.py)."""
    files: list[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(folder.glob(f"*{ext}"))
        files.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(files)


def _parse_duration_mmss(length_str: str) -> float:
    """
    Convierte duracion 'M:SS' o 'H:MM:SS' a segundos.

    Xeno-canto usa formato M:SS (ej. '2:38' = 158s).
    """
    parts = str(length_str).strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    raise ValueError(f"Formato de duracion no reconocido: {length_str!r}")


def _compute_n_clips_analytical(duration_sec: float) -> int:
    """
    Replica la logica de _compute_n_clips + leftover de segmentation.py,
    sin cargar el audio. Usa las mismas constantes de config.py.

    El resultado coincide con el numero real de clips que build_index()
    produciria para este archivo.
    """
    clip_len = CLIP_NUM_SAMPLES                              # 110250 muestras
    hop_len = int(clip_len * (1.0 - CLIP_OVERLAP_RATIO))    # 55125 (50% overlap)
    min_leftover = int(MIN_LEFTOVER_SEC * SAMPLE_RATE)       # 44100 muestras

    # Numero de muestras a 22050 Hz despues de resampleo
    audio_len = int(duration_sec * SAMPLE_RATE)

    # Audio mas corto que un clip
    if audio_len < clip_len:
        return 0 if SHORT_AUDIO_STRATEGY == "drop" else 1

    # Clips completos por ventana deslizante: N = floor((L - T_c) / h) + 1
    n_clips = (audio_len - clip_len) // hop_len + 1

    # Sobrante al final
    last_processed = (n_clips - 1) * hop_len + clip_len
    leftover = audio_len - last_processed
    if leftover >= min_leftover and LEFTOVER_STRATEGY != "drop":
        n_clips += 1

    return n_clips


def _build_metadata_lookup() -> dict[str, float]:
    """
    Construye {xc_id_str: duration_sec} desde data/metadata.csv.

    La clave es el ID numerico como string (ej. "127279"), que coincide
    con el nombre del archivo local XC127279.mp3 tras quitar "XC".
    """
    df = pd.read_csv(METADATA_CSV)
    lookup: dict[str, float] = {}
    for _, row in df.iterrows():
        xc_id = str(int(row["id"]))
        try:
            lookup[xc_id] = _parse_duration_mmss(row["length"])
        except (ValueError, KeyError):
            pass
    return lookup


def _get_duration_from_header(path: Path) -> float:
    """
    Fallback: obtiene duracion desde el header del archivo sin decodificar.
    Usado cuando el archivo no aparece en metadata.csv (caso raro).
    """
    try:
        import soundfile as sf
        return sf.info(str(path)).duration
    except Exception:
        pass
    try:
        import torchaudio
        info = torchaudio.info(str(path))
        return info.num_frames / info.sample_rate
    except Exception:
        pass
    raise RuntimeError(
        f"No se pudo obtener duracion de {path.name} sin decodificar. "
        f"Agrega el archivo a metadata.csv o instala torchaudio."
    )


# =============================================================================
# Construccion del indice analitico (sin I/O de audio)
# =============================================================================

def build_file_records() -> tuple[list[FileRecord], dict[str, int]]:
    """
    Construye la lista de FileRecords y el label_map sin cargar ningun audio.

    Replica la logica de build_index() de dataset.py:
    - Misma deteccion de subdirectorios
    - Mismo orden alfabetico de especies (labels 1..N)
    - Mismo tratamiento de ESC-50 como clase no_ave (label 0)

    Para xeno-canto: duracion desde metadata.csv -> n_clips analitico.
    Para ESC-50: 1 clip por archivo (todos son clips nativos de 5s).
    """
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"No existe: {RAW_DIR}")
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"No existe: {METADATA_CSV}")

    meta_lookup = _build_metadata_lookup()

    # Detectar subcarpetas (misma logica que build_index)
    subdirs = [p for p in RAW_DIR.iterdir() if p.is_dir()]
    species_dirs = sorted(
        [d for d in subdirs if d.name != NO_AVE_DIRNAME],
        key=lambda d: d.name,
    )

    # Construir label_map: no_ave=0, especies=1..N en orden alfabetico
    label_map: dict[str, int] = {NO_AVE_LABEL_NAME: NO_AVE_LABEL}
    for i, d in enumerate(species_dirs, start=1):
        label_map[d.name] = i

    records: list[FileRecord] = []
    missing_from_meta: list[str] = []

    # ---- Clases positivas (xeno-canto) ----
    for species_dir in species_dirs:
        species_name = species_dir.name
        species_label = label_map[species_name]
        audio_files = _list_audio_files(species_dir)

        for audio_path in audio_files:
            file_id = audio_path.stem          # "XC127279"
            xc_id = file_id.removeprefix("XC") # "127279"

            if xc_id in meta_lookup:
                duration_sec = meta_lookup[xc_id]
            else:
                # Archivo no encontrado en metadata.csv (caso infrecuente)
                try:
                    duration_sec = _get_duration_from_header(audio_path)
                    missing_from_meta.append(file_id)
                except RuntimeError as e:
                    print(f"  AVISO: saltando {file_id}: {e}")
                    continue

            n = _compute_n_clips_analytical(duration_sec)
            if n > 0:
                records.append(FileRecord(
                    file_id=file_id,
                    label=species_label,
                    label_name=species_name,
                    n_clips=n,
                ))

    if missing_from_meta:
        print(f"\n  AVISO: {len(missing_from_meta)} archivos no encontrados en "
              f"metadata.csv (duracion leida desde header): "
              f"{missing_from_meta[:5]}{'...' if len(missing_from_meta) > 5 else ''}")

    # ---- Clase negativa: ESC-50 (todos son clips nativos de 5s) ----
    no_ave_dir = RAW_DIR / NO_AVE_DIRNAME
    if no_ave_dir.exists():
        for sub in sorted(no_ave_dir.iterdir()):
            if sub.is_dir():
                for audio_path in _list_audio_files(sub):
                    records.append(FileRecord(
                        file_id=audio_path.stem,
                        label=NO_AVE_LABEL,
                        label_name=NO_AVE_LABEL_NAME,
                        n_clips=1,  # ESC-50: clips nativos de 5s
                    ))

    return records, label_map


# =============================================================================
# Replicacion del split estratificado (identico a stratified_split_by_file)
# =============================================================================

def get_train_file_ids(
    records: list[FileRecord],
    *,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    random_state: int = RANDOM_STATE,
) -> set[str]:
    """
    Replica stratified_split_by_file() de dataset.py y devuelve solo el
    conjunto de file_ids asignados al split TRAIN.

    Usa el mismo algoritmo y semilla para garantizar consistencia exacta.
    """
    rng = np.random.default_rng(random_state)

    files_by_class: dict[int, set[str]] = {}
    for rec in records:
        files_by_class.setdefault(rec.label, set()).add(rec.file_id)

    train_file_ids: set[str] = set()

    for label, file_ids in files_by_class.items():
        files = sorted(file_ids)  # orden deterministico (idem a dataset.py)
        n = len(files)
        if n < 3:
            raise ValueError(
                f"Clase {label} tiene solo {n} archivo(s). Minimo requerido: 3."
            )

        idx_perm = rng.permutation(n)
        files_shuffled = [files[i] for i in idx_perm]

        # Misma logica de redondeo que dataset.py
        n_train = int(np.floor(n * train_ratio))
        n_val = int(np.floor(n * val_ratio))
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1

        train_file_ids.update(files_shuffled[:n_train])

    return train_file_ids


# =============================================================================
# Tabla de resultados
# =============================================================================

def print_results_table(
    result: ClassWeightsResult,
    label_map: dict[str, int],
    file_counts: dict[int, int],
) -> None:
    """Imprime tabla bonita de class weights en consola."""
    # Invertir label_map para lookup por label
    label_to_name = {v: k for k, v in label_map.items()}

    header = (
        f"  {'Lbl':>4} | {'Clase':<30} | {'Archivos':>8} | "
        f"{'Clips':>7} | {'Freq%':>6} | {'Peso':>8}"
    )
    sep = "  " + "-" * 4 + "-+-" + "-" * 30 + "-+-" + "-" * 8 + "-+-" + \
          "-" * 7 + "-+-" + "-" * 6 + "-+-" + "-" * 8

    print("\n" + "=" * 75)
    print("  CLASS WEIGHTS - SPLIT TRAIN")
    print("=" * 75)
    print(header)
    print(sep)

    # Ordenar por numero de clips descendente
    order = sorted(range(NUM_CLASSES), key=lambda c: -result.class_counts[c])
    for cls in order:
        name = label_to_name.get(cls, f"clase_{cls}")
        n_files = file_counts.get(cls, 0)
        n_clips = int(result.class_counts[cls])
        freq = result.class_frequencies[cls] * 100
        weight = result.weights[cls].item()
        print(
            f"  {cls:>4d} | {name:<30} | {n_files:>8d} | "
            f"{n_clips:>7d} | {freq:>5.2f}% | {weight:>8.4f}"
        )

    print(sep)
    total_clips = int(result.class_counts.sum())
    print(f"  {'TOTAL':<37} | {'':>8} | {total_clips:>7d} | {'100%':>6} | {'':>8}")
    print(f"\n  weighted_sum = {result.weighted_sum:.10f}  (debe ser ~1.0)")
    print("=" * 75)


# =============================================================================
# Visualizaciones
# =============================================================================

def generate_plots(
    result: ClassWeightsResult,
    label_map: dict[str, int],
) -> None:
    """
    Genera dos PNGs en results/:
    - class_weights_distribution.png: barras de muestras y pesos lado a lado
    - class_weights_ratios.png: barras horizontales con ratio respecto a no_ave
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    label_to_name = {v: k for k, v in label_map.items()}
    labels = list(range(NUM_CLASSES))
    short_names = [label_to_name.get(i, f"cls_{i}") for i in labels]
    # Nombres cortos para el eje X
    display_names = [n.replace("_", "\n") if len(n) > 12 else n for n in short_names]

    counts = result.class_counts.astype(float)
    weights = result.weights.numpy()

    # ---- Figura 1: distribucion de muestras vs pesos ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    colors_counts = plt.cm.Blues(np.linspace(0.4, 0.9, NUM_CLASSES))
    colors_weights = plt.cm.Oranges(np.linspace(0.4, 0.9, NUM_CLASSES))

    bars1 = ax1.bar(labels, counts, color=colors_counts, edgecolor="white", linewidth=0.5)
    ax1.set_title("Distribucion de clips por clase (train)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Clase")
    ax1.set_ylabel("Numero de clips")
    ax1.set_xticks(labels)
    ax1.set_xticklabels(display_names, fontsize=7)
    ax1.grid(axis="y", alpha=0.3)
    for bar, cnt in zip(bars1, counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            f"{int(cnt):,}",
            ha="center", va="bottom", fontsize=6.5,
        )

    bars2 = ax2.bar(labels, weights, color=colors_weights, edgecolor="white", linewidth=0.5)
    ax2.set_title("Class weights por clase", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Clase")
    ax2.set_ylabel("Peso")
    ax2.set_xticks(labels)
    ax2.set_xticklabels(display_names, fontsize=7)
    ax2.grid(axis="y", alpha=0.3)
    for bar, w in zip(bars2, weights):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(weights) * 0.01,
            f"{w:.3f}",
            ha="center", va="bottom", fontsize=6.5,
        )

    fig.suptitle("SelvaSonic - Analisis de class weights", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out1 = RESULTS_DIR / "class_weights_distribution.png"
    fig.savefig(out1, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {out1}")

    # ---- Figura 2: ratio respecto a no_ave ----
    no_ave_weight = weights[NO_AVE_LABEL]
    ratios = weights / no_ave_weight

    order_by_ratio = np.argsort(ratios)[::-1]
    sorted_names = [short_names[i].replace("_", " ") for i in order_by_ratio]
    sorted_ratios = ratios[order_by_ratio]

    fig2, ax = plt.subplots(figsize=(16, 6), dpi=150)
    colors_ratio = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, NUM_CLASSES))

    bars = ax.barh(range(NUM_CLASSES), sorted_ratios, color=colors_ratio, edgecolor="white")
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1.2, label="Ratio = 1 (no_ave)")
    ax.set_xlabel("Ratio de peso (clase / no_ave)")
    ax.set_title(
        "Ratio de class weight relativo a no_ave\n"
        "(>1 = clase sub-representada, recibe mas enfasis en la loss)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=9)
    for bar, ratio in zip(bars, sorted_ratios):
        ax.text(
            bar.get_width() + max(sorted_ratios) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"x{ratio:.2f}",
            va="center", fontsize=8,
        )
    fig2.suptitle("SelvaSonic - Desbalance relativo por clase", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out2 = RESULTS_DIR / "class_weights_ratios.png"
    fig2.savefig(out2, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Guardado: {out2}")


# =============================================================================
# Guardado del tensor con metadata
# =============================================================================

def save_weights_tensor(
    result: ClassWeightsResult,
    label_map: dict[str, int],
) -> None:
    """Guarda el tensor de pesos con metadata completa en data/class_weights.pt."""
    payload = {
        "weights": result.weights,
        "class_counts": torch.from_numpy(result.class_counts),
        "class_frequencies": torch.from_numpy(result.class_frequencies),
        "weighted_sum_check": result.weighted_sum,
        "label_map": label_map,
        "num_classes": NUM_CLASSES,
        "total_samples": int(result.class_counts.sum()),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(payload, OUTPUT_PT)
    size_kb = OUTPUT_PT.stat().st_size / 1024
    print(f"  Guardado: {OUTPUT_PT}  ({size_kb:.1f} KB)")


# =============================================================================
# Smoke test de consistencia contra el Dataset real (flag opcional)
# =============================================================================

def verify_against_real_dataset(
    analytical_counts: np.ndarray,
    label_map: dict[str, int],
) -> None:
    """
    Instancia el SelvaSonicDataset real y compara los conteos de labels.
    LENTO (~10+ minutos por el build_index). Solo usar para validacion one-shot.
    """
    print("\n[VERIFY] Instanciando dataset real (esto puede tardar mucho)...")
    from src.dataset import build_index, stratified_split_by_file

    records_real, label_map_real = build_index(RAW_DIR, verbose=False)
    train_real, _, _ = stratified_split_by_file(records_real, random_state=RANDOM_STATE)

    real_labels = [r.label for r in train_real]
    real_counts = np.bincount(real_labels, minlength=NUM_CLASSES)

    label_to_name = {v: k for k, v in label_map.items()}
    print(f"\n  {'Clase':<30} | {'Analitico':>10} | {'Real':>10} | {'Diff':>6}")
    print("  " + "-" * 65)
    all_match = True
    for cls in range(NUM_CLASSES):
        name = label_to_name.get(cls, f"cls_{cls}")
        a = int(analytical_counts[cls])
        r = int(real_counts[cls])
        diff = a - r
        match = "OK" if diff == 0 else f"DIFF:{diff:+d}"
        if diff != 0:
            all_match = False
        print(f"  {name:<30} | {a:>10,} | {r:>10,} | {match:>6}")

    print()
    if all_match:
        print("  [OK] Los conteos analiticos coinciden EXACTAMENTE con el dataset real.")
    else:
        print("  [AVISO] Hay discrepancias. Posibles causas:")
        print("    - Archivos en disco no encontrados en metadata.csv")
        print("    - Archivos corruptos que build_index descarto con AudioLoadError")
        print("    - Diferencia en redondeo de duracion al resamplear")


# =============================================================================
# Main
# =============================================================================

def main(*, verify_against_dataset: bool = False) -> None:
    print("=" * 75)
    print("  CALCULAR CLASS WEIGHTS - SelvaSonic ML")
    print("=" * 75)

    # ---- 1. Construir indice analitico ----
    print("\n[1/5] Construyendo indice analitico desde metadata.csv...")
    records, label_map = build_file_records()
    print(f"  Total archivos indexados: {len(records)}")
    print(f"  Clases detectadas: {len(label_map)}")

    # ---- 2. Split estratificado (replicando dataset.py) ----
    print("\n[2/5] Aplicando split estratificado por archivo...")
    train_ids = get_train_file_ids(records)
    train_records = [r for r in records if r.file_id in train_ids]
    print(f"  Archivos en TRAIN: {len(train_ids)}")
    print(f"  Clips en TRAIN:    {sum(r.n_clips for r in train_records):,}")

    # ---- 3. Construir lista de labels del train split ----
    print("\n[3/5] Construyendo lista de labels del train set...")
    train_labels: list[int] = []
    for rec in train_records:
        train_labels.extend([rec.label] * rec.n_clips)
    print(f"  Labels totales en TRAIN: {len(train_labels):,}")

    # Conteo de archivos por clase (para la tabla)
    file_counts: dict[int, int] = {}
    for rec in train_records:
        file_counts[rec.label] = file_counts.get(rec.label, 0) + 1

    # ---- 4. Calcular class weights ----
    print("\n[4/5] Calculando class weights...")
    result = compute_class_weights(labels=train_labels, num_classes=NUM_CLASSES)

    # ---- 5. Tabla de resultados ----
    print_results_table(result, label_map, file_counts)

    # ---- 6. Verificacion matematica ----
    ok = verify_class_weights(result=result)
    print(f"\n  verify_class_weights: {'[OK]' if ok else '[FAIL]'}")
    assert ok, "La verificacion matematica fallo. Revisar compute_class_weights()."

    # ---- 7. Guardar tensor ----
    print("\n[5/5] Guardando resultados...")
    save_weights_tensor(result, label_map)

    # ---- 8. Generar visualizaciones ----
    generate_plots(result, label_map)

    # ---- 9. Verificacion final desde disco ----
    print("\n[Final] Verificando tensor guardado desde disco...")
    loaded = torch.load(OUTPUT_PT, weights_only=False)
    reloaded_result = ClassWeightsResult(
        weights=loaded["weights"],
        class_counts=loaded["class_counts"].numpy(),
        class_frequencies=loaded["class_frequencies"].numpy(),
        weighted_sum=float(loaded["weighted_sum_check"]),
    )
    ok_reloaded = verify_class_weights(result=reloaded_result)
    print(f"  verify_class_weights desde disco: {'[OK]' if ok_reloaded else '[FAIL]'}")
    assert ok_reloaded, "El tensor guardado no paso la verificacion."

    print("\n" + "=" * 75)
    print("  COMPLETADO. class_weights listos para entrenamiento.")
    print("=" * 75)

    # ---- 10. Verificacion contra dataset real (opcional) ----
    if verify_against_dataset:
        verify_against_real_dataset(result.class_counts, label_map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calcula class weights de SelvaSonic sin instanciar el Dataset completo."
    )
    parser.add_argument(
        "--verify-against-dataset",
        action="store_true",
        help="Instancia el SelvaSonicDataset real y compara conteos (LENTO: ~10+ min).",
    )
    args = parser.parse_args()
    main(verify_against_dataset=args.verify_against_dataset)
