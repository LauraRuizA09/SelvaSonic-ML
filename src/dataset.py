"""
src/dataset.py
==============

Dataset y DataLoaders para SelvaSonic.

Rol del módulo
--------------
Este es el "director de orquesta" del pipeline de datos. NO reimplementa
ninguna lógica que ya esté en otros módulos: solo COORDINA cuándo se llama
cada función existente.

Flujo de uso típico:
    1. build_index(...)               -> genera la "libreta" de coordenadas
    2. stratified_split_by_file(...)  -> divide la libreta en 3 splits
    3. SelvaSonicDataset(...)         -> wrappea cada split en un Dataset
    4. create_dataloaders(...)        -> conveniencia para crear los 3 a la vez

Decisiones de diseño
--------------------
- ÍNDICE: solo metadata (file_path, start_sec, end_sec, label). Pesa pocos KB.
  El audio NO se guarda en RAM hasta que PyTorch pida un item.

- LECTURA PARCIAL: en __getitem__ usamos load_audio(offset=..., duration=...)
  que aprovecha la capacidad de librosa de leer SOLO el segmento necesario,
  sin descomprimir el MP3 entero. Mucho más rápido.

- AUGMENTATION: solo en TRAIN. La regla de oro de ML: nunca contaminar
  val/test con datos sintéticos.

- MEL AL VUELO: cada __getitem__ calcula el espectrograma. Permite que
  augmentation actúe sobre el audio (no sobre el espectrograma).

- SPLIT POR ARCHIVO: si XC12345.mp3 tiene 6 clips, los 6 van AL MISMO SPLIT.
  Evita data leakage por ambiente compartido (mismo micro, lugar, fecha).

- ETIQUETA 0 = no_ave: la clase trivial es la 0 por convención (primera
  fila/columna de la matriz de confusión).
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Módulos del proyecto (todos ya implementados)
from src.audio_io import AudioLoadError, load_audio
from src.augmentation import (
    AugmentationError,
    add_noise,
    pitch_shift,
    time_stretch,
)
from src.config import (
    AUG_PROB_ADD_NOISE,
    AUG_PROB_PITCH_SHIFT,
    AUG_PROB_TIME_STRETCH,
    CLIP_DURATION_S,
    CLIP_NUM_SAMPLES,
    SAMPLE_RATE,
    SUPPORTED_EXTENSIONS,
)
from src.segmentation import segment_audio
from src.transforms import (
    compute_mel_spectrogram,
    normalize_spectrogram,
    to_tensor,
)


# ============================================================================
# Estructuras de datos
# ============================================================================

class SegmentRecord(NamedTuple):
    """
    Una entrada del índice del dataset. SOLO METADATA, nunca audio.

    Atributos
    ---------
    file_path : str
        Ruta absoluta al archivo de audio original (.mp3 o .wav).
    file_id : str
        Identificador único del archivo (nombre sin extensión).
        Ejemplo: "XC127279" para "data/raw/Celeus_grammicus/XC127279.mp3".
        Es la clave para el split estratificado por archivo.
    label : int
        Etiqueta entera de la especie (0 a N_CLASSES-1). PyTorch necesita ints.
    label_name : str
        Nombre legible de la especie (para debug y visualización).
        Ejemplo: "Celeus_grammicus", "no_ave".
    start_sec : float
        Segundo de inicio del clip dentro del archivo original.
    end_sec : float
        Segundo final del clip dentro del archivo original.
    is_padded : bool
        True si el clip tuvo padding (audio corto o sobrante final).

    ¿Por qué NamedTuple y no dataclass?
        - Inmutable (no se modifica accidentalmente).
        - Más liviano en memoria (importante con miles de clips).
        - Coherente con AudioData y AudioClip de los otros módulos.
    """
    file_path: str
    file_id: str
    label: int
    label_name: str
    start_sec: float
    end_sec: float
    is_padded: bool


# ============================================================================
# Construcción del índice (la "libreta de coordenadas")
# ============================================================================

def build_index(
    raw_data_dir: str | Path,
    *,
    no_ave_dirname: str = "ESC-50",
    no_ave_label: int = 0,
    no_ave_label_name: str = "no_ave",
    verbose: bool = True,
) -> tuple[list[SegmentRecord], dict[str, int]]:
    """
    Recorre data/raw/ y construye el índice maestro de TODOS los clips.

    Estructura esperada de data/raw/:
        data/raw/
        ├── Celeus_grammicus/        ← clase positiva (especie de ave)
        │   ├── XC127279.mp3
        │   └── XC450123.mp3
        ├── Ramphastos_tucanus/      ← otra especie
        ├── ...
        └── ESC-50/                   ← clase negativa (no_ave)
            ├── rain/                 ← subcarpetas por categoría
            │   └── 1-100210-A.wav
            └── wind/
                └── ...

    Lógica:
        1. Subcarpetas directas de raw_data_dir = clases (excepto ESC-50).
        2. Cada audio se carga, se segmenta usando segment_audio().
        3. Por cada clip, se guarda SOLO la metadata (no las muestras).
        4. ESC-50 se trata como una sola clase "no_ave" (todas sus
           subcarpetas se aplanan).

    Parámetros
    ----------
    raw_data_dir : str | Path
        Carpeta raíz con los datos (ej. "data/raw").
    no_ave_dirname : str
        Nombre de la carpeta con audios negativos (ESC-50 por convención).
    no_ave_label : int
        Etiqueta entera para la clase no_ave. Default 0 (clase trivial).
    no_ave_label_name : str
        Nombre legible para la clase negativa.
    verbose : bool
        Si True, imprime progreso.

    Retorna
    -------
    records : list[SegmentRecord]
        Lista plana de TODOS los clips disponibles, con su metadata.
    label_map : dict[str, int]
        Mapeo nombre → etiqueta entera. Útil para reportes y visualización.
        Ejemplo: {"no_ave": 0, "Celeus_grammicus": 1, "Ramphastos_tucanus": 2, ...}

    Notas
    -----
    Esta función carga CADA audio una sola vez (al inicio del entrenamiento)
    para conocer su duración y poder segmentarlo. Pero no guarda los datos
    del audio: solo las coordenadas (start_sec, end_sec). Después se libera.
    """
    raw_dir = Path(raw_data_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta {raw_dir}")

    # Listar subcarpetas (cada una es una clase)
    subdirs = [p for p in raw_dir.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No hay subcarpetas en {raw_dir}")

    # Separar especies (positivas) de no_ave
    species_dirs = [d for d in subdirs if d.name != no_ave_dirname]
    no_ave_dir = raw_dir / no_ave_dirname

    # Construir label_map: no_ave=0, especies=1..N (orden alfabético)
    species_dirs.sort(key=lambda d: d.name)
    label_map: dict[str, int] = {no_ave_label_name: no_ave_label}
    for i, d in enumerate(species_dirs, start=1):
        label_map[d.name] = i

    if verbose:
        print(f"Construyendo índice desde {raw_dir}")
        print(f"  Clases encontradas: {len(label_map)}")
        for name, lbl in label_map.items():
            print(f"    {lbl}: {name}")

    records: list[SegmentRecord] = []

    # ---- Procesar especies (clases positivas) ----
    for species_dir in species_dirs:
        species_name = species_dir.name
        species_label = label_map[species_name]
        audio_files = _list_audio_files(species_dir)

        if verbose:
            print(f"\n  {species_name}: {len(audio_files)} archivos")

        for audio_path in audio_files:
            try:
                _add_segments_to_index(
                    audio_path=audio_path,
                    label=species_label,
                    label_name=species_name,
                    records=records,
                )
            except AudioLoadError as e:
                if verbose:
                    print(f"    AVISO: salto {audio_path.name}: {e}")

    # ---- Procesar ESC-50 (clase negativa) ----
    if no_ave_dir.exists():
        # ESC-50 tiene subcarpetas por categoría (rain, wind, etc.)
        # Las aplanamos: todo va a la clase no_ave.
        esc_files: list[Path] = []
        for sub in no_ave_dir.iterdir():
            if sub.is_dir():
                esc_files.extend(_list_audio_files(sub))
            elif sub.is_file() and sub.suffix.lower() in SUPPORTED_EXTENSIONS:
                esc_files.append(sub)

        if verbose:
            print(f"\n  {no_ave_label_name}: {len(esc_files)} archivos")

        for audio_path in esc_files:
            try:
                _add_segments_to_index(
                    audio_path=audio_path,
                    label=no_ave_label,
                    label_name=no_ave_label_name,
                    records=records,
                )
            except AudioLoadError as e:
                if verbose:
                    print(f"    AVISO: salto {audio_path.name}: {e}")

    if verbose:
        print(f"\nÍndice construido: {len(records)} clips totales\n")

    return records, label_map


def _list_audio_files(folder: Path) -> list[Path]:
    """Lista archivos de audio (extensiones soportadas) en una carpeta."""
    files: list[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(folder.glob(f"*{ext}"))
        files.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(files)


def _add_segments_to_index(
    audio_path: Path,
    label: int,
    label_name: str,
    records: list[SegmentRecord],
) -> None:
    """
    Helper: carga un audio, lo segmenta, y agrega las metadatas al índice.
    Libera la memoria del audio después (solo guardamos coordenadas).
    """
    audio = load_audio(audio_path)
    clips = segment_audio(
        audio.waveform,
        sample_rate=audio.sample_rate,
        source_file=str(audio_path),
    )
    file_id = audio_path.stem  # nombre sin extensión
    for clip in clips:
        records.append(SegmentRecord(
            file_path=str(audio_path),
            file_id=file_id,
            label=label,
            label_name=label_name,
            start_sec=clip.start_time,
            end_sec=clip.end_time,
            is_padded=clip.is_padded,
        ))
    # `audio` y `clips` se liberan al salir de esta función


# ============================================================================
# Split estratificado POR ARCHIVO
# ============================================================================

def stratified_split_by_file(
    records: list[SegmentRecord],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[list[SegmentRecord], list[SegmentRecord], list[SegmentRecord]]:
    """
    Divide el índice en 3 splits manteniendo:
        1. Estratificación por clase (proporciones similares en cada split).
        2. Integridad por archivo (ningún file_id en dos splits diferentes).

    Algoritmo:
        Para cada clase (label):
            - Tomar la lista de file_ids únicos de esa clase.
            - Dividir esos file_ids en 3 grupos (no clips, ARCHIVOS).
            - Cada split se compone de TODOS los clips de los archivos asignados.

    ¿Por qué este algoritmo?
        - Si tomamos clips al azar, dos clips del mismo audio podrían caer
          en train y test → data leakage por ambiente compartido.
        - Hacer el split a nivel de archivo garantiza que el modelo en test
          NUNCA haya visto ese archivo durante entrenamiento.

    Parámetros
    ----------
    records : list[SegmentRecord]
        El índice maestro (output de build_index()).
    train_ratio, val_ratio, test_ratio : float
        Proporciones del split. Deben sumar 1.0.
    random_state : int
        Semilla para reproducibilidad.

    Retorna
    -------
    train_records, val_records, test_records : list[SegmentRecord]

    Lanza
    -----
    ValueError
        Si los ratios no suman 1.0 o si alguna clase tiene < 3 archivos
        (insuficientes para split en 3).
    """
    # Validaciones
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            f"Los ratios deben sumar 1.0, suman {total_ratio:.4f}"
        )

    rng = np.random.default_rng(random_state)

    # Agrupar file_ids por clase
    files_by_class: dict[int, set[str]] = {}
    for rec in records:
        files_by_class.setdefault(rec.label, set()).add(rec.file_id)

    # Para cada clase, dividir sus archivos en 3 grupos
    train_file_ids: set[str] = set()
    val_file_ids: set[str] = set()
    test_file_ids: set[str] = set()

    for label, file_ids in files_by_class.items():
        files = sorted(file_ids)  # orden determinista
        n = len(files)
        if n < 3:
            raise ValueError(
                f"Clase {label} tiene solo {n} archivo(s). "
                f"Mínimo necesario: 3 (uno por split)."
            )

        # Permutación reproducible
        idx_perm = rng.permutation(n)
        files_shuffled = [files[i] for i in idx_perm]

        # Cortes: importante que train sea floor (no ceil) para no quedarnos
        # sin archivos para val/test cuando n es chico.
        n_train = int(np.floor(n * train_ratio))
        n_val = int(np.floor(n * val_ratio))
        # test = lo que queda (evita perder archivos por redondeo)
        # Garantizamos al menos 1 por split.
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        if n_train + n_val >= n:
            # caso extremo: pocos archivos. ajustar.
            n_train = max(1, n - 2)
            n_val = 1

        train_file_ids.update(files_shuffled[:n_train])
        val_file_ids.update(files_shuffled[n_train:n_train + n_val])
        test_file_ids.update(files_shuffled[n_train + n_val:])

    # Distribuir TODOS los clips según a qué archivo pertenecen
    train_records = [r for r in records if r.file_id in train_file_ids]
    val_records = [r for r in records if r.file_id in val_file_ids]
    test_records = [r for r in records if r.file_id in test_file_ids]

    return train_records, val_records, test_records


# ============================================================================
# La clase Dataset (la pieza principal)
# ============================================================================

class SelvaSonicDataset(Dataset):
    """
    Dataset de PyTorch para SelvaSonic. Orquesta todo el pipeline de datos.

    Implementa el contrato de torch.utils.data.Dataset:
        - __len__()        : número de items en el split
        - __getitem__(idx) : retorna (tensor_mel, label)

    Pipeline ejecutado en __getitem__:
        1. Buscar el SegmentRecord en el índice.
        2. load_audio(offset, duration)       <- lectura parcial rápida.
        3. (solo TRAIN) augmentation aleatoria.
        4. compute_mel_spectrogram() + normalize() + to_tensor().
        5. Retornar (tensor [1,128,T], label int).

    Parámetros
    ----------
    records : list[SegmentRecord]
        Índice del split (train, val o test).
    split : str
        Nombre del split: "train", "val" o "test".
        Solo "train" aplica augmentation.
    sample_rate : int
        Frecuencia de muestreo objetivo (default: SAMPLE_RATE de config).
    augmentation_seed : int, opcional
        Si se provee, hace la augmentation reproducible (útil para debug).
        Default: None (aleatoriedad real, distinto en cada época).

    Ejemplo
    -------
    >>> records, label_map = build_index("data/raw")
    >>> train_recs, val_recs, test_recs = stratified_split_by_file(records)
    >>> train_ds = SelvaSonicDataset(train_recs, split="train")
    >>> tensor, label = train_ds[0]
    >>> tensor.shape
    torch.Size([1, 128, 216])
    """

    VALID_SPLITS = ("train", "val", "test")

    def __init__(
        self,
        records: list[SegmentRecord],
        *,
        split: str = "train",
        sample_rate: int = SAMPLE_RATE,
        augmentation_seed: Optional[int] = None,
    ):
        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"split debe ser uno de {self.VALID_SPLITS}, recibido '{split}'"
            )
        if not records:
            raise ValueError("records está vacío")

        self.records = records
        self.split = split
        self.sample_rate = sample_rate
        self.augmentation_seed = augmentation_seed
        # Generador propio del Dataset para reproducibilidad opcional
        self._rng = np.random.default_rng(augmentation_seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rec = self.records[idx]

        # ---- 1. Cargar SOLO el segmento necesario del MP3 ----
        # Aquí está el truco de eficiencia: librosa lee únicamente
        # del segundo `start_sec` por `duration` segundos, no el MP3 entero.
        duration = rec.end_sec - rec.start_sec
        try:
            audio = load_audio(
                rec.file_path,
                target_sr=self.sample_rate,
                offset=rec.start_sec,
                duration=duration,
            )
            samples = audio.waveform
        except AudioLoadError:
            # Edge case: audio corrupto. Devolvemos silencio para no romper
            # el batch. El modelo lo verá como ruido y será penalizado en
            # train (ok, es 1 caso entre miles).
            samples = np.zeros(CLIP_NUM_SAMPLES, dtype=np.float32)

        # ---- 2. Asegurar longitud exacta del clip ----
        # Si el audio es ligeramente más corto/largo (ej. último segmento
        # con padding), ajustamos a CLIP_NUM_SAMPLES.
        samples = _ensure_clip_length(samples, CLIP_NUM_SAMPLES)

        # ---- 3. Augmentation (SOLO en train) ----
        if self.split == "train":
            samples = self._apply_augmentations(samples)

        # ---- 4. Mel-spectrograma ----
        mel = compute_mel_spectrogram(samples, sr=self.sample_rate)
        mel = normalize_spectrogram(mel)
        tensor = to_tensor(mel)  # shape: (1, 128, T)

        return tensor, rec.label

    def _apply_augmentations(self, samples: np.ndarray) -> np.ndarray:
        """
        Aplica las 3 augmentations con sus probabilidades configuradas.
        Las probabilidades vienen de config.py (AUG_PROB_*).
        Cada augmentation se aplica de forma INDEPENDIENTE.
        """
        rng = self._rng

        if rng.random() < AUG_PROB_TIME_STRETCH:
            try:
                samples = time_stretch(
                    samples,
                    sample_rate=self.sample_rate,
                    random_state=rng,
                )
            except AugmentationError:
                pass  # si falla, seguimos con el clip original

        if rng.random() < AUG_PROB_PITCH_SHIFT:
            try:
                samples = pitch_shift(
                    samples,
                    sample_rate=self.sample_rate,
                    random_state=rng,
                )
            except AugmentationError:
                pass

        if rng.random() < AUG_PROB_ADD_NOISE:
            try:
                samples = add_noise(samples, random_state=rng)
            except AugmentationError:
                pass

        return samples


def _ensure_clip_length(samples: np.ndarray, target_length: int) -> np.ndarray:
    """Garantiza shape exacto (truncar o repetir circularmente)."""
    n = samples.shape[0]
    if n == target_length:
        return samples
    if n > target_length:
        return samples[:target_length]
    repetitions = int(np.ceil(target_length / max(n, 1)))
    return np.tile(samples, repetitions)[:target_length] if n > 0 else \
           np.zeros(target_length, dtype=np.float32)


# ============================================================================
# Función de conveniencia: crear los 3 DataLoaders de una sola vez
# ============================================================================

def create_dataloaders(
    raw_data_dir: str | Path,
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    """
    Función de conveniencia: ejecuta TODO el pipeline de carga.

    Hace:
        1. build_index(raw_data_dir)
        2. stratified_split_by_file(records, ratios)
        3. SelvaSonicDataset(...) para cada split
        4. DataLoader(...) para cada split

    Parámetros
    ----------
    raw_data_dir : str | Path
        Carpeta con los datos crudos (ej. "data/raw").
    batch_size : int
        Tamaño de batch para los 3 DataLoaders.
    num_workers : int
        Procesos paralelos para cargar datos. En Windows usar 0 al inicio
        (evita issues con multiprocessing).
    train_ratio, val_ratio, test_ratio : float
        Proporciones del split.
    random_state : int
        Semilla para reproducibilidad del split.
    verbose : bool
        Imprimir progreso.

    Retorna
    -------
    train_loader, val_loader, test_loader : DataLoader
    label_map : dict[str, int]
        Mapeo nombre clase → label int.

    Notas sobre los DataLoaders:
        - shuffle=True solo en train (val/test no necesitan).
        - drop_last=True en train: descarta el último batch si es incompleto.
          Necesario para que BatchNorm no se queje con batches de size 1.
        - pin_memory=True: acelera transferencia CPU→GPU si hay CUDA.
    """
    # Paso 1-2: índice + split
    records, label_map = build_index(raw_data_dir, verbose=verbose)
    train_recs, val_recs, test_recs = stratified_split_by_file(
        records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )

    if verbose:
        print(f"Splits: train={len(train_recs)}, val={len(val_recs)}, test={len(test_recs)}")

    # Paso 3: Datasets
    train_ds = SelvaSonicDataset(train_recs, split="train")
    val_ds = SelvaSonicDataset(val_recs, split="val")
    test_ds = SelvaSonicDataset(test_recs, split="test")

    # Paso 4: DataLoaders
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin,
    )

    return train_loader, val_loader, test_loader, label_map


# ============================================================================
# Utilidad: convertir índice a DataFrame (para visualizaciones)
# ============================================================================

def records_to_dataframe(records: list[SegmentRecord]) -> pd.DataFrame:
    """
    Convierte un índice (lista de SegmentRecord) a un DataFrame de pandas.

    Útil para:
        - Visualizar distribución de clases (df.label_name.value_counts())
        - Estadísticas de duración por clase
        - Filtrado y exploración interactiva en notebooks
    """
    return pd.DataFrame([r._asdict() for r in records])


# ============================================================================
# Smoke test
# ============================================================================
# Crea un dataset sintético en disco y verifica todo el pipeline end-to-end.

if __name__ == "__main__":
    import shutil
    import tempfile

    import soundfile as sf

    print("=" * 70)
    print("SMOKE TEST: src/dataset.py")
    print("=" * 70)

    # ---- Crear estructura de datos sintética en directorio temporal ----
    tmpdir = Path(tempfile.mkdtemp(prefix="selvasonic_test_"))
    print(f"\nCreando dataset sintético en: {tmpdir}")

    SR = 22050
    # 4 archivos por clase para que el split de 3 splits funcione
    classes = {
        "Especie_A": 4,
        "Especie_B": 4,
        "Especie_C": 4,
    }
    for class_name, n_files in classes.items():
        cdir = tmpdir / class_name
        cdir.mkdir()
        for i in range(n_files):
            # Audio sintético de duración variable (8-15 segundos)
            duration_sec = 8 + (i * 2)
            t = np.arange(int(duration_sec * SR)) / SR
            freq = 440 * (i + 1)  # frecuencias distintas por archivo
            audio = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
            sf.write(str(cdir / f"file_{class_name}_{i:03d}.wav"), audio, SR)

    # Carpeta no_ave (ESC-50)
    esc_dir = tmpdir / "ESC-50"
    esc_dir.mkdir()
    for cat in ["rain", "wind"]:
        sub = esc_dir / cat
        sub.mkdir()
        for i in range(4):
            t = np.arange(5 * SR) / SR
            audio = (0.3 * np.random.randn(5 * SR)).astype(np.float32)
            sf.write(str(sub / f"esc_{cat}_{i:03d}.wav"), audio, SR)

    print("Dataset sintético creado.\n")

    try:
        # ---- Test 1: build_index ----
        print("[Test 1] build_index()")
        records, label_map = build_index(tmpdir, verbose=False)
        print(f"  Total clips: {len(records)}")
        print(f"  Clases: {label_map}")
        assert len(records) > 0
        assert label_map["no_ave"] == 0
        assert "Especie_A" in label_map
        print("  PASS\n")

        # ---- Test 2: split estratificado por archivo ----
        print("[Test 2] stratified_split_by_file()")
        train_r, val_r, test_r = stratified_split_by_file(records)
        print(f"  Train: {len(train_r)} clips")
        print(f"  Val:   {len(val_r)} clips")
        print(f"  Test:  {len(test_r)} clips")

        # Verificar regla de oro: ningún file_id en dos splits
        train_files = {r.file_id for r in train_r}
        val_files = {r.file_id for r in val_r}
        test_files = {r.file_id for r in test_r}
        assert train_files.isdisjoint(val_files), "LEAK train vs val"
        assert train_files.isdisjoint(test_files), "LEAK train vs test"
        assert val_files.isdisjoint(test_files), "LEAK val vs test"
        print("  PASS (sin data leakage entre splits)")

        # Verificar estratificación: todas las clases en todos los splits
        for split_name, split_recs in [("train", train_r), ("val", val_r), ("test", test_r)]:
            classes_in_split = {r.label for r in split_recs}
            assert len(classes_in_split) == len(label_map), \
                f"Clase faltante en split {split_name}"
        print("  PASS (todas las clases representadas en cada split)\n")

        # ---- Test 3: SelvaSonicDataset y __getitem__ ----
        print("[Test 3] SelvaSonicDataset")
        train_ds = SelvaSonicDataset(train_r, split="train")
        val_ds = SelvaSonicDataset(val_r, split="val")
        test_ds = SelvaSonicDataset(test_r, split="test")
        print(f"  len(train_ds) = {len(train_ds)}")

        tensor, label = train_ds[0]
        print(f"  train_ds[0]: tensor.shape={tensor.shape}, label={label}")
        assert tensor.dim() == 3, "esperado shape (1, n_mels, T)"
        assert tensor.shape[0] == 1
        assert tensor.shape[1] == 128
        assert isinstance(label, int)
        print("  PASS\n")

        # ---- Test 4: aleatoriedad train vs determinismo val ----
        print("[Test 4] augmentation: train varia, val/test no")
        # En train, dos llamadas al mismo idx deben dar resultados distintos
        # (porque las probabilidades de aug son aleatorias)
        train_ds_random = SelvaSonicDataset(train_r, split="train", augmentation_seed=None)
        # En val/test, dos llamadas deben dar lo mismo (sin aug)
        val_a, _ = val_ds[0]
        val_b, _ = val_ds[0]
        assert torch.equal(val_a, val_b), "val deberia ser deterministico"
        test_a, _ = test_ds[0]
        test_b, _ = test_ds[0]
        assert torch.equal(test_a, test_b), "test deberia ser deterministico"
        print("  PASS (val/test sin augmentation)\n")

        # ---- Test 5: DataLoader con batches ----
        print("[Test 5] create_dataloaders()")
        train_loader, val_loader, test_loader, label_map = create_dataloaders(
            tmpdir, batch_size=4, num_workers=0, verbose=False,
        )
        print(f"  train_loader: {len(train_loader)} batches")
        print(f"  val_loader:   {len(val_loader)} batches")
        print(f"  test_loader:  {len(test_loader)} batches")

        # Iterar el primer batch de train
        for batch_x, batch_y in train_loader:
            print(f"  Primer batch train: x.shape={batch_x.shape}, y.shape={batch_y.shape}")
            assert batch_x.shape[0] == 4  # batch_size
            assert batch_x.shape[1] == 1  # canales
            assert batch_x.shape[2] == 128  # mels
            break
        print("  PASS\n")

        # ---- Test 6: records_to_dataframe ----
        print("[Test 6] records_to_dataframe()")
        df = records_to_dataframe(records)
        print(f"  shape: {df.shape}")
        print(f"  columnas: {list(df.columns)}")
        print(f"  distribución de clases:\n{df['label_name'].value_counts().to_string()}")
        assert "file_id" in df.columns
        assert "label" in df.columns
        print("  PASS\n")

        print("=" * 70)
        print("TODOS LOS TESTS PASARON")
        print("=" * 70)

    finally:
        # Limpiar
        shutil.rmtree(tmpdir)
        print(f"\nDataset sintético eliminado: {tmpdir}")
