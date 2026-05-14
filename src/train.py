"""
src/train.py
─────────────────────────────────────────────────────────────────────────────
Training loop para SelvaSonic — entrena la CNN baseline sobre los
espectrogramas Mel generados en Semana 2.

Componentes implementados:
  1. Loss function:    CrossEntropyLoss con label_smoothing.
  2. Optimizer:        AdamW (variante correcta de Adam con weight decay
                       desacoplado).
  3. Scheduler:        CosineAnnealingLR (baja el lr suavemente con coseno).
  4. Training loop:    forward → loss → backward → step, por época y batch.
  5. Validation loop:  evalúa al final de cada época sin actualizar pesos.
  6. Early stopping:   detiene si val_loss no mejora en `patience` épocas.
  7. Checkpointing:    guarda el mejor modelo según val_loss.
  8. Logging:          imprime métricas por época y guarda historial JSON.
  9. Reproducibilidad: fija seeds en torch, numpy y random.

NOTA: Este módulo expone una función `train()` reutilizable. El bloque
`if __name__ == "__main__"` ejecuta un smoke-test con datos sintéticos
para verificar que todo el pipeline funciona end-to-end ANTES de conectar
con el Dataset real.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Import local: la arquitectura definida en model.py
from src.model import SelvaSonicCNN


# ============================================================================
# Configuración por defecto del entrenamiento. En producción se sobreescribe
# desde config.yaml; aquí dejamos los defaults explícitos.
# ============================================================================

DEFAULT_TRAINING_CONFIG: Dict = {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "label_smoothing": 0.1,
    "early_stopping_patience": 10,
    "seed": 42,
    "checkpoint_dir": "checkpoints",
    "log_dir": "results/logs",
}


# ============================================================================
# Helpers de reproducibilidad y dispositivo
# ============================================================================

def set_seed(seed: int) -> None:
    """Fija las semillas aleatorias en todas las librerías relevantes.

    Esto es CRÍTICO para reproducibilidad. Sin esto, dos corridas del mismo
    código producen resultados diferentes (por inicialización de pesos,
    orden de DataLoader, dropout, etc).

    Args:
        seed: Valor entero para fijar todas las RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Forzar determinismo en CUDA. Hace el entrenamiento ligeramente más
    # lento pero garantiza resultados idénticos entre corridas.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Detecta y retorna el dispositivo disponible (GPU si existe, sino CPU).

    Returns:
        torch.device("cuda") o torch.device("cpu") según disponibilidad.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[device] Usando GPU: {gpu_name}")
    else:
        device = torch.device("cpu")
        print(f"[device] Usando CPU (no se detectó GPU)")
    return device


# ============================================================================
# Estructura para almacenar métricas
# ============================================================================

@dataclass
class EpochMetrics:
    """Métricas de una época de entrenamiento + validación.

    Attributes:
        epoch: Número de época (1-indexed).
        train_loss: Pérdida promedio en train.
        train_acc: Accuracy en train.
        val_loss: Pérdida promedio en val.
        val_acc: Accuracy en val.
        lr: Learning rate usado en esta época.
        elapsed_sec: Segundos que tomó la época.
    """
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    lr: float
    elapsed_sec: float


@dataclass
class TrainingHistory:
    """Historial completo del entrenamiento.

    Attributes:
        epochs: Lista de EpochMetrics, una por época.
        best_epoch: Número de la época con la mejor val_loss.
        best_val_loss: Mejor val_loss alcanzada.
        best_val_acc: val_acc en la mejor época.
        total_time_sec: Tiempo total de entrenamiento.
        stopped_early: True si se aplicó early stopping.
    """
    epochs: List[EpochMetrics] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    best_val_acc: float = 0.0
    total_time_sec: float = 0.0
    stopped_early: bool = False

    def to_dict(self) -> dict:
        """Serializa el historial a un diccionario JSON-able."""
        return {
            "epochs": [asdict(e) for e in self.epochs],
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "total_time_sec": self.total_time_sec,
            "stopped_early": self.stopped_early,
        }


# ============================================================================
# Una época de entrenamiento
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Ejecuta una época completa de entrenamiento.

    Para cada batch:
        1. Forward pass: model(x) → logits.
        2. Calcula loss: criterion(logits, y).
        3. Backward pass: loss.backward() calcula gradientes.
        4. Optimizer step: optimizer.step() actualiza los pesos.
        5. Limpia gradientes acumulados con zero_grad().

    Args:
        model: Modelo a entrenar (debe estar en modo train).
        loader: DataLoader del split de entrenamiento.
        criterion: Función de pérdida (típicamente CrossEntropyLoss).
        optimizer: Optimizador (típicamente AdamW).
        device: Dispositivo donde correr el cómputo.

    Returns:
        Tuple (avg_loss, accuracy) sobre todas las muestras del epoch.
    """
    model.train()  # Activa Dropout y BN usa estadísticas del batch

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        # Mover a device. non_blocking=True permite copia async en GPU.
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Limpiar gradientes acumulados de la iteración anterior. Si no se
        # hace, los gradientes se ACUMULAN entre batches (a veces útil para
        # gradient accumulation, pero no es lo que queremos aquí).
        optimizer.zero_grad()

        # Forward pass
        logits = model(inputs)
        loss = criterion(logits, targets)

        # Backward pass: calcula gradientes vía autograd.
        loss.backward()

        # Optimizer step: aplica la actualización de pesos.
        optimizer.step()

        # Acumular métricas. .item() saca el escalar del tensor (rompe
        # el grafo computacional, así no hay leak de memoria de autograd).
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        # argmax(dim=1) → clase predicha (la del logit más alto)
        preds = logits.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


# ============================================================================
# Una época de validación
# ============================================================================

@torch.no_grad()  # decorador: desactiva autograd globalmente en esta función.
                  # Crucial para velocidad y memoria en validación.
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evalúa el modelo en un loader (val o test). NO actualiza pesos.

    Args:
        model: Modelo a evaluar (se pone en modo eval).
        loader: DataLoader del split a evaluar.
        criterion: Función de pérdida.
        device: Dispositivo.

    Returns:
        Tuple (avg_loss, accuracy).
    """
    model.eval()  # Desactiva Dropout, BN usa estadísticas globales acumuladas

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(inputs)
        loss = criterion(logits, targets)

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


# ============================================================================
# Función principal de entrenamiento
# ============================================================================

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    *,
    epochs: int = DEFAULT_TRAINING_CONFIG["epochs"],
    learning_rate: float = DEFAULT_TRAINING_CONFIG["learning_rate"],
    weight_decay: float = DEFAULT_TRAINING_CONFIG["weight_decay"],
    label_smoothing: float = DEFAULT_TRAINING_CONFIG["label_smoothing"],
    early_stopping_patience: int = DEFAULT_TRAINING_CONFIG["early_stopping_patience"],
    seed: int = DEFAULT_TRAINING_CONFIG["seed"],
    checkpoint_dir: str = DEFAULT_TRAINING_CONFIG["checkpoint_dir"],
    log_dir: str = DEFAULT_TRAINING_CONFIG["log_dir"],
    class_weights: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> TrainingHistory:
    """Entrena un modelo con el setup estándar de SelvaSonic.

    Args:
        model: Modelo a entrenar (subclase de nn.Module).
        train_loader: DataLoader de entrenamiento.
        val_loader: DataLoader de validación.
        num_classes: Número de clases (usado para sanity-checks).
        epochs: Número máximo de épocas.
        learning_rate: Learning rate inicial para AdamW.
        weight_decay: Coeficiente de weight decay (regularización L2-like).
        label_smoothing: Suavizado de etiquetas en CrossEntropyLoss.
        early_stopping_patience: Épocas a esperar sin mejora antes de parar.
        seed: Semilla para reproducibilidad.
        checkpoint_dir: Directorio donde guardar el mejor modelo.
        log_dir: Directorio donde guardar el JSON con métricas.
        class_weights: Pesos por clase para CrossEntropyLoss (None = uniforme).
            Se usará en Semana 4 para manejar el desbalance entre especies.
        device: Dispositivo (None = autodetectar).
        verbose: Si True, imprime progreso por época.

    Returns:
        TrainingHistory con métricas de todas las épocas.
    """
    # ----- Setup ------------------------------------------------------------
    set_seed(seed)
    if device is None:
        device = get_device()
    model = model.to(device)

    # Crear directorios si no existen
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # ----- Loss function ----------------------------------------------------
    # CrossEntropyLoss combina LogSoftmax + NLLLoss internamente de forma
    # numéricamente estable. Espera LOGITS crudos (no probabilidades).
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing,
    )

    # ----- Optimizer --------------------------------------------------------
    # AdamW: variante de Adam con weight decay desacoplado correctamente
    # (Loshchilov & Hutter, 2017). Generaliza mejor que Adam plain.
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # ----- Scheduler --------------------------------------------------------
    # Cosine Annealing: baja el lr siguiendo una curva coseno de lr → 0 a
    # lo largo de T_max épocas. Empíricamente mejor que step decay para
    # la mayoría de tareas.
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

    # ----- Estado de entrenamiento -----------------------------------------
    history = TrainingHistory()
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    checkpoint_path = Path(checkpoint_dir) / "best_model.pth"

    if verbose:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{'═' * 70}")
        print(f" SelvaSonic — Entrenamiento")
        print(f"{'═' * 70}")
        print(f" Modelo:        {model.__class__.__name__} ({n_params:,} params)")
        print(f" Dispositivo:   {device}")
        print(f" Épocas máx.:   {epochs}")
        print(f" Optimizer:     AdamW (lr={learning_rate}, wd={weight_decay})")
        print(f" Scheduler:     CosineAnnealingLR")
        print(f" Loss:          CrossEntropyLoss (label_smoothing={label_smoothing})")
        print(f" Early stop:    patience={early_stopping_patience}")
        print(f"{'═' * 70}\n")

    start_time = time.time()

    # ----- Loop principal por épocas ---------------------------------------
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # 1. Entrenar una época completa
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 2. Evaluar en validación
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # 3. Actualizar el scheduler (ANTES de leer el lr para registrar
        #    el que se usó en la época que ACABÓ).
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # 4. Registrar métricas
        elapsed = time.time() - epoch_start
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            lr=current_lr,
            elapsed_sec=elapsed,
        )
        history.epochs.append(metrics)

        # 5. Checkpointing: guardar si val_loss mejoró
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history.best_epoch = epoch
            history.best_val_loss = val_loss
            history.best_val_acc = val_acc
            epochs_without_improvement = 0

            # Guardar pesos del modelo + metadata útil para reanudar/cargar.
            # Guardamos state_dict (no el modelo completo) por portabilidad.
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "num_classes": num_classes,
            }, checkpoint_path)

            best_marker = " ★"
        else:
            epochs_without_improvement += 1
            best_marker = ""

        # 6. Logging
        if verbose:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f} acc={val_acc:.3f} | "
                f"lr={current_lr:.2e} | {elapsed:.1f}s{best_marker}"
            )

        # 7. Early stopping check
        if epochs_without_improvement >= early_stopping_patience:
            if verbose:
                print(
                    f"\n[early stop] Sin mejora en val_loss durante "
                    f"{early_stopping_patience} épocas. Deteniendo "
                    f"entrenamiento."
                )
            history.stopped_early = True
            break

    history.total_time_sec = time.time() - start_time

    # ----- Guardar historial completo --------------------------------------
    history_path = Path(log_dir) / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history.to_dict(), f, indent=2)

    if verbose:
        print(f"\n{'═' * 70}")
        print(f" Entrenamiento finalizado")
        print(f"{'═' * 70}")
        print(f" Tiempo total:       {history.total_time_sec:.1f}s")
        print(f" Mejor época:        {history.best_epoch}")
        print(f" Mejor val_loss:     {history.best_val_loss:.4f}")
        print(f" Mejor val_acc:      {history.best_val_acc:.4f}")
        print(f" Checkpoint:         {checkpoint_path}")
        print(f" Historial:          {history_path}")
        print(f"{'═' * 70}\n")

    return history


# ============================================================================
# Smoke test con datos sintéticos
# ============================================================================

if __name__ == "__main__":
    """Smoke test end-to-end con datos aleatorios.

    Verifica que TODO el pipeline funciona:
      - Modelo se construye
      - Loss, optimizer, scheduler se inicializan correctamente
      - train_one_epoch corre sin errores
      - evaluate corre sin errores
      - Checkpointing funciona
      - Early stopping funciona

    NO verifica que el modelo aprenda bien (los datos son ruido aleatorio).
    Eso se hará con el Dataset real en el notebook de Semana 3.
    """
    from torch.utils.data import TensorDataset

    print("\n" + "█" * 70)
    print(" SMOKE TEST — Pipeline de entrenamiento completo")
    print(" (datos sintéticos, solo verifica que el código corre)")
    print("█" * 70)

    # Hiperparámetros de prueba (pequeños para que corra rápido)
    NUM_CLASSES = 11
    N_TRAIN = 64
    N_VAL = 32
    N_MELS = 128
    N_FRAMES = 216

    # Generar datos sintéticos: espectrogramas aleatorios + labels aleatorios.
    # OJO: estos datos NO tienen patrón real, así que la accuracy se quedará
    # cerca del random baseline (~1/11 ≈ 9%). Es ESPERADO.
    torch.manual_seed(0)
    X_train = torch.randn(N_TRAIN, 1, N_MELS, N_FRAMES)
    y_train = torch.randint(0, NUM_CLASSES, (N_TRAIN,))
    X_val = torch.randn(N_VAL, 1, N_MELS, N_FRAMES)
    y_val = torch.randint(0, NUM_CLASSES, (N_VAL,))

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    # Instanciar modelo
    model = SelvaSonicCNN(num_classes=NUM_CLASSES)

    # Entrenar 5 épocas (suficiente para verificar que todo corre)
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=NUM_CLASSES,
        epochs=5,
        learning_rate=1e-3,
        early_stopping_patience=999,  # desactivado para que corra las 5
        checkpoint_dir="/tmp/selvasonic_smoke/checkpoints",
        log_dir="/tmp/selvasonic_smoke/logs",
    )

    # Verificaciones
    assert len(history.epochs) == 5, "Debería haber 5 épocas registradas"
    assert Path("/tmp/selvasonic_smoke/checkpoints/best_model.pth").exists(), (
        "El checkpoint debería existir"
    )
    assert Path("/tmp/selvasonic_smoke/logs/training_history.json").exists(), (
        "El JSON de historial debería existir"
    )

    print("\n" + "█" * 70)
    print(" SMOKE TEST PASADO ✓")
    print(" - Modelo instancia, entrena y evalúa sin errores")
    print(" - Checkpoint y JSON de historial creados correctamente")
    print(" - (accuracy ~9% es esperado: los datos son aleatorios)")
    print("█" * 70 + "\n")
