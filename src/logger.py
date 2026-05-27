"""
src/logger.py
=============

Módulo de logging con TensorBoard para el ciclo de entrenamiento de SelvaSonic.

¿Por qué TensorBoard?
---------------------
TensorBoard es la herramienta de visualización estándar para experimentos de
deep learning. Permite monitorear en tiempo real:
  - La evolución de pérdida y accuracy por época (detectar overfitting).
  - La curva de learning rate (verificar que el scheduler baja como debe).
  - Matrices de confusión por época (ver qué clases se confunden más).
  - El grafo del modelo (verificar la arquitectura visual).
  - Hiperparámetros vs métricas finales (pestaña HPARAMS, comparar runs).

Integración con Google Colab (donde se entrena el modelo real):
    %load_ext tensorboard
    %tensorboard --logdir runs/

Diseño: el patrón ``enabled=False`` permite deshabilitar el logger en smoke
tests o debugging sin modificar el código que lo usa.

Autoras(es)
-----------
Laura Ruiz Arango & Jose Aldair Molina Méndez
Universidad Nacional de Colombia - Sede Medellín
Aprendizaje Automático (Prof. Alcides Montoya) - 2026
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import matplotlib
# Backend sin ventana: evita errores en Colab y servidores sin display.
# Se establece antes del primer import de pyplot.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter

from src.config import (
    LOG_CONFUSION_MATRIX_EVERY_N_EPOCHS,
    TENSORBOARD_ENABLED,
    TENSORBOARD_LOG_DIR_NAME,
)


# ============================================================================
# Excepción personalizada
# ============================================================================

class LoggerError(Exception):
    """Error específico del módulo de logging con TensorBoard.

    Se lanza cuando:
    - El directorio de logs no se puede crear (permisos, disco lleno).
    - El SummaryWriter no está disponible (TensorBoard no instalado).
    """


# ============================================================================
# TrainingLogger — clase principal
# ============================================================================

class TrainingLogger:
    """Logger de TensorBoard para el ciclo de entrenamiento de SelvaSonic.

    Encapsula el SummaryWriter de torch.utils.tensorboard y agrega:
      - Jerarquía de nombres: "Loss/train", "Accuracy/val", etc. TensorBoard
        agrupa automáticamente métricas con el mismo prefijo en paneles.
      - Generación de matrices de confusión cada N épocas (configurable).
      - Registro del grafo del modelo (una sola vez, con manejo de errores).
      - Registro de hiperparámetros + métricas finales en la pestaña HPARAMS.

    Parámetros
    ----------
    log_dir : str | Path
        Directorio raíz donde guardar los logs de TensorBoard.
        Estructura creada: log_dir / run_name / tensorboard /
    run_name : str
        Nombre único del run (ej. "baseline_S3_v2_20260526").
        Aparecerá como nombre de la curva en TensorBoard.
    label_map : dict[str, int]
        Mapeo nombre de clase → label entero (output de dataset.build_index).
        Se usa para etiquetar filas/columnas de la matriz de confusión.
    hparams : dict
        Hiperparámetros del run (lr, batch_size, epochs, etc.).
        Se registran al final con add_hparams() para comparación en HPARAMS.
    enabled : bool
        Si False, todos los métodos se vuelven no-ops silenciosos.
        Default: TENSORBOARD_ENABLED de config.py.

    Ejemplo
    -------
    >>> label_map = {"no_ave": 0, "Ara_macao": 1, "Celeus_grammicus": 2}
    >>> logger = TrainingLogger(
    ...     log_dir="runs",
    ...     run_name="baseline_S3_20260526",
    ...     label_map=label_map,
    ...     hparams={"lr": 1e-3, "batch_size": 32, "epochs": 100},
    ... )
    >>> logger.log_epoch(
    ...     1,
    ...     train_loss=0.8, train_acc=0.60,
    ...     val_loss=0.9,   val_acc=0.55,
    ...     lr=1e-3,
    ... )
    >>> logger.log_hparams_final(metrics={"best_val_acc": 0.87})
    >>> logger.close()
    """

    def __init__(
        self,
        log_dir: str | Path,
        run_name: str,
        label_map: dict[str, int],
        hparams: dict,
        *,
        enabled: bool = TENSORBOARD_ENABLED,
    ) -> None:
        self.enabled = enabled
        self.run_name = run_name
        self.label_map = label_map
        self.hparams = hparams
        self._writer: Optional[SummaryWriter] = None

        if not self.enabled:
            return

        # Estructura: log_dir / run_name / tensorboard /
        tb_log_dir = Path(log_dir) / run_name / TENSORBOARD_LOG_DIR_NAME
        try:
            tb_log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise LoggerError(
                f"No se pudo crear el directorio de logs: {tb_log_dir}. "
                f"Verificar permisos y espacio en disco. Error original: {e}"
            ) from e

        self._tb_log_dir = tb_log_dir

        try:
            self._writer = SummaryWriter(log_dir=str(tb_log_dir))
        except Exception as e:
            raise LoggerError(
                f"No se pudo crear el SummaryWriter en {tb_log_dir}. "
                f"¿Está instalado tensorboard? (pip install tensorboard). "
                f"Error original: {e}"
            ) from e

    # -------------------------------------------------------------------------
    # Métodos públicos
    # -------------------------------------------------------------------------

    def log_epoch(
        self,
        epoch: int,
        *,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float,
    ) -> None:
        """Registra las métricas escalares de una época en TensorBoard.

        Los nombres jerárquicos con '/' agrupan métricas en paneles:
          - "Loss/train" y "Loss/val" aparecen juntas en el panel "Loss".
          - "Accuracy/train" y "Accuracy/val" en el panel "Accuracy".
          - "LR" en su propio panel.

        Esta estructura facilita comparar train vs val en la misma gráfica,
        que es el principal caso de uso (detectar overfitting).

        Parámetros
        ----------
        epoch : int
            Número de época (eje X en TensorBoard). Debe ser 1-indexed.
        train_loss : float
            Pérdida promedio en el split de entrenamiento.
        train_acc : float
            Accuracy en el split de entrenamiento (fracción en [0, 1]).
        val_loss : float
            Pérdida promedio en el split de validación.
        val_acc : float
            Accuracy en el split de validación.
        lr : float
            Learning rate activo en esta época (leído del optimizer).
        """
        if not self.enabled or self._writer is None:
            return

        self._writer.add_scalar("Loss/train", train_loss, epoch)
        self._writer.add_scalar("Loss/val", val_loss, epoch)
        self._writer.add_scalar("Accuracy/train", train_acc, epoch)
        self._writer.add_scalar("Accuracy/val", val_acc, epoch)
        self._writer.add_scalar("LR", lr, epoch)

    def log_confusion_matrix(
        self,
        epoch: int,
        *,
        y_true: list[int] | np.ndarray,
        y_pred: list[int] | np.ndarray,
        class_names: list[str],
    ) -> None:
        """Genera y registra una figura de matriz de confusión en TensorBoard.

        La figura se crea con matplotlib, se registra con writer.add_figure()
        y aparece en la pestaña "Images" de TensorBoard.

        ¿Por qué no cada época?
            Generar la matriz requiere: (a) forward pass sobre todo el val set,
            (b) generar figura matplotlib, (c) serializar a PNG en memoria.
            Hacerlo cada época añade ~10-30% de tiempo por época según el
            tamaño del val set. LOG_CONFUSION_MATRIX_EVERY_N_EPOCHS controla
            la frecuencia (default: 5).

        La matriz se NORMALIZA por fila: cada celda muestra qué fracción de
        esa clase real fue predicha como cada clase. Más informativa que
        conteos brutos cuando las clases están desbalanceadas.

        Parámetros
        ----------
        epoch : int
            Número de época (para el eje X de TensorBoard y el título).
        y_true : list[int] | np.ndarray
            Etiquetas reales del val o test set.
        y_pred : list[int] | np.ndarray
            Predicciones del modelo para las mismas muestras.
        class_names : list[str]
            Nombres de las clases en orden de label entero.
            Ejemplo: ["no_ave", "Ara_macao", "Celeus_grammicus"]
        """
        if not self.enabled or self._writer is None:
            return

        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        cm = confusion_matrix(
            y_true_arr, y_pred_arr,
            labels=list(range(len(class_names))),
        )

        fig = _plot_confusion_matrix(cm, class_names=class_names, epoch=epoch)
        self._writer.add_figure("ConfusionMatrix/val", fig, global_step=epoch)
        plt.close(fig)

    def log_model_graph(
        self,
        model: nn.Module,
        *,
        sample_input: torch.Tensor,
    ) -> None:
        """Registra el grafo computacional del modelo en TensorBoard.

        Se llama UNA SOLA VEZ al inicio del entrenamiento (epoch=1, antes del
        primer backward). Aparece en la pestaña "Graphs" de TensorBoard.

        ¿Por qué puede fallar?
            add_graph() usa torch.jit.trace internamente. El trace falla con:
            - Control flow dinámico (if/for sobre valores de tensores).
            - Algunas implementaciones de Multi-Head Attention con máscaras.
            - Módulos con side effects (ej. ciertos RNNs).
            Si falla, emitimos un Warning y seguimos. Los scalars y la matriz
            de confusión siguen funcionando con normalidad.

        Parámetros
        ----------
        model : nn.Module
            El modelo a registrar. No se modifica, solo se hace un trace.
        sample_input : torch.Tensor
            Tensor de ejemplo del shape correcto (ej. torch.zeros(1, 1, 128, 216)).
            Solo importa el shape, no los valores.
        """
        if not self.enabled or self._writer is None:
            return

        try:
            self._writer.add_graph(model, sample_input)
        except Exception as e:
            warnings.warn(
                f"[TrainingLogger] No se pudo registrar el grafo del modelo. "
                f"Los demás logs siguen funcionando normalmente. "
                f"Causa: {type(e).__name__}: {e}",
                stacklevel=2,
            )

    def log_hparams_final(self, *, metrics: dict[str, float]) -> None:
        """Registra hiperparámetros + métricas finales al terminar el entrenamiento.

        Usa writer.add_hparams(), que alimenta la pestaña HPARAMS de TensorBoard.
        Esta pestaña muestra una tabla donde cada fila es un run y las columnas
        son hiperparámetros y métricas. Ideal para comparar experimentos.

        ¿Por qué al final y no al inicio?
            add_hparams() necesita las métricas finales (best_val_acc, test_acc).
            Si se llama al inicio, la tabla aparece con métricas nulas. Llamar
            al final garantiza que los valores reflejan el run completo.

        Parámetros
        ----------
        metrics : dict[str, float]
            Métricas finales del run. Claves recomendadas:
            {"best_val_acc": float, "best_val_loss": float, "test_acc": float}

        Ejemplo
        -------
        >>> logger.log_hparams_final(metrics={
        ...     "best_val_acc": 0.87,
        ...     "best_val_loss": 0.42,
        ...     "test_acc": 0.84,
        ... })
        """
        if not self.enabled or self._writer is None:
            return

        # add_hparams() rechaza tipos no primitivos (list, tuple, None).
        # Convertimos a str como fallback seguro.
        safe_hparams = _sanitize_hparams(self.hparams)

        try:
            self._writer.add_hparams(safe_hparams, metrics)
        except Exception as e:
            warnings.warn(
                f"[TrainingLogger] No se pudo registrar hparams: "
                f"{type(e).__name__}: {e}",
                stacklevel=2,
            )

    def close(self) -> None:
        """Cierra el SummaryWriter y vacía el buffer pendiente a disco.

        Siempre llamar al final del entrenamiento (o usar como context manager).
        Si no se cierra limpiamente, los últimos eventos pueden no aparecer
        en TensorBoard (el writer tiene un buffer interno).
        """
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    # -------------------------------------------------------------------------
    # Context manager (permite: with TrainingLogger(...) as logger: ...)
    # -------------------------------------------------------------------------

    def __enter__(self) -> TrainingLogger:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ============================================================================
# Helpers privados
# ============================================================================

def _plot_confusion_matrix(
    cm: np.ndarray,
    *,
    class_names: list[str],
    epoch: int,
) -> plt.Figure:
    """Genera una figura matplotlib de la matriz de confusión normalizada.

    Normalización por fila: cada celda = fracción de esa clase real que fue
    predicha como cada clase. La diagonal = recall por clase.

    Parámetros
    ----------
    cm : np.ndarray
        Matriz de confusión cruda (output de sklearn.metrics.confusion_matrix).
        Shape: (n_classes, n_classes), dtype int.
    class_names : list[str]
        Nombres de las clases para etiquetar ejes.
    epoch : int
        Número de época (para el título de la figura).

    Retorna
    -------
    plt.Figure
        Figura lista para pasar a writer.add_figure() o plt.savefig().
    """
    n_classes = len(class_names)

    # Normalizar por fila evitando división por cero (clases sin muestras)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    # Tamaño dinámico: ~0.7 pulgadas por clase, mínimo 6x6
    fig_size = max(6.0, n_classes * 0.7)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(n_classes)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=8)

    # Texto en cada celda: proporción normalizada + conteo bruto entre paréntesis
    thresh = 0.5
    for i in range(n_classes):
        for j in range(n_classes):
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(
                j, i,
                f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                ha="center", va="center",
                color=color, fontsize=7,
            )

    ax.set_xlabel("Clase predicha", fontsize=10)
    ax.set_ylabel("Clase real", fontsize=10)
    ax.set_title(f"Matriz de confusión — Época {epoch}", fontsize=12, pad=12)

    fig.tight_layout()
    return fig


def _sanitize_hparams(hparams: dict) -> dict:
    """Convierte valores del dict a tipos primitivos aceptados por add_hparams().

    TensorBoard rechaza tipos como list, tuple, None, torch.Tensor.
    Los valores no primitivos se convierten a str.

    Parámetros
    ----------
    hparams : dict
        Diccionario de hiperparámetros crudo (puede contener cualquier tipo).

    Retorna
    -------
    dict
        Diccionario con claves str y valores bool | int | float | str.
    """
    safe: dict = {}
    for k, v in hparams.items():
        if isinstance(v, (bool, int, float, str)):
            safe[str(k)] = v
        else:
            # Fallback: convertir a str para que add_hparams no falle
            safe[str(k)] = str(v)
    return safe


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == "__main__":
    import shutil
    import tempfile

    print("=" * 70)
    print("SMOKE TEST: src/logger.py")
    print("=" * 70)

    tmpdir = Path(tempfile.mkdtemp(prefix="selvasonic_logger_test_"))
    print(f"\nDirectorio temporal: {tmpdir}")

    label_map: dict[str, int] = {
        "no_ave": 0,
        "Ara_macao": 1,
        "Celeus_grammicus": 2,
        "Ramphastos_tucanus": 3,
    }
    class_names = sorted(label_map.keys(), key=lambda k: label_map[k])

    hparams: dict = {
        "lr": 1e-3,
        "batch_size": 32,
        "epochs": 100,
        "dropout": 0.3,
        "label_smoothing": 0.1,
    }

    try:
        # ---- Test 1: instanciar TrainingLogger ----
        print("\n[Test 1] Instanciar TrainingLogger...")
        logger = TrainingLogger(
            log_dir=str(tmpdir),
            run_name="smoke_test_run",
            label_map=label_map,
            hparams=hparams,
            enabled=True,
        )
        assert logger._writer is not None
        print("  PASS: SummaryWriter creado correctamente")

        # ---- Test 2: log_epoch con valores realistas ----
        print("\n[Test 2] Registrar 6 épocas de prueba...")
        rng = np.random.default_rng(42)
        for epoch in range(1, 7):
            train_loss = 1.2 - epoch * 0.12 + float(rng.normal(0, 0.02))
            val_loss   = 1.3 - epoch * 0.10 + float(rng.normal(0, 0.03))
            logger.log_epoch(
                epoch,
                train_loss=float(train_loss),
                train_acc=float(0.30 + epoch * 0.08),
                val_loss=float(val_loss),
                val_acc=float(0.27 + epoch * 0.07),
                lr=float(1e-3 * (0.9 ** epoch)),
            )
        print("  PASS: 6 épocas registradas (Loss/*, Accuracy/*, LR)")

        # ---- Test 3: log_confusion_matrix ----
        print("\n[Test 3] Registrar matriz de confusión (época 5)...")
        n_val = 80
        y_true = rng.integers(0, len(class_names), size=n_val)
        y_pred = rng.integers(0, len(class_names), size=n_val)
        logger.log_confusion_matrix(
            5,
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
        )
        print("  PASS: figura matplotlib generada y registrada como imagen")

        # ---- Test 4: log_hparams_final ----
        print("\n[Test 4] Registrar hparams finales...")
        logger.log_hparams_final(metrics={
            "best_val_acc": 0.78,
            "best_val_loss": 0.62,
        })
        print("  PASS: hparams + métricas registrados (pestaña HPARAMS)")

        # ---- Test 5: close() ----
        print("\n[Test 5] Cerrar el writer...")
        logger.close()
        assert logger._writer is None
        print("  PASS: writer cerrado, buffer vaciado a disco")

        # ---- Test 6: verificar archivos .tfevents en disco ----
        print("\n[Test 6] Verificar archivos .tfevents en disco...")
        tb_dir = tmpdir / "smoke_test_run" / TENSORBOARD_LOG_DIR_NAME
        tfevents_files = list(tb_dir.glob("*.tfevents.*"))
        assert len(tfevents_files) > 0, (
            f"No se encontraron archivos .tfevents en {tb_dir}"
        )
        print(f"  PASS: {len(tfevents_files)} archivo(s) .tfevents encontrado(s):")
        for f in tfevents_files:
            size_kb = f.stat().st_size / 1024
            print(f"    {f.name} ({size_kb:.1f} KB)")

        # ---- Test 7: enabled=False → no-op silencioso ----
        print("\n[Test 7] TrainingLogger(enabled=False) es no-op...")
        null_logger = TrainingLogger(
            log_dir=str(tmpdir),
            run_name="null_run",
            label_map=label_map,
            hparams=hparams,
            enabled=False,
        )
        # Ninguno de estos debe lanzar excepción
        null_logger.log_epoch(
            1,
            train_loss=0.5, train_acc=0.6,
            val_loss=0.6, val_acc=0.55,
            lr=1e-3,
        )
        null_logger.log_confusion_matrix(
            1, y_true=[0, 1], y_pred=[0, 0], class_names=class_names
        )
        null_logger.log_hparams_final(metrics={"best_val_acc": 0.8})
        null_logger.close()
        assert null_logger._writer is None
        # enabled=False no debe crear subdirectorios
        assert not (tmpdir / "null_run").exists()
        print("  PASS: enabled=False no hace nada y no crea archivos")

        # ---- Test 8: context manager (__enter__ / __exit__) ----
        print("\n[Test 8] Context manager...")
        with TrainingLogger(
            log_dir=str(tmpdir),
            run_name="ctx_manager_test",
            label_map=label_map,
            hparams=hparams,
            enabled=True,
        ) as ctx_logger:
            ctx_logger.log_epoch(
                1,
                train_loss=0.9, train_acc=0.5,
                val_loss=1.0, val_acc=0.45,
                lr=1e-3,
            )
        # Al salir del bloque `with`, el writer debe estar cerrado
        assert ctx_logger._writer is None
        print("  PASS: context manager cierra automáticamente al salir")

        # ---- Test 9: log_model_graph (arquitectura mínima) ----
        print("\n[Test 9] Registrar grafo del modelo...")
        with TrainingLogger(
            log_dir=str(tmpdir),
            run_name="graph_test",
            label_map=label_map,
            hparams=hparams,
            enabled=True,
        ) as graph_logger:
            tiny_model = torch.nn.Linear(10, 4)
            sample = torch.zeros(1, 10)
            graph_logger.log_model_graph(tiny_model, sample_input=sample)
        print("  PASS: grafo registrado (o warning emitido si no soportado)")

        print("\n" + "=" * 70)
        print("TODOS LOS TESTS PASARON")
        print("=" * 70)
        print(f"\nPara visualizar los logs generados por este test:")
        print(f"  tensorboard --logdir {tmpdir}")

    finally:
        shutil.rmtree(tmpdir)
        print(f"\nDirectorio temporal eliminado: {tmpdir}")
