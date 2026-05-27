# TensorBoard en SelvaSonic

## ¿Qué es TensorBoard?

TensorBoard es la herramienta de visualización estándar para experimentos de
deep learning (desarrollada por Google, compatible con PyTorch desde 2019).
Ofrece una interfaz web interactiva con paneles de:

- **Scalars**: curvas de pérdida, accuracy y learning rate por época.
- **Images**: matrices de confusión visualizadas por época.
- **Graphs**: grafo computacional del modelo (arquitectura visual).
- **HPARAMS**: tabla comparativa de experimentos (hiperparámetros vs métricas).

## ¿Por qué se usa en este proyecto?

El entrenamiento de SelvaSonic dura entre 1 y 3 horas en Colab con GPU.
Sin visualización en tiempo real es imposible detectar a tiempo:

- **Overfitting**: `val_loss` sube mientras `train_loss` sigue bajando.
- **Divergencia del LR**: si el scheduler está mal configurado, la curva de pérdida
  no converge o explota.
- **Clases problemáticas**: la matriz de confusión muestra qué especies se confunden
  entre sí (ej. Ara_macao vs Ara_chloropterus por similitud espectral).
- **Comparación de experimentos**: la pestaña HPARAMS permite comparar runs con
  diferentes learning rates, batch sizes o arquitecturas en una sola tabla.

## Métricas registradas

| Nombre en TensorBoard   | Descripción |
|-------------------------|-------------|
| `Loss/train`            | CrossEntropyLoss promedio en train por época |
| `Loss/val`              | CrossEntropyLoss promedio en validación por época |
| `Accuracy/train`        | Accuracy en train por época |
| `Accuracy/val`          | Accuracy en validación por época |
| `LR`                    | Learning rate activo (del scheduler) por época |
| `ConfusionMatrix/val`   | Imagen de la matriz de confusión normalizada (cada 5 épocas) |
| `Graphs`                | Grafo del modelo (una vez al inicio del entrenamiento) |

## Cómo se generan los logs durante el entrenamiento

Los logs se generan automáticamente con `src/logger.py` mediante la clase
`TrainingLogger`. Se integra con el loop de entrenamiento de `src/train.py`:

```python
from src.logger import TrainingLogger
from src.config import LOG_CONFUSION_MATRIX_EVERY_N_EPOCHS

logger = TrainingLogger(
    log_dir="runs",
    run_name="baseline_S3_v2_20260526",
    label_map=label_map,        # de dataset.build_index()
    hparams={
        "lr": 1e-3,
        "batch_size": 32,
        "epochs": 100,
        "dropout": 0.3,
    },
)

for epoch in range(1, epochs + 1):
    # ... loop de entrenamiento ...

    logger.log_epoch(
        epoch,
        train_loss=train_loss,
        train_acc=train_acc,
        val_loss=val_loss,
        val_acc=val_acc,
        lr=current_lr,
    )

    if epoch % LOG_CONFUSION_MATRIX_EVERY_N_EPOCHS == 0:
        logger.log_confusion_matrix(
            epoch,
            y_true=all_targets,
            y_pred=all_predictions,
            class_names=class_names,
        )

logger.log_hparams_final(metrics={
    "best_val_acc": history.best_val_acc,
    "best_val_loss": history.best_val_loss,
})
logger.close()
```

Los archivos de eventos se guardan en:

```
runs/
└── baseline_S3_v2_20260526/
    └── tensorboard/
        └── events.out.tfevents.TIMESTAMP.HOSTNAME
```

## Cómo visualizar los logs

### Escenario 1: Desde Colab durante el entrenamiento

Ejecutar en una celda **antes** de iniciar el entrenamiento:

```python
%load_ext tensorboard
%tensorboard --logdir runs/
```

TensorBoard se abre en un panel dentro del notebook y se actualiza automáticamente
mientras el entrenamiento corre. Para ver múltiples runs juntos, apuntar al
directorio raíz `runs/` (no a un run específico).

### Escenario 2: Desde Colab después del entrenamiento

Si los logs están en Google Drive:

```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/SelvaSonic/runs/
```

Si están en el directorio local de Colab (se perdieron al cerrar la sesión,
usar el Escenario 3 para recuperarlos desde Drive).

### Escenario 3: Desde la máquina local (descargando logs de Drive)

1. Descargar la carpeta `runs/` desde Google Drive a tu máquina local.

2. Activar el entorno virtual de SelvaSonic:

   ```powershell
   # Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   ```

3. Lanzar TensorBoard:

   ```bash
   tensorboard --logdir runs/
   ```

4. Abrir en el navegador: `http://localhost:6006`

   TensorBoard detecta automáticamente todos los runs dentro de `runs/` y
   los muestra como curvas separadas con el nombre del run como leyenda.

## Recuperar runs anteriores con `history_to_tensorboard.py`

Si corriste entrenamiento **antes** de integrar TensorBoard (por ejemplo, los
runs de Semana 3 guardados en Drive), puedes reconstruir los logs a posteriori:

```bash
python scripts/history_to_tensorboard.py \
    --history-json results/logs/training_history.json \
    --output-dir /tmp/tb_recovered \
    --run-name "baseline_S3_v1_recovered"
```

Luego visualizar:

```bash
tensorboard --logdir /tmp/tb_recovered
```

O en Colab:

```python
!python scripts/history_to_tensorboard.py \
    --history-json /content/drive/MyDrive/SelvaSonic/results/training_history.json \
    --output-dir /tmp/tb_recovered

%load_ext tensorboard
%tensorboard --logdir /tmp/tb_recovered
```

### Formatos de history.json soportados

El script detecta automáticamente el formato:

**Formato A** (moderno — generado por `TrainingLogger`):
```json
{
  "history": {
    "train_loss":   [1.20, 1.10, 0.95],
    "train_acc":    [0.45, 0.55, 0.62],
    "val_loss":     [1.30, 1.15, 0.98],
    "val_acc":      [0.42, 0.52, 0.60],
    "lr":           [0.001, 0.0009, 0.0008],
    "epoch_time_s": [12.3, 12.1, 12.5]
  },
  "best_val_acc": 0.78,
  "run_name": "baseline_S3_v1"
}
```

**Formato B** (legado — generado por `train.py` sin logger):
```json
{
  "epochs": [
    {"epoch": 1, "train_loss": 1.20, "train_acc": 0.45,
     "val_loss": 1.30, "val_acc": 0.42, "lr": 0.001, "elapsed_sec": 12.3},
    {"epoch": 2, "train_loss": 1.10, "train_acc": 0.55,
     "val_loss": 1.15, "val_acc": 0.52, "lr": 0.0009, "elapsed_sec": 12.1}
  ],
  "best_val_loss": 0.95,
  "best_val_acc": 0.78
}
```

## Pestaña HPARAMS — Comparar experimentos

La pestaña HPARAMS de TensorBoard muestra una tabla comparativa:

| run_name               | lr     | batch_size | dropout | best_val_acc |
|------------------------|--------|------------|---------|--------------|
| baseline_S3_v1         | 0.001  | 32         | 0.3     | 0.78         |
| baseline_S3_lr_low     | 0.0003 | 32         | 0.3     | 0.81         |
| baseline_S4_attention  | 0.001  | 16         | 0.5     | 0.87         |

Se genera llamando a `logger.log_hparams_final(metrics={...})` al final del
entrenamiento. Cada run que llamó a este método aparece como una fila en la tabla.

## Desactivar TensorBoard (para tests o debugging rápido)

```python
# En config.py o al instanciar manualmente:
logger = TrainingLogger(..., enabled=False)
# Todos los métodos son no-ops; no se crean archivos ni directorios.
```

También se puede establecer `TENSORBOARD_ENABLED = False` en `src/config.py`
para deshabilitar globalmente (útil en CI/CD o smoke tests automáticos).
