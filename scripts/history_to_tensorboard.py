"""
scripts/history_to_tensorboard.py
==================================

Convierte un history.json existente en logs de TensorBoard a posteriori.

¿Para qué sirve?
----------------
Si corriste entrenamiento ANTES de integrar TrainingLogger (ej. los runs
de Semana 3 guardados en Drive), puedes recuperar las curvas de pérdida y
accuracy en TensorBoard sin necesidad de re-entrenar.

El script lee el historial JSON, itera cada época y llama a
SummaryWriter.add_scalar() exactamente como lo haría el TrainingLogger
en tiempo real. El resultado es indistinguible de un log generado durante
el entrenamiento.

Estructura esperada del history.json (formato A — moderno):
    {
      "history": {
        "train_loss":   [1.20, 1.10, 0.95, ...],
        "train_acc":    [0.45, 0.55, 0.62, ...],
        "val_loss":     [1.30, 1.15, 0.98, ...],
        "val_acc":      [0.42, 0.52, 0.60, ...],
        "lr":           [0.001, 0.0009, ...],
        "epoch_time_s": [12.3, 12.1, ...]
      },
      "best_val_acc": 0.78,
      "run_name": "baseline_S3_v1",
      "label_map": {"no_ave": 0, "Ara_macao": 1, ...}
    }

Formato B (legado — generado por train.py antes de la integración de logger):
    {
      "epochs": [
        {"epoch": 1, "train_loss": 1.20, "train_acc": 0.45,
         "val_loss": 1.30, "val_acc": 0.42, "lr": 0.001, "elapsed_sec": 12.3},
        ...
      ],
      "best_val_loss": 0.95,
      "best_val_acc": 0.78
    }

El script detecta automáticamente el formato y convierte B → A.

Uso:
    python scripts/history_to_tensorboard.py \\
        --history-json results/logs/training_history.json \\
        --output-dir /tmp/tb_recovered \\
        --run-name "baseline_S3_v1_recovered"

    tensorboard --logdir /tmp/tb_recovered

En Google Colab:
    !python scripts/history_to_tensorboard.py \\
        --history-json /content/drive/MyDrive/SelvaSonic/results/training_history.json \\
        --output-dir /tmp/tb_recovered

    %load_ext tensorboard
    %tensorboard --logdir /tmp/tb_recovered

Autoras(es)
-----------
Laura Ruiz Arango & Jose Aldair Molina Méndez
Universidad Nacional de Colombia - Sede Medellín
Aprendizaje Automático (Prof. Alcides Montoya) - 2026
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


# ============================================================================
# Carga y validación del historial
# ============================================================================

def load_history(history_path: Path) -> dict:
    """Lee y parsea el history.json desde disco.

    Parámetros
    ----------
    history_path : Path
        Ruta al archivo history.json.

    Retorna
    -------
    dict
        El historial parseado como diccionario Python.

    Lanza
    -----
    SystemExit
        Si el archivo no existe o el JSON es inválido.
    """
    if not history_path.exists():
        print(f"[ERROR] No se encontró el archivo: {history_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(history_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(
            f"[ERROR] JSON inválido en {history_path}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    return data


def validate_and_normalize(history_data: dict) -> dict:
    """Valida la estructura del historial y lo normaliza al formato A.

    Acepta dos formatos históricos del proyecto:

    Formato A (moderno, generado por TrainingLogger):
        {"history": {"train_loss": [...], "train_acc": [...], ...}}

    Formato B (legado, generado por train.py sin TensorBoard):
        {"epochs": [{"train_loss": ..., "train_acc": ..., ...}, ...]}

    Retorna siempre Formato A. Advierte si las listas tienen longitudes distintas.

    Parámetros
    ----------
    history_data : dict
        Diccionario cargado desde JSON.

    Retorna
    -------
    dict
        Diccionario con clave "history" que contiene dict de listas.

    Lanza
    -----
    SystemExit
        Si el formato no es reconocido.
    """
    # ---- Detectar Formato A ----
    if "history" in history_data and isinstance(history_data["history"], dict):
        inner = history_data["history"]
        required_keys = ["train_loss", "train_acc", "val_loss", "val_acc"]

        # Verificar que las claves esenciales existen
        missing = [k for k in required_keys if k not in inner]
        if missing:
            print(
                f"[AVISO] Faltan claves en history.json: {missing}. "
                f"Las épocas faltantes se omitirán.",
                file=sys.stderr,
            )

        # Advertir si las listas tienen longitudes distintas (puede indicar
        # un historial corrupto de un run que falló a mitad)
        lengths = {k: len(inner[k]) for k in required_keys if k in inner}
        if len(set(lengths.values())) > 1:
            print(
                f"[AVISO] Las listas del historial tienen longitudes distintas: "
                f"{lengths}. Se procesarán solo las primeras {min(lengths.values())} "
                f"épocas (la mínima).",
                file=sys.stderr,
            )

        return history_data

    # ---- Detectar Formato B (legado) ----
    if "epochs" in history_data and isinstance(history_data["epochs"], list):
        epochs_list = history_data["epochs"]
        print(
            f"[INFO] Formato legado detectado (lista de épocas). "
            f"Convirtiendo a formato moderno...",
        )

        converted: dict[str, list] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
            "epoch_time_s": [],
        }

        for ep in epochs_list:
            converted["train_loss"].append(float(ep.get("train_loss", 0.0)))
            converted["train_acc"].append(float(ep.get("train_acc", 0.0)))
            converted["val_loss"].append(float(ep.get("val_loss", 0.0)))
            converted["val_acc"].append(float(ep.get("val_acc", 0.0)))
            converted["lr"].append(float(ep.get("lr", 0.0)))
            # train.py guarda "elapsed_sec"; el campo puede llamarse distinto
            converted["epoch_time_s"].append(
                float(ep.get("elapsed_sec", ep.get("epoch_time_s", 0.0)))
            )

        result = {**history_data, "history": converted}
        return result

    # ---- Formato desconocido ----
    print(
        "[ERROR] Formato de history.json no reconocido.\n"
        "  Se esperaba:\n"
        "    Formato A: {'history': {'train_loss': [...], ...}}\n"
        "    Formato B: {'epochs': [{'train_loss': ..., ...}, ...]}\n"
        f"  Claves encontradas en el JSON: {list(history_data.keys())}",
        file=sys.stderr,
    )
    sys.exit(1)


# ============================================================================
# Escritura a TensorBoard
# ============================================================================

def write_to_tensorboard(
    history_data: dict,
    *,
    output_dir: Path,
    run_name: str,
) -> None:
    """Escribe los escalares del historial en logs de TensorBoard.

    Itera cada época del historial y registra Loss/train, Loss/val,
    Accuracy/train, Accuracy/val, LR con los mismos nombres jerárquicos
    que usa TrainingLogger.log_epoch(), de modo que los logs son
    intercambiables con los generados en tiempo real.

    Parámetros
    ----------
    history_data : dict
        Historial normalizado (output de validate_and_normalize).
    output_dir : Path
        Directorio raíz donde crear los logs.
    run_name : str
        Nombre del run. Crea: output_dir / run_name / events.out.*
    """
    inner = history_data["history"]

    # Número de épocas disponibles: mínimo de las listas con datos
    keys_required = ["train_loss", "train_acc", "val_loss", "val_acc"]
    available_lengths = [len(inner[k]) for k in keys_required if k in inner]
    if not available_lengths:
        print("[ERROR] El historial no contiene datos de épocas.", file=sys.stderr)
        sys.exit(1)

    n_epochs = min(available_lengths)

    # LR es opcional (algunos historiales antiguos no lo registraban)
    lr_list: list[float] = inner.get("lr", [0.0] * n_epochs)

    log_path = output_dir / run_name
    log_path.mkdir(parents=True, exist_ok=True)

    print(f"\nEscribiendo {n_epochs} épocas en: {log_path}")

    writer = SummaryWriter(log_dir=str(log_path))
    try:
        for epoch_idx in range(n_epochs):
            epoch = epoch_idx + 1  # 1-indexed, igual que el training loop

            train_loss = float(inner["train_loss"][epoch_idx])
            train_acc  = float(inner["train_acc"][epoch_idx])
            val_loss   = float(inner["val_loss"][epoch_idx])
            val_acc    = float(inner["val_acc"][epoch_idx])
            lr = float(lr_list[epoch_idx]) if epoch_idx < len(lr_list) else 0.0

            # Nombres idénticos a TrainingLogger.log_epoch()
            writer.add_scalar("Loss/train",     train_loss, epoch)
            writer.add_scalar("Loss/val",       val_loss,   epoch)
            writer.add_scalar("Accuracy/train", train_acc,  epoch)
            writer.add_scalar("Accuracy/val",   val_acc,    epoch)
            writer.add_scalar("LR",             lr,         epoch)

            # Progreso cada 10 épocas o en la primera
            if epoch_idx == 0 or (epoch_idx + 1) % 10 == 0:
                print(
                    f"  Época {epoch:4d}/{n_epochs} | "
                    f"train_loss={train_loss:.4f}  val_acc={val_acc:.4f}"
                )
    finally:
        writer.close()

    print(f"\n[OK] Logs escritos en: {log_path}")
    print("\n" + "─" * 50)
    print("Para visualizar los logs:")
    print(f"  tensorboard --logdir {output_dir}")
    print("\nEn Google Colab:")
    print("  %load_ext tensorboard")
    print(f"  %tensorboard --logdir {output_dir}")
    print("─" * 50)


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Define y parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Convierte un history.json de SelvaSonic en logs de TensorBoard. "
            "Permite recuperar experimentos anteriores donde no se usó TensorBoard."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--history-json",
        type=str,
        required=True,
        metavar="PATH",
        help="Ruta al archivo history.json generado por el entrenamiento.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        metavar="DIR",
        help="Directorio donde escribir los logs de TensorBoard.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Nombre del run en TensorBoard. "
            "Si no se provee, se deriva del path del history.json "
            "(nombre del directorio padre)."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Punto de entrada del script."""
    args = parse_args()

    history_path = Path(args.history_json)
    output_dir   = Path(args.output_dir)

    # Derivar run_name desde el path si no se especificó
    if args.run_name is not None:
        run_name = args.run_name
    else:
        # Preferir el nombre del directorio padre del history.json
        # (ej. "results/run_S3_v1/history.json" → "run_S3_v1")
        parent_name = history_path.parent.name
        run_name = parent_name if parent_name else history_path.stem

    print("=" * 60)
    print(" SelvaSonic — history.json → TensorBoard")
    print("=" * 60)
    print(f"  history-json : {history_path}")
    print(f"  output-dir   : {output_dir}")
    print(f"  run-name     : {run_name}")
    print("=" * 60)

    # Cargar y normalizar
    raw_data     = load_history(history_path)
    history_data = validate_and_normalize(raw_data)

    # Imprimir resumen del historial
    inner    = history_data.get("history", {})
    n_epochs = len(inner.get("train_loss", []))
    print(f"\n[OK] Historial cargado:")
    print(f"  Épocas disponibles : {n_epochs}")

    if "best_val_acc" in history_data:
        print(f"  best_val_acc       : {history_data['best_val_acc']:.4f}")
    if "best_val_loss" in history_data:
        print(f"  best_val_loss      : {history_data['best_val_loss']:.4f}")
    if "run_name" in history_data:
        print(f"  run_name original  : {history_data['run_name']}")

    # Escribir logs de TensorBoard
    write_to_tensorboard(
        history_data,
        output_dir=output_dir,
        run_name=run_name,
    )


if __name__ == "__main__":
    main()
