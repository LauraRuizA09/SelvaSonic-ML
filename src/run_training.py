"""
src/run_training.py
─────────────────────────────────────────────────────────────────────────────
Script principal de entrenamiento: conecta el pipeline de datos de Semana 2
con el training loop de Semana 3.

Este script es solo "pegamento" entre módulos existentes — toda la lógica
real vive en src/dataset.py (datos) y src/train.py (entrenamiento).

Lo que hace, paso a paso:
    1. Lee config.yaml (hiperparámetros centralizados).
    2. Llama a create_dataloaders() para obtener train/val/test loaders.
    3. Instancia SelvaSonicCNN(num_classes=11).
    4. Llama a train() pasando los DataLoaders y los hiperparámetros.
    5. Al final imprime un resumen y guarda el historial JSON.

Para correr en VS Code (Windows + venv):
    .\venv\Scripts\Activate.ps1
    python -m src.run_training

Para correr con flags personalizados (sobreescribir config):
    python -m src.run_training --epochs 5 --batch_size 16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

# Imports relativos: estos módulos están en el mismo paquete src/
# Compatible con `python -m src.run_training` y `python src/run_training.py`.
try:
    from src.dataset import create_dataloaders
    from src.model import SelvaSonicCNN
    from src.train import train, get_device
except ImportError:
    from dataset import create_dataloaders
    from model import SelvaSonicCNN
    from train import train, get_device


def load_config(config_path: str | Path = "config.yaml") -> dict:
    """Lee el archivo de configuración YAML.

    Args:
        config_path: ruta al archivo config.yaml (default: raíz del proyecto).

    Returns:
        Diccionario con la configuración parseada.

    Raises:
        FileNotFoundError: si el config.yaml no existe.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"No se encontró {config_path}. Verifica que estés en la raíz "
            f"del proyecto SelvaSonic-ML."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos.

    Todos son opcionales: si no se pasan, se usan los valores de config.yaml.
    """
    parser = argparse.ArgumentParser(
        description="Entrenar SelvaSonicCNN baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Ruta al archivo config.yaml",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Sobreescribe el número de épocas del config",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Sobreescribe el batch size del config",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=None,
        help="Sobreescribe el learning rate del config",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/raw",
        help="Carpeta con los datos de audio",
    )
    return parser.parse_args()


def main() -> None:
    # ===================================================================
    # 1. Setup: argumentos y configuración
    # ===================================================================
    args = parse_args()
    config = load_config(args.config)

    # Permitir override desde CLI (útil para experimentar rápido)
    epochs = args.epochs if args.epochs is not None else config["training"]["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else config["training"]["batch_size"]
    learning_rate = args.learning_rate if args.learning_rate is not None else config["training"]["learning_rate"]

    weight_decay = config["training"]["weight_decay"]
    label_smoothing = config["training"]["label_smoothing"]
    early_stopping_patience = config["training"]["early_stopping_patience"]
    seed = config["training"]["seed"]

    # Banner inicial: deja claro qué se va a entrenar
    print("\n" + "#" * 70)
    print(" SELVASONIC - Entrenamiento Baseline (Semana 3)")
    print("#" * 70)
    print(f" Config:          {args.config}")
    print(f" Datos:           {args.data_dir}")
    print(f" Epochs:          {epochs}")
    print(f" Batch size:      {batch_size}")
    print(f" Learning rate:   {learning_rate}")
    print(f" Weight decay:    {weight_decay}")
    print(f" Label smoothing: {label_smoothing}")
    print(f" Seed:            {seed}")
    print("#" * 70 + "\n")

    # ===================================================================
    # 2. Cargar los DataLoaders (orquestados por src/dataset.py)
    # ===================================================================
    print("[1/3] Construyendo DataLoaders...")
    train_loader, val_loader, test_loader, label_map = create_dataloaders(
        raw_data_dir=args.data_dir,
        batch_size=batch_size,
        num_workers=0,  # Windows: usar 0 para evitar problemas de multiprocessing
        train_ratio=config["data"]["split_ratios"]["train"],
        val_ratio=config["data"]["split_ratios"]["val"],
        test_ratio=config["data"]["split_ratios"]["test"],
        random_state=seed,
        verbose=True,
    )

    num_classes = len(label_map)
    print(f"\n      OK {num_classes} clases detectadas: {list(label_map.keys())}")
    print(f"      OK Batches: train={len(train_loader)} val={len(val_loader)} test={len(test_loader)}")

    # ===================================================================
    # 3. Instanciar el modelo
    # ===================================================================
    print("\n[2/3] Instanciando modelo SelvaSonicCNN...")
    model = SelvaSonicCNN(
        num_classes=num_classes,
        in_channels=1,
        cnn_channels=config["model"]["cnn_channels"],
        dropout=config["model"]["dropout"],
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      OK Modelo creado: {n_params:,} parametros entrenables")
    print(f"      OK Tamaño en memoria: ~{n_params * 4 / (1024 * 1024):.2f} MB (float32)")

    # ===================================================================
    # 4. Entrenamiento
    # ===================================================================
    print("\n[3/3] Iniciando entrenamiento...\n")
    device = get_device()

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        early_stopping_patience=early_stopping_patience,
        seed=seed,
        checkpoint_dir="checkpoints",
        log_dir="results/logs",
        # NO pasamos class_weights: queremos baseline puro para Semana 3
        class_weights=None,
        device=device,
        verbose=True,
    )

    # ===================================================================
    # 5. Resumen final
    # ===================================================================
    print("\n" + "#" * 70)
    print(" RESUMEN DEL ENTRENAMIENTO")
    print("#" * 70)
    print(f" Tiempo total:     {history.total_time_sec:.1f}s "
          f"({history.total_time_sec / 60:.1f} min)")
    print(f" Epocas corridas:  {len(history.epochs)} / {epochs}")
    if history.stopped_early:
        print(f" Early stopping:   SI (no mejoraba)")
    print(f" Mejor epoca:      {history.best_epoch}")
    print(f" Mejor val_loss:   {history.best_val_loss:.4f}")
    print(f" Mejor val_acc:    {history.best_val_acc:.4f}")
    print(f" Checkpoint:       checkpoints/best_model.pth")
    print(f" Historial:        results/logs/training_history.json")
    print("#" * 70)

    # Aviso amistoso sobre la accuracy
    if history.best_val_acc > 0.75:
        print("\n[AVISO] Accuracy alta (>75%) puede deberse al desbalance.")
        print("        En Semana 4 con class_weights veremos si realmente")
        print("        el modelo aprende todas las clases o solo predice no_ave.")
        print("        Para confirmarlo, analiza la matriz de confusion")
        print("        (tarea 6 de Semana 3).\n")


if __name__ == "__main__":
    main()
