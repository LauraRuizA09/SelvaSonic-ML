"""
check_dataset.py
─────────────────────────────────────────────────────────────────────────────
Script de verificación rápida: comprueba que el Dataset y DataLoader de
Semana 2 devuelven datos en el formato exacto que el training loop de
Semana 3 espera.

Formato esperado:
  - DataLoader devuelve tuplas (inputs, targets)
  - inputs: torch.Tensor de shape (batch_size, 1, 128, 216), dtype float32
  - targets: torch.Tensor de shape (batch_size,), dtype torch.long (int64)

Si todas las verificaciones pasan → puedes proceder al entrenamiento real.
Si algo falla → te dice exactamente qué hay que arreglar.

Cómo correr:
    python check_dataset.py
"""

import torch
from src.dataset import create_dataloaders


def main() -> None:
    print("=" * 70)
    print(" VERIFICACIÓN DEL DATASET — compatibilidad con train.py")
    print("=" * 70)

    # 1. Crear los DataLoaders usando la función de tu Semana 2.
    # Ajusta el path a 'data/raw' si tu función lo necesita.
    print("\n[1/5] Creando DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir="data/raw",
        batch_size=4,  # usamos un batch chico para verificación rápida
    )
    print(f"      ✓ train_loader creado ({len(train_loader)} batches)")
    print(f"      ✓ val_loader creado   ({len(val_loader)} batches)")
    print(f"      ✓ test_loader creado  ({len(test_loader)} batches)")

    # 2. Sacar UN solo batch del train_loader
    print("\n[2/5] Extrayendo el primer batch de train_loader...")
    batch = next(iter(train_loader))

    # 3. Verificar que es una tupla de 2 elementos
    print("\n[3/5] Verificando estructura del batch...")
    assert isinstance(batch, (tuple, list)), (
        f"❌ El batch debería ser tupla o lista, pero es {type(batch).__name__}. "
        f"Si tu Dataset devuelve un dict, hay que ajustarlo."
    )
    assert len(batch) == 2, (
        f"❌ El batch debería tener 2 elementos (inputs, targets), "
        f"pero tiene {len(batch)}."
    )
    inputs, targets = batch
    print(f"      ✓ Batch es tupla de 2 elementos")

    # 4. Verificar el TENSOR DE INPUTS (espectrogramas)
    print("\n[4/5] Verificando 'inputs' (espectrogramas)...")
    assert isinstance(inputs, torch.Tensor), (
        f"❌ inputs debería ser torch.Tensor, pero es {type(inputs).__name__}"
    )
    print(f"      Tipo:   {type(inputs).__name__}")
    print(f"      Shape:  {tuple(inputs.shape)}")
    print(f"      Dtype:  {inputs.dtype}")
    print(f"      Min:    {inputs.min().item():.4f}")
    print(f"      Max:    {inputs.max().item():.4f}")
    print(f"      Mean:   {inputs.mean().item():.4f}")

    # Verificar shape: (B, 1, 128, 216) — 4 dimensiones
    assert inputs.ndim == 4, (
        f"❌ inputs debería tener 4 dimensiones (B, C, F, T), "
        f"pero tiene {inputs.ndim}."
    )
    assert inputs.shape[1] == 1, (
        f"❌ inputs debería tener 1 canal (mono), pero tiene {inputs.shape[1]}"
    )
    assert inputs.shape[2] == 128, (
        f"❌ inputs debería tener 128 bandas Mel, pero tiene {inputs.shape[2]}"
    )
    # Dtype debe ser float32 (no float64): es lo que PyTorch usa por default
    # y lo que las capas Conv2d esperan.
    assert inputs.dtype == torch.float32, (
        f"❌ inputs debería ser float32, pero es {inputs.dtype}. "
        f"Convierte con .float() en tu Dataset."
    )
    print(f"      ✓ inputs OK: shape (B, 1, 128, T), dtype float32")

    # 5. Verificar el TENSOR DE TARGETS (etiquetas de clase)
    print("\n[5/5] Verificando 'targets' (etiquetas de clase)...")
    assert isinstance(targets, torch.Tensor), (
        f"❌ targets debería ser torch.Tensor, pero es {type(targets).__name__}. "
        f"Si tu Dataset devuelve strings, hay que mapearlos a enteros."
    )
    print(f"      Tipo:    {type(targets).__name__}")
    print(f"      Shape:   {tuple(targets.shape)}")
    print(f"      Dtype:   {targets.dtype}")
    print(f"      Valores: {targets.tolist()}")
    print(f"      Únicos:  {sorted(set(targets.tolist()))}")

    # Targets deben ser LONG (int64): es lo que CrossEntropyLoss exige.
    # Si vienen como int32 o float, hay que castear.
    assert targets.dtype == torch.long, (
        f"❌ targets debería ser torch.long (int64), pero es {targets.dtype}. "
        f"En tu Dataset, asegúrate de hacer torch.tensor(label, dtype=torch.long)."
    )
    # Shape debe ser 1D: (batch_size,) — un entero por muestra.
    assert targets.ndim == 1, (
        f"❌ targets debería ser 1D (B,), pero tiene {targets.ndim} dims. "
        f"NO debe ser one-hot encoded — CrossEntropyLoss espera el índice."
    )
    # Que los labels estén en [0, num_classes)
    assert (targets >= 0).all(), "❌ Hay targets negativos"
    print(f"      ✓ targets OK: shape (B,), dtype int64, valores no-negativos")

    # 6. Test final: pasar el batch al modelo y verificar que NO crashea
    print("\n[BONUS] Probando que el batch funciona con el modelo...")
    from src.model import SelvaSonicCNN

    # Asumimos num_classes = max(target) + 1 como referencia rápida,
    # pero el número real lo sabes tú (probablemente 11 = 10 aves + no_ave).
    num_classes = max(targets.max().item() + 1, 11)
    model = SelvaSonicCNN(num_classes=num_classes)
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
    print(f"      ✓ Forward pass exitoso")
    print(f"      ✓ Output logits shape: {tuple(logits.shape)}")

    print("\n" + "=" * 70)
    print(" ✓ TODAS LAS VERIFICACIONES PASARON")
    print(" Tu Dataset es compatible con train.py — puedes entrenar.")
    print("=" * 70)


if __name__ == "__main__":
    main()
