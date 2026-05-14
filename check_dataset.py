"""
check_dataset.py
─────────────────────────────────────────────────────────────────────────────
Script de verificación: comprueba que el Dataset y DataLoader de Semana 2
devuelven datos en el formato que el training loop de Semana 3 espera.

Formato esperado:
  - DataLoader devuelve tuplas (inputs, targets)
  - inputs: torch.Tensor shape (B, 1, 128, 216), dtype float32
  - targets: torch.Tensor shape (B,), dtype torch.long (int64)

Cómo correr:
    python check_dataset.py

Si tu create_dataloaders() usa nombres de parámetros distintos a lo que asume
este script, el inicio del script imprime la FIRMA REAL para que ajustes los
argumentos.
"""

import inspect
import sys
import torch


def descubrir_firma() -> None:
    """Inspecciona create_dataloaders y SelvaSonicDataset para ver qué
    parámetros aceptan. Útil para depurar problemas de TypeError."""
    print("=" * 70)
    print(" PASO 0: Descubriendo la firma de tus funciones de Semana 2")
    print("=" * 70)

    try:
        from src.dataset import create_dataloaders
        sig = inspect.signature(create_dataloaders)
        print(f"\ncreate_dataloaders{sig}")
        print("\nParametros aceptados:")
        for name, param in sig.parameters.items():
            default = "" if param.default is inspect.Parameter.empty else f" = {param.default!r}"
            print(f"  - {name}{default}")
    except ImportError as e:
        print(f"[X] No se pudo importar create_dataloaders: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[X] Error inspeccionando create_dataloaders: {e}")

    try:
        from src.dataset import SelvaSonicDataset
        sig = inspect.signature(SelvaSonicDataset.__init__)
        print(f"\nSelvaSonicDataset.__init__{sig}")
    except ImportError:
        print("\n(SelvaSonicDataset no encontrada con ese nombre, no es critico)")
    except Exception as e:
        print(f"(No se pudo inspeccionar SelvaSonicDataset: {e})")

    print()


def main() -> None:
    # Paso 0: descubrir qué parámetros acepta la función ANTES de llamarla
    descubrir_firma()

    print("=" * 70)
    print(" VERIFICACION DEL DATASET - compatibilidad con train.py")
    print("=" * 70)

    from src.dataset import create_dataloaders

    # Inspeccionar la firma para llamar a create_dataloaders con los nombres
    # correctos. Probamos varios alias comunes que pudimos haber usado en
    # Semana 2: 'raw_data_dir', 'data_dir', 'root_dir', 'data_root'.
    sig = inspect.signature(create_dataloaders)
    param_names = list(sig.parameters.keys())

    kwargs = {}
    posibles_data = ["raw_data_dir", "data_dir", "root_dir", "data_root", "raw_dir"]
    posibles_batch = ["batch_size", "bs"]

    nombre_data = next((n for n in posibles_data if n in param_names), None)
    nombre_batch = next((n for n in posibles_batch if n in param_names), None)

    if nombre_data is None:
        print(
            "[X] No encontre ningun parametro reconocible para el directorio "
            f"de datos. Parametros disponibles: {param_names}.\n"
            "Edita check_dataset.py y agrega el nombre correcto a "
            "posibles_data."
        )
        sys.exit(1)

    kwargs[nombre_data] = "data/raw"
    if nombre_batch is not None:
        kwargs[nombre_batch] = 4

    print(f"[1/5] Llamando create_dataloaders con {kwargs}...")

    # 1. Crear los DataLoaders
    result = create_dataloaders(**kwargs)

    # create_dataloaders puede devolver: (train, val, test) o
    # (train, val, test, label_map) o un dict. Manejamos varios casos.
    if isinstance(result, dict):
        train_loader = result.get("train") or result.get("train_loader")
        val_loader = result.get("val") or result.get("val_loader")
        test_loader = result.get("test") or result.get("test_loader")
    elif isinstance(result, tuple):
        if len(result) == 3:
            train_loader, val_loader, test_loader = result
        elif len(result) == 4:
            train_loader, val_loader, test_loader, _ = result
        else:
            print(f"[X] create_dataloaders devolvio {len(result)} valores, no se que hacer.")
            sys.exit(1)
    else:
        print(f"[X] Tipo de retorno inesperado: {type(result).__name__}")
        sys.exit(1)

    print(f"      OK train_loader creado ({len(train_loader)} batches)")
    print(f"      OK val_loader creado   ({len(val_loader)} batches)")
    print(f"      OK test_loader creado  ({len(test_loader)} batches)")

    # 2. Sacar UN batch del train_loader
    print("\n[2/5] Extrayendo el primer batch de train_loader...")
    batch = next(iter(train_loader))

    # 3. Verificar estructura del batch
    print("\n[3/5] Verificando estructura del batch...")
    assert isinstance(batch, (tuple, list)), (
        f"[X] El batch deberia ser tupla o lista, pero es {type(batch).__name__}. "
        f"Si tu Dataset devuelve un dict, hay que ajustarlo."
    )
    assert len(batch) == 2, (
        f"[X] El batch deberia tener 2 elementos (inputs, targets), "
        f"pero tiene {len(batch)}."
    )
    inputs, targets = batch
    print(f"      OK batch es tupla de 2 elementos")

    # 4. Verificar el tensor de INPUTS
    print("\n[4/5] Verificando 'inputs' (espectrogramas)...")
    assert isinstance(inputs, torch.Tensor), (
        f"[X] inputs deberia ser torch.Tensor, pero es {type(inputs).__name__}"
    )
    print(f"      Tipo:   {type(inputs).__name__}")
    print(f"      Shape:  {tuple(inputs.shape)}")
    print(f"      Dtype:  {inputs.dtype}")
    print(f"      Min:    {inputs.min().item():.4f}")
    print(f"      Max:    {inputs.max().item():.4f}")
    print(f"      Mean:   {inputs.mean().item():.4f}")

    assert inputs.ndim == 4, (
        f"[X] inputs deberia tener 4 dimensiones (B, C, F, T), pero tiene {inputs.ndim}."
    )
    assert inputs.shape[1] == 1, (
        f"[X] inputs deberia tener 1 canal (mono), pero tiene {inputs.shape[1]}"
    )
    assert inputs.shape[2] == 128, (
        f"[X] inputs deberia tener 128 bandas Mel, pero tiene {inputs.shape[2]}"
    )
    assert inputs.dtype == torch.float32, (
        f"[X] inputs deberia ser float32, pero es {inputs.dtype}."
    )
    print(f"      OK inputs OK: shape (B, 1, 128, T), dtype float32")

    # 5. Verificar el tensor de TARGETS
    print("\n[5/5] Verificando 'targets' (etiquetas de clase)...")
    assert isinstance(targets, torch.Tensor), (
        f"[X] targets deberia ser torch.Tensor, pero es {type(targets).__name__}."
    )
    print(f"      Tipo:    {type(targets).__name__}")
    print(f"      Shape:   {tuple(targets.shape)}")
    print(f"      Dtype:   {targets.dtype}")
    print(f"      Valores: {targets.tolist()}")
    print(f"      Unicos:  {sorted(set(targets.tolist()))}")

    assert targets.dtype == torch.long, (
        f"[X] targets deberia ser torch.long (int64), pero es {targets.dtype}.\n"
        f"   En tu Dataset, asegurate de hacer torch.tensor(label, dtype=torch.long)."
    )
    assert targets.ndim == 1, (
        f"[X] targets deberia ser 1D (B,), pero tiene {targets.ndim} dims.\n"
        f"   NO debe ser one-hot encoded - CrossEntropyLoss espera el indice."
    )
    assert (targets >= 0).all(), "[X] Hay targets negativos"
    print(f"      OK targets OK: shape (B,), dtype int64, valores no-negativos")

    # 6. Bonus: probar batch con el modelo
    print("\n[BONUS] Probando que el batch funciona con el modelo...")
    from src.model import SelvaSonicCNN

    num_classes = max(targets.max().item() + 1, 11)
    model = SelvaSonicCNN(num_classes=num_classes)
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
    print(f"      OK forward pass exitoso")
    print(f"      OK output logits shape: {tuple(logits.shape)}")

    print("\n" + "=" * 70)
    print(" OK TODAS LAS VERIFICACIONES PASARON")
    print(" Tu Dataset es compatible con train.py - puedes entrenar.")
    print("=" * 70)


if __name__ == "__main__":
    main()
