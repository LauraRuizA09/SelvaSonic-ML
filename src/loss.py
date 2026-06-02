"""
src/loss.py
===========

Construcción de la función de pérdida para SelvaSonic-ML.

Centraliza la lógica de class weights y label smoothing en una sola función
para que el notebook de entrenamiento no duplique lógica condicional frágil.

Por qué un módulo separado:
    La configuración de la loss implica validaciones no triviales (verificar
    que el archivo existe, que el tensor tiene el shape correcto, que está en
    el device correcto). Meter esto inline en el notebook mezcla lógica de
    setup con el flujo de entrenamiento. Un módulo limpio es testeable y
    reutilizable en futuros notebooks de entrenamiento.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def build_loss(
    *,
    use_class_weights: bool,
    label_smoothing: float,
    class_weights_path: Path | None,
    num_classes: int,
    device: torch.device,
) -> tuple[nn.CrossEntropyLoss, dict[str, Any]]:
    """Construye CrossEntropyLoss con class weights y/o label smoothing.

    Centraliza las validaciones y la lógica condicional para que el notebook
    de entrenamiento solo llame a esta función con los parámetros del config.

    Parameters
    ----------
    use_class_weights : bool
        Si True, carga el tensor de pesos desde class_weights_path y lo pasa
        como argumento `weight` a CrossEntropyLoss.
    label_smoothing : float
        Factor alpha de label smoothing en [0.0, 1.0). 0.0 = sin smoothing.
        Se pasa directamente a CrossEntropyLoss.
    class_weights_path : Path | None
        Ruta al archivo .pt generado por scripts/calcular_class_weights.py.
        Requerido si use_class_weights=True; ignorado si False.
    num_classes : int
        Número de clases del modelo. Se usa para validar el shape del tensor
        de pesos (debe ser [num_classes]).
    device : torch.device
        Device al que se mueve el tensor de pesos (CPU o CUDA). Debe coincidir
        con el device del modelo y de los batches de entrenamiento.

    Returns
    -------
    loss_fn : nn.CrossEntropyLoss
        Función de pérdida configurada y lista para usar en el training loop.
        Los pesos ya están en el device correcto.
    info : dict[str, Any]
        Diccionario con metadata para logging:
            - use_class_weights : bool
            - label_smoothing   : float
            - weights_min       : float | None (mínimo peso si use_class_weights)
            - weights_max       : float | None (máximo peso si use_class_weights)
            - weights_vector    : list[float] | None (vector completo de pesos)

    Raises
    ------
    FileNotFoundError
        Si use_class_weights=True y class_weights_path no existe. El mensaje
        incluye el comando exacto para generar el archivo.
    ValueError
        Si el tensor cargado no tiene shape [num_classes].

    Examples
    --------
    >>> import torch
    >>> from pathlib import Path
    >>> loss_fn, info = build_loss(
    ...     use_class_weights=False,
    ...     label_smoothing=0.1,
    ...     class_weights_path=None,
    ...     num_classes=11,
    ...     device=torch.device("cpu"),
    ... )
    >>> logits = torch.randn(4, 11)
    >>> targets = torch.tensor([0, 3, 7, 10])
    >>> loss = loss_fn(logits, targets)
    >>> loss.item() > 0
    True
    """
    weight_tensor: torch.Tensor | None = None
    weights_list: list[float] | None = None

    if use_class_weights:
        if class_weights_path is None:
            raise ValueError(
                "use_class_weights=True pero class_weights_path es None. "
                "Pasa la ruta al archivo .pt o desactiva use_class_weights."
            )

        path = Path(class_weights_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Tensor de class weights no encontrado en:\n"
                f"  {path}\n\n"
                f"Genera el archivo ejecutando:\n"
                f"  python scripts/calcular_class_weights.py\n\n"
                f"En Colab: sube data/class_weights.pt a Drive y copíalo al "
                f"proyecto (la celda de verificación lo hace automáticamente)."
            )

        payload = torch.load(str(path), weights_only=False, map_location="cpu")

        # El payload guarda los pesos bajo la clave "weights"
        if "weights" not in payload:
            raise ValueError(
                f"El archivo {path} no contiene la clave 'weights'. "
                f"Claves encontradas: {list(payload.keys())}. "
                f"Regenera el archivo con scripts/calcular_class_weights.py."
            )

        weight_tensor = payload["weights"].float()

        expected_shape = (num_classes,)
        if tuple(weight_tensor.shape) != expected_shape:
            raise ValueError(
                f"Shape del tensor de pesos incorrecto: {tuple(weight_tensor.shape)}. "
                f"Se esperaba {expected_shape} ({num_classes} clases). "
                f"Verifica que num_classes={num_classes} coincide con el dataset."
            )

        weight_tensor = weight_tensor.to(device)
        weights_list = weight_tensor.tolist()

    loss_fn = nn.CrossEntropyLoss(
        weight=weight_tensor,
        label_smoothing=label_smoothing,
    )

    info: dict[str, Any] = {
        "use_class_weights": use_class_weights,
        "label_smoothing": label_smoothing,
        "weights_min": float(weight_tensor.min()) if weight_tensor is not None else None,
        "weights_max": float(weight_tensor.max()) if weight_tensor is not None else None,
        "weights_vector": weights_list,
    }

    return loss_fn, info


# =============================================================================
# Smoke test — 4 casos: (with/without weights) × (with/without smoothing)
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Añadir raíz del proyecto al path para importar src.config
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from src.config import CLASS_WEIGHTS_PATH, LABEL_SMOOTHING_ALPHA, NUM_CLASSES

    print("=" * 70)
    print(" SMOKE TEST: src/loss.py")
    print("=" * 70)

    _device = torch.device("cpu")
    _num_classes = NUM_CLASSES  # 11
    _logits = torch.randn(4, _num_classes)
    _targets = torch.tensor([0, 3, 7, 10])

    # ---- Caso 1: sin pesos, sin smoothing ----
    print("\n[Test 1] Sin class weights, sin label smoothing")
    fn1, info1 = build_loss(
        use_class_weights=False,
        label_smoothing=0.0,
        class_weights_path=None,
        num_classes=_num_classes,
        device=_device,
    )
    loss1 = fn1(_logits, _targets)
    assert loss1.item() > 0, "Loss debe ser positiva"
    assert info1["use_class_weights"] is False
    assert info1["weights_min"] is None
    print(f"  Loss: {loss1.item():.4f}  [OK]")
    print(f"  Info: {info1}")

    # ---- Caso 2: sin pesos, con smoothing ----
    print("\n[Test 2] Sin class weights, con label smoothing=0.1")
    fn2, info2 = build_loss(
        use_class_weights=False,
        label_smoothing=LABEL_SMOOTHING_ALPHA,
        class_weights_path=None,
        num_classes=_num_classes,
        device=_device,
    )
    loss2 = fn2(_logits, _targets)
    assert loss2.item() > 0
    assert info2["label_smoothing"] == LABEL_SMOOTHING_ALPHA
    print(f"  Loss: {loss2.item():.4f}  [OK]")
    print(f"  Info: {info2}")

    # ---- Caso 3: con pesos + smoothing (si existe class_weights.pt) ----
    print(f"\n[Test 3] Con class weights ({CLASS_WEIGHTS_PATH.name}) + label smoothing")
    if CLASS_WEIGHTS_PATH.exists():
        fn3, info3 = build_loss(
            use_class_weights=True,
            label_smoothing=LABEL_SMOOTHING_ALPHA,
            class_weights_path=CLASS_WEIGHTS_PATH,
            num_classes=_num_classes,
            device=_device,
        )
        loss3 = fn3(_logits, _targets)
        assert loss3.item() > 0
        assert info3["use_class_weights"] is True
        assert info3["weights_min"] is not None
        assert len(info3["weights_vector"]) == _num_classes
        print(f"  Loss: {loss3.item():.4f}  [OK]")
        print(f"  weights_min={info3['weights_min']:.4f}, weights_max={info3['weights_max']:.4f}")
    else:
        print(f"  SALTEADO: {CLASS_WEIGHTS_PATH} no existe en local.")
        print(f"  Correr primero: python scripts/calcular_class_weights.py")

    # ---- Caso 4: errores esperados ----
    print("\n[Test 4] Manejo de errores")

    # Archivo inexistente
    try:
        build_loss(
            use_class_weights=True,
            label_smoothing=0.0,
            class_weights_path=Path("data/no_existe.pt"),
            num_classes=_num_classes,
            device=_device,
        )
        print("  [FAIL] Debio lanzar FileNotFoundError")
    except FileNotFoundError as e:
        print(f"  [OK] FileNotFoundError capturado correctamente")

    # Shape incorrecto
    import tempfile
    _bad_pt = Path(tempfile.mktemp(suffix=".pt"))
    torch.save({"weights": torch.ones(5)}, _bad_pt)  # shape (5,) != (11,)
    try:
        build_loss(
            use_class_weights=True,
            label_smoothing=0.0,
            class_weights_path=_bad_pt,
            num_classes=_num_classes,
            device=_device,
        )
        print("  [FAIL] Debio lanzar ValueError por shape incorrecto")
    except ValueError as e:
        print(f"  [OK] ValueError por shape incorrecto capturado")
    finally:
        _bad_pt.unlink(missing_ok=True)

    print("\n" + "=" * 70)
    print(" TODOS LOS SMOKE TESTS PASARON [OK]")
    print("=" * 70)
