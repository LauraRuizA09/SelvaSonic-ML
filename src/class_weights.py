"""
src/class_weights.py
====================

Cálculo de pesos por clase para mitigar el desbalance en la función de pérdida.

Este módulo implementa la estrategia de class weighting inversamente proporcional
a la frecuencia de cada clase, una técnica estándar para problemas de clasificación
con datasets desbalanceados.

Fundamento teórico
------------------
Dado un dataset con N muestras totales distribuidas en K clases, donde la clase c
tiene n_c muestras, el peso w_c se calcula como:

    w_c = N / (K * n_c)

Esta normalización garantiza que la suma ponderada de los pesos por la frecuencia
relativa de cada clase sea igual a 1:

    sum_c (n_c / N) * w_c = sum_c (n_c / N) * (N / (K * n_c))
                          = sum_c (1 / K)
                          = K * (1/K) = 1

Por lo tanto, los pesos preservan la escala de la pérdida y solo modifican la
importancia relativa entre clases.

Referencias
-----------
- He, H., & Garcia, E. A. (2009). Learning from Imbalanced Data.
  IEEE Transactions on Knowledge and Data Engineering.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Estructuras de datos
# ---------------------------------------------------------------------------

class ClassWeightsResult(NamedTuple):
    """Resultado del cálculo de class weights.

    Attributes
    ----------
    weights : torch.Tensor
        Tensor 1-D de tamaño K con el peso de cada clase. Tipo float32.
    class_counts : np.ndarray
        Array 1-D de tamaño K con el número de muestras de cada clase.
    class_frequencies : np.ndarray
        Array 1-D de tamaño K con la frecuencia relativa (n_c / N) de cada clase.
    weighted_sum : float
        Suma ponderada sum_c (n_c / N) * w_c. Debe ser ~1.0 por construcción.
        Sirve como verificación de la propiedad de normalización.
    """
    weights: torch.Tensor
    class_counts: np.ndarray
    class_frequencies: np.ndarray
    weighted_sum: float


# ---------------------------------------------------------------------------
# Cálculo de class weights
# ---------------------------------------------------------------------------

def compute_class_weights(
    *,
    labels: Sequence[int],
    num_classes: int,
    dtype: torch.dtype = torch.float32,
) -> ClassWeightsResult:
    """Calcula class weights inversamente proporcionales a la frecuencia.

    Aplica la fórmula w_c = N / (K * n_c), donde N es el total de muestras,
    K el número de clases y n_c el número de muestras de la clase c.

    Parameters
    ----------
    labels : Sequence[int]
        Secuencia de labels enteros (uno por muestra) del conjunto de entrenamiento.
        IMPORTANTE: solo se debe usar el split de train; usar val o test produce
        data leakage.
    num_classes : int
        Número total de clases K. Debe ser >= max(labels) + 1.
    dtype : torch.dtype, opcional
        Tipo de dato del tensor de pesos resultante. Por defecto float32.

    Returns
    -------
    ClassWeightsResult
        NamedTuple con los pesos, conteos, frecuencias y suma ponderada de
        verificación.

    Raises
    ------
    ValueError
        Si `labels` está vacío, si `num_classes` es no positivo, o si alguna
        clase no tiene muestras (lo cual produciría división por cero).

    Examples
    --------
    >>> result = compute_class_weights(labels=[0, 0, 0, 1, 2, 2], num_classes=3)
    >>> result.class_counts
    array([3, 1, 2])
    >>> result.weights
    tensor([0.6667, 2.0000, 1.0000])
    >>> abs(result.weighted_sum - 1.0) < 1e-6
    True
    """
    # --- Validación de entrada ---
    if len(labels) == 0:
        raise ValueError("`labels` no puede estar vacío.")
    if num_classes <= 0:
        raise ValueError(f"`num_classes` debe ser > 0, recibido: {num_classes}")

    labels_array = np.asarray(labels, dtype=np.int64)

    if labels_array.max() >= num_classes:
        raise ValueError(
            f"Encontrado label {labels_array.max()} pero num_classes={num_classes}. "
            f"Los labels deben estar en el rango [0, num_classes-1]."
        )
    if labels_array.min() < 0:
        raise ValueError(
            f"Encontrado label negativo {labels_array.min()}. "
            f"Los labels deben ser enteros no negativos."
        )

    # --- Conteo por clase ---
    # np.bincount asegura que tenemos exactamente num_classes posiciones,
    # incluso si alguna clase no aparece (queda en 0).
    class_counts = np.bincount(labels_array, minlength=num_classes)

    if (class_counts == 0).any():
        clases_vacias = np.where(class_counts == 0)[0].tolist()
        raise ValueError(
            f"Las siguientes clases no tienen muestras en el dataset: {clases_vacias}. "
            f"No se pueden calcular pesos sin división por cero."
        )

    # --- Cálculo del peso por clase ---
    # Fórmula: w_c = N / (K * n_c)
    N = labels_array.shape[0]
    K = num_classes
    weights_np = N / (K * class_counts.astype(np.float64))

    # --- Frecuencias relativas (para análisis e informes) ---
    class_frequencies = class_counts.astype(np.float64) / N

    # --- Verificación de la propiedad de normalización ---
    # Por construcción matemática, sum_c (n_c/N) * w_c debe ser exactamente 1.
    weighted_sum = float(np.sum(class_frequencies * weights_np))

    # --- Conversión a torch tensor ---
    weights_tensor = torch.tensor(weights_np, dtype=dtype)

    return ClassWeightsResult(
        weights=weights_tensor,
        class_counts=class_counts,
        class_frequencies=class_frequencies,
        weighted_sum=weighted_sum,
    )


# ---------------------------------------------------------------------------
# Verificación matemática (útil para tests)
# ---------------------------------------------------------------------------

def verify_class_weights(
    *,
    result: ClassWeightsResult,
    tolerance: float = 1e-6,
) -> bool:
    """Verifica las propiedades matemáticas de los class weights.

    Comprueba dos propiedades clave:

    1. **Normalización:** sum_c (n_c / N) * w_c == 1.0
       Garantiza que la escala de la pérdida ponderada es la misma que la
       pérdida sin pesos en promedio.

    2. **Monotonicidad inversa:** clases con más muestras tienen pesos menores.
       Si n_a > n_b entonces w_a < w_b.

    Parameters
    ----------
    result : ClassWeightsResult
        Resultado de `compute_class_weights`.
    tolerance : float, opcional
        Tolerancia numérica para la verificación de normalización.

    Returns
    -------
    bool
        True si todas las propiedades se cumplen.
    """
    # Propiedad 1: normalización
    if abs(result.weighted_sum - 1.0) > tolerance:
        return False

    # Propiedad 2: monotonicidad inversa
    # Ordenamos los conteos de mayor a menor; los pesos deben quedar de menor a mayor.
    sorted_indices = np.argsort(result.class_counts)[::-1]  # descendente por count
    weights_in_order = result.weights.numpy()[sorted_indices]

    # Verificamos que los pesos quedan en orden ascendente
    for i in range(len(weights_in_order) - 1):
        if weights_in_order[i] > weights_in_order[i + 1] + tolerance:
            return False

    return True


# ---------------------------------------------------------------------------
# Helper: extracción de labels desde un torch Dataset
# ---------------------------------------------------------------------------

def extract_labels_from_dataset(dataset) -> list[int]:
    """Extrae todos los labels de un torch.utils.data.Dataset.

    Útil para calcular class weights sobre un Dataset sin cargar todos los
    espectrogramas en memoria. Asume que el dataset tiene un atributo `samples`
    o que cada item retorna una tupla `(features, label)`.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset del que extraer los labels. Idealmente debe exponer los labels
        sin tener que cargar las features (ver nota de eficiencia abajo).

    Returns
    -------
    list[int]
        Lista de labels en el mismo orden que el dataset.

    Notes
    -----
    Si el dataset es perezoso (carga features on-the-fly), llamar a `dataset[i]`
    para cada i puede ser lento porque carga el audio completo. Idealmente, el
    Dataset debería exponer un atributo `_index` o similar con los labels ya
    precomputados (como hace `AmazonAudioDataset` en este proyecto).

    Esta función intenta primero acceder a atributos comunes; si fallan, hace
    fallback a iteración completa.
    """
    # Intento 1: atributo `labels` (rápido si existe)
    if hasattr(dataset, "labels"):
        return list(dataset.labels)

    # Intento 2: atributo `_index` con label en cada entrada (común en proyectos
    # con índice de coordenadas como SelvaSonic)
    if hasattr(dataset, "_index"):
        try:
            return [entry.label for entry in dataset._index]
        except AttributeError:
            pass

    # Fallback: iteración completa (LENTO, carga features)
    # Solo usar como último recurso.
    return [int(dataset[i][1]) for i in range(len(dataset))]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print(" SMOKE TEST: src/class_weights.py")
    print("=" * 70)

    # --- Test 1: caso de juguete pequeño ---
    print("\n[Test 1] Caso de juguete pequeño")
    print("-" * 70)
    labels_toy = [0, 0, 0, 1, 2, 2]  # 3, 1, 2 muestras
    result = compute_class_weights(labels=labels_toy, num_classes=3)

    print(f"  Labels: {labels_toy}")
    print(f"  Counts: {result.class_counts.tolist()}")
    print(f"  Weights: {result.weights.tolist()}")
    print(f"  Weighted sum (debe ser ~1.0): {result.weighted_sum:.10f}")

    # Verificación manual: w_0 = 6/(3*3)=0.667, w_1=6/(3*1)=2.0, w_2=6/(3*2)=1.0
    expected = torch.tensor([0.6667, 2.0000, 1.0000])
    assert torch.allclose(result.weights, expected, atol=1e-4), \
        f"Pesos incorrectos: {result.weights} vs esperado {expected}"
    assert verify_class_weights(result=result), "Verificación de propiedades falló"
    print("  ✓ Pesos correctos")
    print("  ✓ Propiedad de normalización OK")
    print("  ✓ Monotonicidad inversa OK")

    # --- Test 2: simulación realista del dataset SelvaSonic ---
    print("\n[Test 2] Simulación realista de SelvaSonic (11 clases desbalanceadas)")
    print("-" * 70)

    # Aproximación a la distribución real conocida del proyecto
    counts_real = {
        0: 1600,  # no_ave (clase dominante)
        1: 200,   # Crypturellus_cinereus
        2: 120,   # Ramphastos_tucanus
        3: 110,   # Celeus_grammicus
        4: 90,    # Glaucidium_brasilianum
        5: 80,    # Chordeiles_pusillus
        6: 80,    # Rupornis_magnirostris
        7: 80,    # Frederickena_fulva
        8: 110,   # Crypturellus_undulatus
        9: 140,   # Lipaugus_vociferans
        10: 300,  # Trogon_viridis
    }
    labels_sim: list[int] = []
    for cls, n in counts_real.items():
        labels_sim.extend([cls] * n)

    result_sim = compute_class_weights(labels=labels_sim, num_classes=11)

    print(f"  Total de muestras (N): {sum(counts_real.values())}")
    print(f"  Número de clases (K): 11")
    print(f"  Weighted sum: {result_sim.weighted_sum:.10f}")

    print("\n  Tabla de pesos por clase:")
    print(f"  {'Clase':>6} | {'Muestras':>9} | {'Freq (%)':>9} | {'Peso':>8}")
    print(f"  {'-'*6} | {'-'*9} | {'-'*9} | {'-'*8}")
    for cls in range(11):
        print(
            f"  {cls:>6d} | "
            f"{result_sim.class_counts[cls]:>9d} | "
            f"{result_sim.class_frequencies[cls]*100:>8.2f}% | "
            f"{result_sim.weights[cls].item():>8.4f}"
        )

    assert verify_class_weights(result=result_sim), "Verificación falló en test realista"
    print("\n  ✓ Verificación matemática OK en dataset realista")

    # --- Test 3: manejo de errores ---
    print("\n[Test 3] Manejo de errores")
    print("-" * 70)

    # Labels vacíos
    try:
        compute_class_weights(labels=[], num_classes=3)
        print("  ✗ FALLO: debió rechazar labels vacíos")
    except ValueError as e:
        print(f"  ✓ Rechaza labels vacíos: {e}")

    # Clase sin muestras
    try:
        compute_class_weights(labels=[0, 0, 2, 2], num_classes=3)  # falta clase 1
        print("  ✗ FALLO: debió rechazar clase sin muestras")
    except ValueError as e:
        print(f"  ✓ Rechaza clase sin muestras: clase 1 vacía")

    # Label fuera de rango
    try:
        compute_class_weights(labels=[0, 1, 5], num_classes=3)
        print("  ✗ FALLO: debió rechazar label fuera de rango")
    except ValueError as e:
        print(f"  ✓ Rechaza label fuera de rango")

    print("\n" + "=" * 70)
    print(" TODOS LOS SMOKE TESTS PASARON ✓")
    print("=" * 70)
