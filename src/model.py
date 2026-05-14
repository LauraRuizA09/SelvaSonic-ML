"""
src/model.py
─────────────────────────────────────────────────────────────────────────────
Arquitectura del modelo SelvaSonic — CNN baseline para clasificación de
espectrogramas Mel de vocalizaciones de aves amazónicas.

Esta es la versión Semana 3: CNN-only (sin attention). En Semana 4 se le
agregará el módulo Multi-Head Self-Attention al final del feature extractor,
para poder comparar honestamente la contribución del attention.

Pipeline interno:
    Input (B, 1, 128, T) ──┐
                           ├─> Feature Extractor (4 bloques Conv2D)
                           │   Conv → BatchNorm → ReLU → MaxPool
                           │   Canales: 1 → 32 → 64 → 128 → 256
                           │   Espacial: (128, T) → (8, T/16)
                           │
                           ├─> Global Average Pooling (B, 256, 1, 1)
                           │
                           ├─> Classifier Head
                           │   Linear(256, 128) → ReLU → Dropout(0.3)
                           │   Linear(128, num_classes)
                           │
                           └─> Logits (B, num_classes)

Notas de diseño:
- Se usa BatchNorm DESPUÉS de Conv2D y ANTES de ReLU. Es el orden canónico
  (Ioffe & Szegedy 2015): la convolución produce activaciones potencialmente
  con cualquier escala, BatchNorm las normaliza, y ReLU las rectifica.
- El padding=1 en Conv 3x3 mantiene la dimensión espacial (only MaxPool
  reduce). Esto da control explícito sobre la reducción.
- Global Average Pooling > Flatten porque (a) menos parámetros en la FC,
  (b) invarianza a tamaño de entrada, (c) menos riesgo de overfitting.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


# ============================================================================
# Constantes del módulo (overrides de los valores de config.py si se quiere
# instanciar el modelo independientemente; en producción se pasarán desde
# config). Las dejamos como defaults documentados.
# ============================================================================

DEFAULT_CNN_CHANNELS: List[int] = [32, 64, 128, 256]
DEFAULT_DROPOUT: float = 0.3
DEFAULT_HIDDEN_FC_DIM: int = 128


# ============================================================================
# Bloque convolucional reutilizable
# ============================================================================

def _build_cnn_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Construye un bloque Conv2D + BatchNorm2D + ReLU + MaxPool2D.

    Patrón canónico moderno para CNNs sobre espectrogramas / imágenes.
    El kernel 3x3 con padding=1 preserva la dimensión espacial, y el MaxPool
    2x2 la reduce a la mitad.

    Args:
        in_channels: Número de canales de entrada al bloque.
        out_channels: Número de canales que producirá la convolución.

    Returns:
        Un nn.Sequential que recibe tensor (B, in_channels, H, W) y devuelve
        (B, out_channels, H/2, W/2).
    """
    return nn.Sequential(
        # Conv 3x3 con padding=1 → mantiene dimensión espacial.
        # bias=False porque BatchNorm tiene su propio bias (beta), redundancia.
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        # BatchNorm normaliza activaciones por canal a media 0 y var 1
        # dentro del batch. Estabiliza el entrenamiento y actúa como
        # regularizador suave.
        nn.BatchNorm2d(num_features=out_channels),
        # ReLU introduce no-linealidad. inplace=True ahorra memoria
        # sobreescribiendo el tensor de entrada (seguro aquí porque
        # no necesitamos las activaciones pre-ReLU para nada después).
        nn.ReLU(inplace=True),
        # MaxPool 2x2 reduce H y W a la mitad. Da invarianza traslacional
        # local y reduce el costo computacional cuadráticamente.
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


# ============================================================================
# Modelo principal: SelvaSonicCNN
# ============================================================================

class SelvaSonicCNN(nn.Module):
    """CNN baseline para clasificación de espectrogramas Mel.

    Recibe un batch de espectrogramas de forma (B, 1, n_mels, T) y produce
    logits (B, num_classes). El softmax NO se aplica internamente: se asume
    que el caller usará nn.CrossEntropyLoss (que aplica log-softmax + NLLLoss
    internamente de forma numéricamente estable) durante entrenamiento, y
    F.softmax para obtener probabilidades durante inferencia.

    Attributes:
        feature_extractor: Secuencia de bloques convolucionales.
        global_pool: Global Average Pooling (output (B, C, 1, 1)).
        classifier: Cabeza FC con dropout que produce los logits finales.
        num_classes: Número de clases de salida.

    Example:
        >>> model = SelvaSonicCNN(num_classes=11)
        >>> x = torch.randn(4, 1, 128, 216)  # batch de 4 espectrogramas
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([4, 11])
    """

    def __init__(
        self,
        num_classes: int,
        *,
        in_channels: int = 1,
        cnn_channels: List[int] = None,
        hidden_fc_dim: int = DEFAULT_HIDDEN_FC_DIM,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        """Inicializa el modelo.

        Args:
            num_classes: Número de clases de salida (especies + 'no_ave').
            in_channels: Canales del espectrograma de entrada (1 = mono).
            cnn_channels: Lista con el número de canales de cada bloque conv.
                Su longitud determina cuántos bloques se crean. Default:
                [32, 64, 128, 256] (4 bloques).
            hidden_fc_dim: Tamaño de la capa oculta FC entre el pooling y
                los logits finales.
            dropout: Probabilidad de dropout en la cabeza clasificadora.
        """
        super().__init__()

        if cnn_channels is None:
            cnn_channels = DEFAULT_CNN_CHANNELS

        self.num_classes = num_classes

        # --- Feature Extractor: lista de bloques convolucionales -----------
        # Construimos los bloques iterativamente. El primero recibe
        # in_channels (1 para mono), los siguientes reciben la salida del
        # bloque anterior.
        blocks = []
        prev_channels = in_channels
        for out_channels in cnn_channels:
            blocks.append(_build_cnn_block(prev_channels, out_channels))
            prev_channels = out_channels

        # nn.ModuleList NO es lo que queremos aquí: queremos que los bloques
        # se ejecuten en secuencia automáticamente, así que usamos Sequential
        # con desempaquetado.
        self.feature_extractor = nn.Sequential(*blocks)

        # --- Global Average Pooling ----------------------------------------
        # AdaptiveAvgPool2d con output_size=(1, 1) hace el promedio espacial
        # completo sin importar el tamaño espacial de entrada. Esto es lo
        # que hace al modelo robusto a cambios en clip_duration o n_mels.
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # --- Classifier Head -----------------------------------------------
        # Después del GAP el tensor tiene forma (B, cnn_channels[-1], 1, 1).
        # Después de flatten: (B, cnn_channels[-1]).
        final_conv_channels = cnn_channels[-1]
        self.classifier = nn.Sequential(
            nn.Linear(final_conv_channels, hidden_fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_fc_dim, num_classes),
        )

        # --- Inicialización de pesos ---------------------------------------
        # PyTorch usa Kaiming uniform por default para Conv2d y Linear, que
        # ya está bien para ReLU. Lo explicitamos para que quede claro y
        # para tener un punto de control en caso de querer experimentar.
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Inicializa pesos con Kaiming He para capas con ReLU.

        Kaiming He (2015) demostró que para activaciones ReLU la inicialización
        correcta es N(0, sqrt(2/fan_in)). Esto previene que las varianzas
        de las activaciones colapsen a 0 o exploten a través de las capas.
        Para BatchNorm: gamma=1, beta=0 (la BN aprenderá a desplazar si
        lo necesita).
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass del modelo.

        Args:
            x: Tensor de espectrogramas con forma (B, 1, n_mels, T).

        Returns:
            Tensor de logits con forma (B, num_classes). NO está pasado por
            softmax — eso lo hace CrossEntropyLoss o el caller en inferencia.

        Raises:
            ValueError: Si el tensor de entrada no tiene 4 dimensiones.
        """
        if x.ndim != 4:
            raise ValueError(
                f"Se esperaba tensor 4D (B, C, H, W), pero se recibió "
                f"shape {tuple(x.shape)}."
            )

        # 1. Feature extraction: (B, 1, 128, T) → (B, 256, 8, T/16)
        features = self.feature_extractor(x)

        # 2. Global Average Pooling: (B, 256, 8, T/16) → (B, 256, 1, 1)
        pooled = self.global_pool(features)

        # 3. Flatten para entrar al clasificador: (B, 256, 1, 1) → (B, 256)
        # torch.flatten(start_dim=1) aplana todas las dimensiones DESPUÉS
        # del batch. Equivalente a pooled.view(pooled.size(0), -1) pero
        # más legible.
        pooled_flat = torch.flatten(pooled, start_dim=1)

        # 4. Clasificación: (B, 256) → (B, num_classes)
        logits = self.classifier(pooled_flat)

        return logits

    def count_parameters(self) -> int:
        """Cuenta el número de parámetros entrenables del modelo.

        Útil para reportar el tamaño del modelo y verificar que coincide
        con el cálculo teórico.

        Returns:
            Número total de parámetros entrenables.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Smoke test: ejecutar este archivo directamente verifica que el modelo
# se construye correctamente y produce el shape esperado.
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" SelvaSonicCNN — smoke test ")
    print("=" * 70)

    # Configuración de prueba (coincide con config.yaml y el batch
    # esperado del DataLoader de Semana 2).
    BATCH_SIZE = 4
    N_MELS = 128
    N_FRAMES = 216  # ≈ 5s * 22050Hz / 512 hop_length
    NUM_CLASSES = 11  # 10 especies + clase 'no_ave' (ESC-50)

    # 1. Instanciar el modelo
    model = SelvaSonicCNN(num_classes=NUM_CLASSES)
    print(f"\n✓ Modelo instanciado: {model.__class__.__name__}")
    print(f"  Parámetros entrenables: {model.count_parameters():,}")

    # 2. Crear un batch sintético del shape correcto
    dummy_input = torch.randn(BATCH_SIZE, 1, N_MELS, N_FRAMES)
    print(f"\n✓ Input dummy: shape {tuple(dummy_input.shape)}")

    # 3. Forward pass
    model.eval()  # modo evaluación (desactiva Dropout y usa stats fijas
                  # en BatchNorm)
    with torch.no_grad():
        logits = model(dummy_input)
    print(f"✓ Output logits: shape {tuple(logits.shape)}")

    # 4. Verificar shape esperado
    expected_shape = (BATCH_SIZE, NUM_CLASSES)
    assert logits.shape == expected_shape, (
        f"Shape incorrecto: esperado {expected_shape}, "
        f"obtenido {tuple(logits.shape)}"
    )
    print(f"✓ Shape correcto: {expected_shape}")

    # 5. Verificar que los logits son finitos (no NaN, no Inf)
    assert torch.isfinite(logits).all(), "Hay NaN o Inf en los logits!"
    print(f"✓ Logits finitos (sin NaN/Inf)")

    # 6. Probar el flujo softmax para inferencia
    probs = torch.softmax(logits, dim=1)
    # Cada fila debe sumar ~1
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
        f"Las probabilidades no suman 1: {sums}"
    )
    print(f"✓ Softmax válido (cada fila suma 1)")

    # 7. Resumen visual de la arquitectura
    print("\n" + "─" * 70)
    print(" Arquitectura completa: ")
    print("─" * 70)
    print(model)

    print("\n" + "=" * 70)
    print(" Smoke test PASADO ✓ ")
    print("=" * 70)
