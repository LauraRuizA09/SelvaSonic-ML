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

from src.config import ATTENTION_DROPOUT, ATTENTION_NUM_HEADS


# ============================================================================
# Constantes del módulo (overrides de los valores de config.py si se quiere
# instanciar el modelo independientemente; en producción se pasarán desde
# config). Las dejamos como defaults documentados.
# ============================================================================

DEFAULT_CNN_CHANNELS: List[int] = [32, 64, 128, 256]
DEFAULT_DROPOUT: float = 0.3
DEFAULT_HIDDEN_FC_DIM: int = 128

# Dimensiones del spatial pooling intermedio en SelvaSonicCNNAttention.
# El CNN produce (B, 256, H', T'). Se aplica AdaptiveAvgPool2d a estas
# dimensiones para garantizar una longitud de secuencia fija, requisito
# del positional encoding aprendible. H=8 es el valor natural con n_mels=128
# y 4 bloques MaxPool(2x2); T=14 es el más cercano al valor natural para
# clips de 5 s (216 frames → T'=13.5, redondeado a 14).
_ATTN_SPATIAL_H: int = 8
_ATTN_SPATIAL_T: int = 14
_ATTN_NUM_TOKENS: int = _ATTN_SPATIAL_H * _ATTN_SPATIAL_T  # 112


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
# Modelo con atención: SelvaSonicCNNAttention
# ============================================================================

class SelvaSonicCNNAttention(nn.Module):
    """CNN + Multi-Head Self-Attention para clasificación de espectrogramas Mel.

    Extiende SelvaSonicCNN añadiendo un módulo de atención entre el feature
    extractor y la cabeza clasificadora. Self-attention captura dependencias
    de largo alcance entre regiones tiempo-frecuencia del espectrograma; la
    CNN sola no puede hacerlo por su campo receptivo local.

    Pipeline:
        Input (B, 1, 128, T)
          → feature_extractor  [4 bloques Conv→BN→ReLU→MaxPool, igual que
                                 SelvaSonicCNN; pesos NO compartidos]
          → (B, 256, H', T')   H'≈8, T'≈T/16
          → spatial_pool       AdaptiveAvgPool2d(8, 14)
          → (B, 256, 8, 14)
          → reshape            (B, 112, 256)  [cada posición = "token"]
          → + pos_encoding     (B, 112, 256)  [PE aprendible, ver nota]
          → MultiheadAttention q=k=v=tokens   (self-attention)
          → residual + LayerNorm  (post-norm, Vaswani et al. 2017)
          → mean(dim=1)        Global Average Pooling sobre tokens
          → (B, 256)
          → classifier head    Linear→ReLU→Dropout→Linear
          → logits (B, num_classes)

    Positional encoding aprendible:
        La posición tiempo-frecuencia SÍ importa en espectrogramas: la nota
        LA (440 Hz) siempre ocupa las mismas bandas Mel, y el ritmo de un
        canto define la especie. Sin PE, self-attention es invariante a
        permutaciones y pierde esta información.
        Se elige PE APRENDIBLE sobre sinusoidal porque el dominio bioacústico
        tiene patrones posicionales específicos (frecuencias dominantes por
        especie en ciertas bandas) que el modelo puede aprender directamente.
        El tamaño fijo (112 tokens) se garantiza con AdaptiveAvgPool2d antes
        del reshape, haciendo el PE compatible con cualquier largo de clip.

    Post-norm vs pre-norm:
        Se usa post-norm (Transformer original, Vaswani 2017):
            tokens = LayerNorm(tokens + Attention(tokens))
        Pre-norm (GPT, LLaMA) es más estable para redes muy profundas, pero
        con un solo bloque de atención la diferencia es marginal. Post-norm
        es el punto de partida canónico para comparar contra el baseline.

    Attributes:
        feature_extractor: 4 bloques convolucionales (misma arquitectura que
            SelvaSonicCNN, pesos independientes).
        spatial_pool: AdaptiveAvgPool2d(8, 14) — fija la secuencia a 112
            tokens independientemente del largo del clip.
        pos_encoding: nn.Parameter (1, 112, 256) — PE aprendible.
        attention: nn.MultiheadAttention(embed_dim=256, batch_first=True).
        layer_norm: LayerNorm post-residual.
        classifier: Cabeza FC idéntica a SelvaSonicCNN.
        num_classes: Número de clases de salida.

    Example:
        >>> model = SelvaSonicCNNAttention(num_classes=11)
        >>> x = torch.randn(4, 1, 128, 216)
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
        attention_num_heads: int = ATTENTION_NUM_HEADS,
        attention_dropout: float = ATTENTION_DROPOUT,
    ) -> None:
        """Inicializa el modelo CNN + Attention.

        Args:
            num_classes: Número de clases de salida (especies + 'no_ave').
            in_channels: Canales del espectrograma de entrada (1 = mono).
            cnn_channels: Canales por bloque Conv. embed_dim = cnn_channels[-1].
                Default: [32, 64, 128, 256].
            hidden_fc_dim: Dimensión de la capa oculta FC del clasificador.
            dropout: Dropout en la cabeza clasificadora.
            attention_num_heads: Cabezas de MultiheadAttention.
                Debe dividir exactamente cnn_channels[-1].
            attention_dropout: Dropout interno del módulo de atención.
        """
        super().__init__()

        if cnn_channels is None:
            cnn_channels = DEFAULT_CNN_CHANNELS

        self.num_classes = num_classes
        embed_dim: int = cnn_channels[-1]  # 256

        # --- Feature Extractor: idéntico a SelvaSonicCNN -------------------
        # Pesos NUEVOS (no compartidos): los dos modelos son independientes
        # para poder comparar baseline vs attention de forma honesta.
        blocks = []
        prev_ch = in_channels
        for out_ch in cnn_channels:
            blocks.append(_build_cnn_block(prev_ch, out_ch))
            prev_ch = out_ch
        self.feature_extractor = nn.Sequential(*blocks)

        # --- Spatial pooling: fija la longitud de la secuencia -------------
        # Con n_mels=128 y 4 bloques MaxPool(2x2): H' = 128/16 = 8 (exacto).
        # Con T=216 y 4 MaxPool(2x2): T' = floor(216/16) = 13. El pooling
        # adapativo redondea a 14, el entero más cercano al valor "canónico"
        # de 14 frames/octava que aparece en la literatura AST.
        self.spatial_pool = nn.AdaptiveAvgPool2d(
            output_size=(_ATTN_SPATIAL_H, _ATTN_SPATIAL_T)
        )

        # --- Positional encoding aprendible --------------------------------
        # Forma (1, num_tokens, embed_dim): la dim de batch se broadcastea.
        # Inicializado con N(0, 0.02) [escala BERT] en _initialize_weights.
        self.pos_encoding = nn.Parameter(
            torch.empty(1, _ATTN_NUM_TOKENS, embed_dim)
        )

        # --- Multi-Head Self-Attention + LayerNorm -------------------------
        # batch_first=True: espera (B, seq, embed_dim), coherente con el resto
        # del pipeline que trabaja con batch como primera dimensión.
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=attention_num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        # LayerNorm sobre la dimensión del embedding (post-norm).
        self.layer_norm = nn.LayerNorm(embed_dim)

        # --- Classifier Head -----------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_fc_dim, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Inicializa pesos de Conv2d, BatchNorm2d, Linear y pos_encoding.

        Para Conv2d y Linear: Kaiming He, misma justificación que
        SelvaSonicCNN (previene colapso/explosión de varianzas con ReLU).
        Para pos_encoding: N(0, 0.02) al estilo BERT. La escala pequeña
        evita que el PE domine sobre las features CNN al inicio del
        entrenamiento; el modelo lo incorpora gradualmente.
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

        # pos_encoding no es nn.Linear → no lo toca el loop anterior
        nn.init.normal_(self.pos_encoding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass del modelo CNN + Attention.

        Args:
            x: Tensor de espectrogramas con forma (B, 1, n_mels, T).

        Returns:
            Tensor de logits con forma (B, num_classes). Sin softmax.

        Raises:
            ValueError: Si el tensor de entrada no tiene 4 dimensiones.
        """
        if x.ndim != 4:
            raise ValueError(
                f"Se esperaba tensor 4D (B, C, H, W), pero se recibió "
                f"shape {tuple(x.shape)}."
            )

        # 1. Feature extraction: (B, 1, 128, T) → (B, 256, H', T')
        features = self.feature_extractor(x)

        # 2. Spatial pooling: (B, 256, H', T') → (B, 256, 8, 14)
        features = self.spatial_pool(features)

        # 3. Reshape a secuencia de tokens: (B, 256, 8, 14) → (B, 112, 256)
        # permute pone los canales al final (dim del embedding);
        # reshape aplana las dos dimensiones espaciales en una secuencia.
        B, C, H, W = features.shape
        tokens = features.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # 4. Añadir positional encoding: broadcast sobre B
        tokens = tokens + self.pos_encoding

        # 5. Multi-Head Self-Attention (q=k=v=tokens — self-attention)
        attn_out, _ = self.attention(tokens, tokens, tokens)

        # 6. Conexión residual + LayerNorm (post-norm)
        tokens = self.layer_norm(tokens + attn_out)

        # 7. Global Average Pooling sobre tokens: (B, 112, 256) → (B, 256)
        pooled = tokens.mean(dim=1)

        # 8. Clasificación: (B, 256) → (B, num_classes)
        logits = self.classifier(pooled)

        return logits

    def count_parameters(self) -> int:
        """Cuenta el número de parámetros entrenables (incluyendo pos_encoding).

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
    print(" Smoke test SelvaSonicCNN PASADO ✓ ")
    print("=" * 70)

    # =========================================================================
    # SelvaSonicCNNAttention — smoke test
    # =========================================================================
    print("\n" + "=" * 70)
    print(" SelvaSonicCNNAttention — smoke test ")
    print("=" * 70)

    # 1. Instanciar el modelo attention
    model_attn = SelvaSonicCNNAttention(num_classes=NUM_CLASSES)
    print(f"\n✓ Modelo instanciado: {model_attn.__class__.__name__}")
    params_attn = model_attn.count_parameters()
    params_base = model.count_parameters()
    print(f"  Parámetros entrenables (Attention): {params_attn:,}")
    print(f"  Parámetros entrenables (Baseline):  {params_base:,}")
    print(f"  Parámetros adicionales del attention: "
          f"+{params_attn - params_base:,} "
          f"(+{100 * (params_attn / params_base - 1):.1f}%)")

    # 2. Forward pass con el mismo batch dummy que el test del baseline
    model_attn.eval()
    with torch.no_grad():
        logits_attn = model_attn(dummy_input)
    print(f"\n✓ Output logits: shape {tuple(logits_attn.shape)}")

    # 3. Verificar shape
    assert logits_attn.shape == expected_shape, (
        f"Shape incorrecto: esperado {expected_shape}, "
        f"obtenido {tuple(logits_attn.shape)}"
    )
    print(f"✓ Shape correcto: {expected_shape}")

    # 4. Verificar que no hay NaN/Inf
    assert torch.isfinite(logits_attn).all(), "Hay NaN o Inf en los logits!"
    print(f"✓ Logits finitos (sin NaN/Inf)")

    # 5. Softmax suma 1
    probs_attn = torch.softmax(logits_attn, dim=1)
    sums_attn = probs_attn.sum(dim=1)
    assert torch.allclose(sums_attn, torch.ones_like(sums_attn), atol=1e-5), (
        f"Las probabilidades no suman 1: {sums_attn}"
    )
    print(f"✓ Softmax válido (cada fila suma 1)")

    # 6. Verificar que el ValueError funciona con input incorrecto
    try:
        model_attn(torch.randn(4, 128, 216))  # 3D — debe fallar
        assert False, "Debería haber lanzado ValueError"
    except ValueError:
        pass
    print(f"✓ ValueError correcto para input 3D")

    # 7. Resumen visual de la arquitectura
    print("\n" + "─" * 70)
    print(" Arquitectura completa: ")
    print("─" * 70)
    print(model_attn)

    print("\n" + "=" * 70)
    print(" Smoke test SelvaSonicCNNAttention PASADO ✓ ")
    print("=" * 70)
