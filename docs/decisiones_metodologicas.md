# Decisiones Metodológicas — SelvaSonic-ML

> **Propósito:** Este documento registra las decisiones técnicas y metodológicas tomadas durante el desarrollo del proyecto, así como las decisiones de *no implementación* deliberadas. Su objetivo es dar trazabilidad científica a los resultados y orientar trabajo futuro.

**Autoras/es:** Laura Ruiz Arango, Jose Aldair Molina Méndez
**Asignatura:** Aprendizaje Automático — Prof. Alcides Montoya
**Universidad Nacional de Colombia — Sede Medellín**

---

## 1. Resumen ejecutivo de decisiones

| # | Decisión | Estado | Justificación principal |
|---|----------|--------|-------------------------|
| 1 | Splits estratificados a **nivel de archivo** (no de clip) | ✅ Aplicada | Prevenir data leakage entre train/val/test |
| 2 | Clase `no_ave` desde ESC-50 (excluyendo categoría *Animals*) | ✅ Aplicada | Evitar confusión bioacústica en la clase de fondo |
| 3 | Entrenamiento **desde cero** (no usar pretrained) | ✅ Aplicada | Objetivo pedagógico: entender CNN end-to-end |
| 4 | Arquitectura **CNN + Multi-Head Self-Attention** | ✅ Aplicada | Capturar dependencias temporales largas en espectrogramas |
| 5 | **Class weights** en CrossEntropyLoss | ✅ Aplicada (S4 extendida) | Mitigar desbalance severo entre clases |
| 6 | **Label Smoothing** (α = 0.1) | ✅ Aplicada (S4 extendida) | Reducir overconfidence detectada en attention |
| 7 | Hyperparameter tuning sistemático | ⏸ **Diferida** | Ver Sección 3.1 |
| 8 | Sistema formal de rechazo "No identificado" con umbral calibrado | ⏸ **Diferida** | Ver Sección 3.2 |

---

## 2. Decisiones aplicadas: detalle técnico

### 2.1. Class weights para mitigar desbalance

**Problema identificado:** el análisis de errores del baseline (notebook `04_analisis_errores_baseline.ipynb`) reveló una correlación positiva clara entre tamaño de clase y F1-score. Las clases con menos de 100 muestras alcanzaron F1 cercanos a cero, mientras que la clase `no_ave` (~1,600 muestras) dominaba las predicciones.

**Solución aplicada:** se ponderó la función de pérdida con pesos inversamente proporcionales a la frecuencia de cada clase:

$$w_c = \frac{N}{K \cdot n_c}$$

donde $N$ es el total de muestras, $K = 11$ es el número de clases, y $n_c$ es el número de muestras de la clase $c$ en el conjunto de entrenamiento.

**Propiedad de normalización:** esta formulación garantiza que el peso promedio ponderado sea 1, preservando la escala de la pérdida y modificando únicamente la importancia relativa entre clases.

**Implementación:** `CrossEntropyLoss(weight=class_weights, ...)` en PyTorch, con `class_weights` calculados desde el train set únicamente (sin contaminación de val/test).

### 2.2. Label Smoothing para mejorar calibración

**Problema identificado:** el test cualitativo con audios externos del Amazonas (notebook `11_test_audios_amazonas.ipynb`) reveló un **overconfidence bias** en el modelo con atención: predicciones de alta confianza incluso en segmentos ambiguos o de silencio. Este es un síntoma clásico de entrenamiento con etiquetas *one-hot* que fuerzan distribuciones de salida picudas.

**Solución aplicada:** se reemplazaron las etiquetas one-hot por distribuciones suavizadas:

$$y_i^{\text{smooth}} = (1-\alpha) \cdot y_i^{\text{hard}} + \frac{\alpha}{K}, \quad \alpha = 0.1$$

**Efecto teórico:** la pérdida resultante es equivalente a

$$\mathcal{L}_{LS} = (1-\alpha)\,\mathcal{L}_{CE} + \alpha \cdot H(u, p)$$

donde $H(u, p)$ es la entropía cruzada contra la distribución uniforme $u$. Este segundo término penaliza distribuciones de salida excesivamente picudas, forzando al modelo a producir probabilidades más calibradas (Müller et al., 2019, *When Does Label Smoothing Help?*, NeurIPS).

**Implementación:** `CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)` (soporte nativo en PyTorch ≥ 1.10).

**Evaluación específica:** se midió el **Expected Calibration Error (ECE)** antes y después de la aplicación de label smoothing, además de las métricas estándar de clasificación, para verificar la hipótesis de reducción de overconfidence.

---

## 3. Decisiones diferidas (trabajo futuro)

### 3.1. Hyperparameter tuning sistemático

El cronograma original contemplaba una búsqueda sistemática de hiperparámetros (learning rate, batch size, n_mels, parámetros de augmentation). Tras evaluar el costo-beneficio, se decidió **diferir esta etapa a trabajo futuro** por las siguientes razones técnicas:

1. **Tamaño del dataset:** con aproximadamente 3,800 clips totales y splits de validación pequeños (~570 muestras), las métricas de validación tienen alta varianza experimento-a-experimento. En este régimen, las diferencias de ±2-3% entre configuraciones son indistinguibles del ruido estadístico sin múltiples seeds, lo cual multiplicaría el costo computacional sin garantía de mejora generalizable.

2. **Identificación del cuello de botella real:** el análisis de errores demostró que la principal limitante del rendimiento es el **desbalance de clases**, no la configuración de optimización. Atacar primero el problema de mayor impacto (class weights + label smoothing) es metodológicamente más sólido que optimizar hiperparámetros sobre una loss subóptima.

3. **Restricciones computacionales:** el entrenamiento se realizó en Google Colab con GPUs compartidas y desconexiones frecuentes. Un grid search de incluso $3 \times 3 \times 3$ configuraciones requeriría aproximadamente 27 entrenamientos, lo cual es operativamente inviable y propenso a corrupción de experimentos.

4. **Principio de parsimonia experimental:** se prefirió completar bien una mejora con fundamento teórico claro (combatir desbalance y miscalibración) que ejecutar un sweep que no aborda los problemas identificados en el análisis de errores.

**Propuesta para trabajo futuro:** realizar hyperparameter tuning con **Bayesian Optimization** (ej. [Optuna](https://optuna.org/)) sobre una infraestructura con GPU dedicada, idealmente sobre un dataset ampliado mediante data augmentation más agresivo o adquisición de más grabaciones de Xeno-canto. Variables candidatas: `learning_rate`, `weight_decay`, `n_mels`, `dropout_rate`, y parámetros de las transformaciones de augmentation.

### 3.2. Sistema formal de rechazo "No identificado"

El objetivo del proyecto incluye reportar "No identificado" cuando el modelo encuentra sonidos que no corresponden a ninguna especie conocida. En la versión actual, esto se implementa de forma básica mediante un umbral fijo sobre la probabilidad máxima del softmax.

**Limitación detectada:** el análisis cualitativo demostró que las probabilidades del softmax **no están bien calibradas** (overconfidence bias), por lo que un umbral fijo es poco confiable.

**Propuesta para trabajo futuro:**
- Aplicar **Temperature Scaling** post-hoc (Guo et al., 2017, *On Calibration of Modern Neural Networks*) para calibrar las probabilidades en el conjunto de validación.
- Determinar el umbral óptimo mediante curvas ROC sobre el conjunto de validación, optimizando un trade-off explícito entre precisión y cobertura.
- Considerar arquitecturas de **detección de out-of-distribution** (OOD) como complemento al softmax, por ejemplo basadas en Mahalanobis distance sobre embeddings.

### 3.3. Data augmentation avanzada

Las técnicas de augmentation aplicadas (time stretch, pitch shift, add noise) son estándar pero no específicas del dominio bioacústico. Posibles extensiones:

- **SpecAugment** (Park et al., 2019): time masking y frequency masking directamente sobre espectrogramas.
- **Mixup / CutMix** entre clips de la misma especie.
- **Simulación de canal acústico forestal:** convolución con respuestas al impulso de selva (RIR).

---

## 4. Lecciones metodológicas

Esta sección registra aprendizajes generales del proyecto que trascienden las decisiones técnicas puntuales:

- **El análisis de errores debe preceder a la mejora de arquitectura.** Identificar el cuello de botella real (desbalance) hubiera permitido aplicar class weights desde el primer entrenamiento, en lugar de hacerlo en una iteración posterior.

- **La calibración de probabilidades es tan importante como la accuracy.** Un modelo con accuracy alta pero mal calibrado es inútil en escenarios donde el umbral de confianza es relevante (como el sistema de rechazo "No identificado").

- **El testeo cualitativo con datos fuera de distribución es informativo aunque no sea métrica formal.** El hallazgo del overconfidence bias surgió de inspeccionar predicciones sobre grabaciones de campo reales, no de las métricas del test set.

- **Los splits estratificados a nivel de archivo son no-negociables en audio.** Dividir clips del mismo archivo entre splits produce métricas infladas que no generalizan.

---

## 5. Referencias

- Müller, R., Kornblith, S., & Hinton, G. (2019). **When Does Label Smoothing Help?** Advances in Neural Information Processing Systems (NeurIPS).
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). **On Calibration of Modern Neural Networks.** International Conference on Machine Learning (ICML).
- Park, D. S., Chan, W., Zhang, Y., Chiu, C. C., Zoph, B., Cubuk, E. D., & Le, Q. V. (2019). **SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition.** Interspeech.
- He, H., & Garcia, E. A. (2009). **Learning from Imbalanced Data.** IEEE Transactions on Knowledge and Data Engineering.

---

*Documento vivo — última actualización: junio 2026.*
