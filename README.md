<p align="center">
  <img src="assets/selvasonic_banner.png" alt="SelvaSonic Banner" width="800"/>
</p>

<h1 align="center">🦜 SelvaSonic</h1>

<p align="center">
  <strong>Clasificador bioacústico de especies amazónicas mediante CNN + Attention — entrenado desde cero con PyTorch</strong>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="#"><img src="https://img.shields.io/badge/librosa-0.10+-8B5CF6?style=for-the-badge" alt="librosa"></a>
  <a href="#"><img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/status-en%20desarrollo-yellow?style=for-the-badge" alt="Status"></a>
</p>

<p align="center">
  <a href="#-descripción">Descripción</a> •
  <a href="#-motivación">Motivación</a> •
  <a href="#-arquitectura">Arquitectura</a> •
  <a href="#-dataset">Dataset</a> •
  <a href="#-pipeline">Pipeline</a> •
  <a href="#-estructura-del-proyecto">Estructura</a> •
  <a href="#-instalación">Instalación</a> •
  <a href="#-uso">Uso</a> •
  <a href="#-resultados">Resultados</a> •
  <a href="#-autores">Autores</a>
</p>

---

## 📖 Descripción

**SelvaSonic** es un clasificador de audio que identifica especies animales del Amazonas colombiano a partir de sus vocalizaciones. El modelo se entrena **desde cero** con PyTorch, utilizando una arquitectura CNN + Multi-Head Self-Attention que procesa espectrogramas Mel extraídos de grabaciones de campo.

El sistema toma un audio arbitrario, extrae su representación espectral, y clasifica el sonido contra una base de datos de especies amazónicas. Si el sonido no corresponde a ninguna especie conocida, el modelo reporta **"No identificado"** mediante un sistema de umbral de confianza.

> *"La selva habla. Este modelo aprende a escucharla."*

## 🌿 Motivación

Colombia es el **segundo país con mayor diversidad de aves del mundo** (~1.966 especies). El monitoreo acústico pasivo es una herramienta fundamental para estudiar la biodiversidad sin perturbar los ecosistemas. Este proyecto explora cómo el deep learning puede automatizar la identificación de especies a partir de sus vocalizaciones, contribuyendo a la **bioacústica computacional** y la conservación de la Amazonía.

## 🧠 Arquitectura

El modelo sigue una arquitectura híbrida **CNN + Attention**:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           SELVASONIC — Arquitectura                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. INPUT                                                                    │
│     Audio WAV → Mel-spectrograma (128 bandas × T frames)                     │
│     → tensor [batch, 1, 128, T]                                              │
│                                                                              │
│  2. FEATURE EXTRACTOR (CNN)                                                  │
│     3-4 bloques Conv2D + BatchNorm + ReLU + MaxPool                          │
│     → reduce dimensiones espaciales                                          │
│                                                                              │
│  3. ATTENTION MODULE                                                         │
│     Multi-Head Self-Attention sobre la secuencia temporal                     │
│     → captura dependencias de largo alcance                                  │
│                                                                              │
│  4. CLASSIFIER HEAD                                                          │
│     Global Average Pooling → FC layers → Softmax                             │
│     → N_especies + clase 'Desconocido'                                       │
│                                                                              │
│  5. UMBRAL DE CONFIANZA                                                      │
│     Si max(softmax) < threshold → 'No identificado'                          │
│                                                                              │
│  6. LOSS FUNCTION                                                            │
│     CrossEntropyLoss + class weights + Label Smoothing                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 🗂️ Dataset

### Fuentes principales

| Fuente | Descripción | Uso |
|---|---|---|
| [**Xeno-canto**](https://xeno-canto.org/) | Base de datos global de vocalizaciones de aves (~1M grabaciones) | Dataset principal — aves amazónicas colombianas |
| [**ESC-50**](https://github.com/karolpiczak/ESC-50) | 50 categorías de sonidos ambientales (2000 clips) | Clase "No animal" / sonidos negativos |

### Complementarios (opcionales)

| Fuente | Descripción | Prioridad |
|---|---|---|
| [BirdCLEF (Kaggle)](https://www.kaggle.com/competitions/birdclef-2024) | Competencia anual de clasificación de aves por audio | Media |
| [AudioSet (Google)](https://research.google.com/audioset/) | ~2M clips de YouTube etiquetados | Media |
| [DCASE Challenges](https://dcase.community/) | Desafíos de detección/clasificación acústica | Baja |

### Criterios de selección de datos Xeno-canto

- **Región:** Colombia, con foco en la Amazonía
- **Calidad:** Solo grabaciones con calidad **A** o **B**
- **Volumen objetivo:** ~500-2000+ clips por especie
- **Formato:** MP3 (descarga) → WAV 22050 Hz (procesamiento)

## 🔬 Pipeline

```
 Descarga          Preproceso         Feature Eng.        Modelado          Evaluación
┌──────────┐     ┌──────────────┐     ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Xeno-canto│────▶│ Segmentar    │────▶│ Mel-spectro  │──▶│ CNN base     │──▶│ Confusion    │
│ API       │     │ 5s clips     │     │ 128 bandas   │   │ + Attention  │   │ matrix       │
│ ESC-50    │     │ Resample     │     │ MFCCs        │   │ + Threshold  │   │ F1 / AUC-ROC │
│           │     │ Normalize    │     │ Augmentation │   │              │   │ Grad-CAM     │
└──────────┘     └──────────────┘     └──────────────┘   └──────────────┘   └──────────────┘
```

## 📁 Estructura del Proyecto

```
SelvaSonic/
│
├── README.md                          # Este archivo
├── LICENSE                            # Licencia MIT
├── requirements.txt                   # Dependencias del proyecto
├── .gitignore                         # Ignorar data/, checkpoints/, __pycache__/
│
├── config/
│   └── config.yaml                    # Hiperparámetros, rutas, configuración general
│
├── data/                              # ⚠️ NO versionado en GitHub (.gitignore)
│   ├── raw/                           # Audios originales descargados
│   ├── processed/                     # Espectrogramas generados (.pt o .npy)
│   └── metadata.csv                   # Archivo, especie, duración, split
│
├── src/                               # Código fuente principal
│   ├── __init__.py
│   ├── dataset.py                     # Clase AmazonAudioDataset (torch Dataset)
│   ├── transforms.py                  # Data augmentation y preprocesamiento
│   ├── model.py                       # Arquitectura CNN + Attention
│   ├── train.py                       # Training loop, validación, checkpointing
│   ├── evaluate.py                    # Métricas, confusion matrix, reports
│   └── inference.py                   # Script: audio → predicción + confianza
│
├── notebooks/                         # Exploración y experimentación
│   ├── 01_EDA_audio.ipynb             # Exploración y visualización del dataset
│   ├── 02_feature_extraction.ipynb    # Extracción y análisis de features
│   ├── 03_training_experiments.ipynb  # Registro de experimentos
│   └── 04_demo_final.ipynb            # Demo completo del sistema
│
├── checkpoints/                       # ⚠️ Modelos guardados (NO en GitHub)
│
├── results/                           # Gráficas, métricas, reportes generados
│   └── figures/
│
└── assets/                            # Imágenes para README y documentación
    └── selvasonic_banner.png
```

## ⚙️ Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/TU-USUARIO/SelvaSonic.git
cd SelvaSonic
```

### 2. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
# venv\Scripts\activate          # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Verificar instalación

```python
python -c "import torch; import librosa; print(f'PyTorch {torch.__version__} | GPU: {torch.cuda.is_available()}'); print('🦜 SelvaSonic listo!')"
```

## 🚀 Uso

### Descargar datos (Xeno-canto)

```python
# Desde el notebook 01_EDA_audio.ipynb o directamente:
python src/download.py --species "Ramphastos tucanus" --quality A --country Colombia
```

### Entrenar modelo

```bash
python src/train.py --config config/config.yaml
```

### Inferencia sobre un audio nuevo

```bash
python src/inference.py --audio path/to/audio.wav
# Output: Especie: Ramphastos tucanus | Confianza: 94.2%
```

## 📊 Resultados

> 🚧 *Sección en construcción — se completará a medida que avance el entrenamiento.*

| Métrica | Baseline (CNN) | CNN + Attention | Objetivo |
|---|---|---|---|
| Accuracy | — | — | > 85% |
| F1-score (macro) | — | — | > 0.80 |
| AUC-ROC (macro) | — | — | > 0.90 |

## 📚 Referencias

- Gong et al. (2021). *AST: Audio Spectrogram Transformer.* Proc. Interspeech 2021.
- Kong et al. (2020). *PANNs: Large-Scale Pretrained Audio Neural Networks.* arXiv:1912.10211.
- [HuggingFace Audio Course](https://huggingface.co/learn/audio-course) — Procesamiento de audio con transformers.
- [librosa documentation](https://librosa.org/) — Análisis de audio en Python.

## 🛠️ Herramientas

| Herramienta | Uso |
|---|---|
| Python 3.10+ | Lenguaje principal |
| PyTorch | Framework de deep learning |
| librosa / torchaudio | Procesamiento de audio |
| Weights & Biases / TensorBoard | Tracking de experimentos |
| Google Colab | Entrenamiento con GPU |
| GitHub | Control de versiones |

## 👥 Autores

| | Nombre | Rol |
|---|---|---|
| 🧑‍💻 | **Laura Ruiz Arango** | Desarrollo, pipeline de datos, evaluación |
| 🧑‍💻 | **Jose Aldair Molina Méndez** | Desarrollo, arquitectura del modelo, augmentation |

**Universidad Nacional de Colombia — Sede Medellín**
Aprendizaje Automático (Machine Learning) — Prof. Alcides Montoya

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

---

<p align="center">
  <strong>🌿 SelvaSonic — Escuchando la Amazonía con Machine Learning 🦜</strong>
</p>
