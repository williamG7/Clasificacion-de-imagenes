# Clasificación de Imágenes con TensorFlow - CIFAR-10

Proyecto de Deep Learning para la clasificación automática de imágenes utilizando Redes Neuronales Convolucionales (CNN) con el dataset CIFAR-10.

## Descripción

Este proyecto implementa una **Red Neuronal Convolucional (CNN)** entrenada para clasificar imágenes del dataset **CIFAR-10** en 10 categorías diferentes. El modelo fue desarrollado originalmente en Google Colab y utiliza TensorFlow/Keras para el procesamiento y clasificación de imágenes de 32x32 píxeles.

### Objetivo

Desarrollar un sistema de visión artificial capaz de reconocer y clasificar automáticamente objetos y animales en imágenes de baja resolución con alta precisión.

## Dataset: CIFAR-10

El dataset CIFAR-10 es un estándar en Machine Learning que contiene:

- **50,000 imágenes de entrenamiento**
- **10,000 imágenes de prueba**
- **Resolución**: 32x32 píxeles RGB
- **10 categorías**:

| Etiqueta | Categoría |
|----------|-----------|
| 0 | ✈️ Airplane (Avión) |
| 1 | 🚗 Automobile (Automóvil) |
| 2 | 🐦 Bird (Pájaro) |
| 3 | 🐱 Cat (Gato) |
| 4 | 🦌 Deer (Ciervo) |
| 5 | 🐕 Dog (Perro) |
| 6 | 🐸 Frog (Rana) |
| 7 | 🐴 Horse (Caballo) |
| 8 | 🚢 Ship (Barco) |
| 9 | 🚚 Truck (Camión) |

## Arquitectura del Modelo

El modelo implementa una CNN con la siguiente estructura:

```
INPUT [32x32x3]
    ↓
CONV2D (64 filtros, 3x3) + ReLU
    ↓
MAX POOLING (2x2)
    ↓
CONV2D (64 filtros, 3x3) + ReLU
    ↓
MAX POOLING (2x2)
    ↓
FLATTEN
    ↓
DENSE (512 neuronas) + ReLU
    ↓
DENSE (10 neuronas) + Softmax
    ↓
OUTPUT [10 categorías]
```

### Características técnicas:
- **Capas convolucionales**: 2 capas para extracción de características
- **Filtros**: 64 filtros 3x3 en cada capa convolucional
- **Pooling**: MaxPooling 2x2 para reducción dimensional
- **Capa densa oculta**: 512 neuronas
- **Función de activación**: ReLU (capas intermedias), Softmax (salida)
- **Optimizador**: Adam
- **Función de pérdida**: SparseCategoricalCrossentropy

## Tecnologías Utilizadas

- **Python 3.x**
- **TensorFlow / Keras** - Framework de Deep Learning
- **NumPy** - Procesamiento numérico
- **Matplotlib** - Visualización de datos
- **Google Colab** - Entorno de desarrollo original

## Resultados

### Métricas finales:
- **Precisión en Test**: ~68.37%
- **Precisión en Entrenamiento**: ~91.35%
- **Épocas entrenadas**: 8 (de 100 configuradas)
- **Early Stopping**: Activado con paciencia de 3 épocas

### Análisis de rendimiento:
- El modelo demuestra buena capacidad de aprendizaje de características visuales
- Se detectó **overfitting** moderado (brecha entre train y validation)
- El callback `EarlyStopping` previno el sobreentrenamiento excesivo
- La arquitectura CNN logró capturar patrones complejos en imágenes de baja resolución

## Uso

### Abrir en Google Colab

Puedes ejecutar el notebook directamente en Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/williamG7/Clasificacion-de-imagenes/blob/main/Clasificaci%C3%B3n_de_imagenes_GuzmanWilliam.ipynb)

### Ejecución local

```bash
# Clonar el repositorio
git clone https://github.com/williamG7/Clasificacion-de-imagenes.git

# Entrar al directorio
cd Clasificacion-de-imagenes

# Instalar dependencias
pip install tensorflow numpy matplotlib

# Abrir el notebook
jupyter notebook Clasificación_de_imagenes_GuzmanWilliam.ipynb
```

## Estructura del Proyecto

```
Clasificacion-de-imagenes/
│
├── Clasificación_de_imagenes_GuzmanWilliam.ipynb  # Notebook principal
└── README.md                                        # Este archivo
```

## Características del Notebook

1. **Carga y exploración** del dataset CIFAR-10
2. **Análisis exploratorio de datos (EDA)**
3. **Preprocesamiento**: Normalización de píxeles (0-1)
4. **Definición de la arquitectura CNN**
5. **Entrenamiento con Early Stopping**
6. **Evaluación del modelo**
7. **Visualización de predicciones**
8. **Gráficos de rendimiento** (accuracy y loss)

## Visualizaciones

El proyecto incluye:
- Grid de 25 imágenes de muestra con sus etiquetas
- Visualización de una imagen individual con barra de color
- Predicciones con probabilidades por categoría
- Gráficos de evolución de accuracy y loss
- Comparativa entre train, validation y test

## Aprendizajes

Este proyecto demuestra:
- Implementación de CNNs para clasificación de imágenes
- Técnicas de regularización (Early Stopping)
- Manejo de datasets de visión artificial
- Evaluación y diagnóstico de modelos (overfitting)
- Buenas prácticas en Deep Learning

## Autor

**William Guzmán** - [@williamG7](https://github.com/williamG7)

## Licencia

Este proyecto es de código abierto y está disponible para fines educativos.

---

⭐ Si este proyecto te resultó útil, considera darle una estrella en GitHub
