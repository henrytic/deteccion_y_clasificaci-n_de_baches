# 🛣️ Sistema de Detección y Clasificación de Baches

Sistema completo de detección y clasificación de baches usando **YOLOv11** con segmentación de instancias y clasificación según la norma **ASTM D6433-03**.

## 📋 Características

- ✅ **Detección con YOLOv11**: Segmentación de instancias para detección precisa de baches
- ✅ **Clasificación ASTM D6433-03**: Clasificación automática según severidad (Low, Medium, High)
- ✅ **Optimización Bayesiana**: Búsqueda automática de mejores hiperparámetros con Optuna
- ✅ **Procesamiento de Video**: Análisis completo de videos con detección frame por frame
- ✅ **Mapas de Calor**: Visualización de densidad y severidad de baches
- ✅ **Aplicación Web**: Interface web interactiva con Streamlit
- ✅ **Reportes Automáticos**: Generación de reportes en JSON y PDF
- ✅ **Hardware Optimizado**: Optimizado para RTX 5090 (32GB VRAM)

## 🏗️ Estructura del Proyecto

```
.
├── config/                          # Configuraciones
│   ├── pothole_dataset.yaml        # Configuración del dataset YOLO
│   └── training_config.yaml        # Configuración de entrenamiento (generado)
│
├── dataset/                         # Dataset (crear esta carpeta)
│   ├── train/
│   │   ├── images/                 # Imágenes de entrenamiento (.png)
│   │   └── labels/                 # Etiquetas YOLO (.txt con polígonos)
│   └── valid/
│       ├── images/                 # Imágenes de validación
│       └── labels/                 # Etiquetas de validación
│
├── models/                          # Modelos entrenados
│   ├── optimization/               # Resultados de optimización bayesiana
│   ├── final_training/             # Modelo final entrenado
│   └── exports/                    # Modelos exportados (.pt, .onnx, etc.)
│
├── notebooks/                       # Jupyter Notebooks
│   ├── 01_bayesian_optimization.ipynb  # Optimización de hiperparámetros
│   └── 02_final_training.ipynb         # Entrenamiento final
│
├── scripts/                         # Scripts de utilidad
│   └── process_video.py            # Procesamiento de videos por lotes
│
├── utils/                           # Módulos de utilidades
│   ├── __init__.py
│   ├── astm_classifier.py          # Clasificador según ASTM D6433-03
│   └── visualizations.py           # Funciones de visualización
│
├── webapp/                          # Aplicación web
│   └── app.py                      # Aplicación Streamlit
│
├── requirements.txt                 # Dependencias Python
└── README.md                        # Este archivo
```

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd deteccion_y_clasificaci-n_de_baches
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate  # En Linux/Mac
# o
venv\Scripts\activate  # En Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Verificar instalación de PyTorch con CUDA

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

## 📊 Preparación del Dataset

### Estructura del Dataset

Crea la carpeta `dataset/` con la siguiente estructura:

```
dataset/
├── train/
│   ├── images/     # Archivos .png o .jpg
│   └── labels/     # Archivos .txt (mismo nombre que las imágenes)
└── valid/
    ├── images/
    └── labels/
```

### Formato de Etiquetas

Las etiquetas deben estar en formato **YOLO Segmentación de Instancias**:

```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

- `class_id`: Siempre 0 (solo una clase: bache)
- `x1 y1 ... xn yn`: Coordenadas normalizadas (0-1) del polígono

**Ejemplo:**
```
0 0.514032 0.660228 0.457314 0.635170 0.450803 0.628660 ...
```

### Actualizar Configuración

Edita `config/pothole_dataset.yaml` si tu dataset está en otra ubicación:

```yaml
path: ./dataset  # Ruta al dataset
train: train/images
val: valid/images
```

## 🔧 Uso

### Paso 1: Optimización de Hiperparámetros

Ejecuta el notebook de optimización bayesiana para encontrar los mejores hiperparámetros:

```bash
jupyter notebook notebooks/01_bayesian_optimization.ipynb
```

Este proceso:
- Ejecuta 50 trials de optimización bayesiana con Optuna
- Prueba diferentes combinaciones de hiperparámetros
- Guarda los mejores resultados en `models/optimization/best_hyperparameters.json`
- Genera visualizaciones de la optimización

**Duración estimada**: 8-12 horas (dependiendo del tamaño del dataset)

### Paso 2: Entrenamiento Final

Una vez completada la optimización, ejecuta el entrenamiento final:

```bash
jupyter notebook notebooks/02_final_training.ipynb
```

Este proceso:
- Carga los mejores hiperparámetros encontrados
- Entrena el modelo YOLOv11x-seg completo (300 epochs)
- Guarda el modelo en `models/final_training/`
- Exporta el modelo a diferentes formatos (.pt, .onnx, .torchscript)

**Duración estimada**: 24-48 horas (dependiendo del dataset)

### Paso 3: Procesamiento de Videos

#### Opción A: Aplicación Web (Recomendado)

Lanza la aplicación web interactiva:

```bash
streamlit run webapp/app.py
```

La aplicación te permite:
- Cargar videos directamente desde el navegador
- Ver el procesamiento en tiempo real
- Explorar mapas de calor y estadísticas
- Generar reportes PDF y JSON
- Descargar resultados

#### Opción B: Línea de Comandos

Procesa videos desde la terminal:

```bash
python scripts/process_video.py path/to/video.mp4 \
    --model models/exports/yolo11x_pothole_best.pt \
    --output output/ \
    --conf 0.25 \
    --iou 0.7 \
    --pixels-per-mm 2.0
```

**Parámetros:**
- `input`: Video o directorio con videos
- `--model`: Ruta al modelo entrenado
- `--output`: Directorio de salida
- `--conf`: Umbral de confianza (default: 0.25)
- `--iou`: Umbral de IoU para NMS (default: 0.7)
- `--pixels-per-mm`: Factor de calibración (default: 1.0)
- `--no-video`: No guardar video procesado
- `--save-frames`: Guardar frames individuales
- `--frame-interval`: Intervalo de frames a guardar

**Ejemplo con múltiples videos:**

```bash
python scripts/process_video.py videos/ \
    --model models/exports/yolo11x_pothole_best.pt \
    --output results/ \
    --save-frames \
    --frame-interval 30
```

## 📐 Clasificación según ASTM D6433-03

El sistema clasifica los baches según la norma **ASTM D6433-03** basándose en el diámetro:

| Severidad | Criterio (Diámetro) | Color | Score |
|-----------|-------------------|-------|-------|
| **Low (L)** | < 200 mm | 🟢 Verde | 0-33 |
| **Medium (M)** | 200-450 mm | 🟠 Naranja | 34-66 |
| **High (H)** | > 450 mm | 🔴 Rojo | 67-100 |

### Calibración

Para obtener mediciones precisas en mm, calibra el factor `pixels_per_mm`:

```python
from utils import estimate_pixels_per_mm

# Ejemplo: Cámara a 3m de altura, FOV 60°
pixels_per_mm = estimate_pixels_per_mm(
    image_height_px=1080,
    camera_height_m=3.0,
    camera_fov_degrees=60.0
)

print(f"Píxeles por mm: {pixels_per_mm}")
```

## 📊 Reportes Generados

### 1. Reporte JSON

Contiene:
- Metadata del video
- Estadísticas de detección
- Lista completa de detecciones con coordenadas y clasificación
- Información de calibración

### 2. Reporte PDF

Incluye:
- Información general del análisis
- Distribución por severidad
- Estadísticas de diámetros
- Cumplimiento con ASTM D6433-03

### 3. Visualizaciones

- **Mapa de calor**: Densidad y severidad de baches
- **Histogramas**: Distribución de diámetros
- **Gráficas de pastel**: Distribución por severidad
- **Video procesado**: Video con anotaciones

## 🎯 Optimización con RTX 5090

El sistema está optimizado para aprovechar la RTX 5090:

### Configuración Recomendada

```python
# Optimización Bayesiana (rápida)
batch_size = 32-40
imgsz = 640-896
model = 'yolo11n-seg.pt'  # Nano para optimización

# Entrenamiento Final (máxima calidad)
batch_size = 24-32
imgsz = 896-1024
model = 'yolo11x-seg.pt'  # Extra-large para mejor accuracy
amp = True  # Automatic Mixed Precision
```

### Monitorear Uso de GPU

```python
import torch

print(f"VRAM Asignada: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"VRAM Reservada: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
```

## 🔬 Mejoras y Personalización

### Agregar Nuevas Clases

Si quieres detectar múltiples tipos de daños, modifica `config/pothole_dataset.yaml`:

```yaml
nc: 3  # Número de clases
names:
  0: pothole
  1: crack
  2: patch
```

### Personalizar Umbrales ASTM

Modifica los umbrales en `utils/astm_classifier.py`:

```python
classifier = ASTMPotholeClassifier(
    pixels_per_mm=2.0,
    low_threshold_mm=150,   # Personalizado
    high_threshold_mm=400   # Personalizado
)
```

### Fine-tuning

Para continuar el entrenamiento desde un checkpoint:

```python
model = YOLO('models/final_training/yolo11x_pothole_final/weights/best.pt')
model.train(resume=True)
```

## 📝 Ejemplos de Uso

### Ejemplo 1: Detección en Imagen

```python
from ultralytics import YOLO
from utils import ASTMPotholeClassifier, draw_detections_on_frame
import cv2

# Cargar modelo
model = YOLO('models/exports/yolo11x_pothole_best.pt')

# Cargar clasificador
classifier = ASTMPotholeClassifier(pixels_per_mm=2.0)

# Procesar imagen
image = cv2.imread('road.jpg')
results = model.predict(image, conf=0.25)

# Clasificar
potholes = classifier.process_yolo_results(results)

# Dibujar
annotated = draw_detections_on_frame(image, potholes)
cv2.imwrite('output.jpg', annotated)

# Ver estadísticas
for p in potholes:
    print(f"Bache {p.id}: {p.diameter_mm:.1f}mm - {p.severity.value}")
```

### Ejemplo 2: Mapa de Calor

```python
from utils import create_heatmap, apply_heatmap_colormap
import cv2

# Crear mapa de calor
heatmap = create_heatmap(
    image_shape=image.shape[:2],
    potholes=potholes,
    sigma=50.0,
    use_severity=True
)

# Aplicar colores
heatmap_colored = apply_heatmap_colormap(heatmap, cv2.COLORMAP_JET)

# Guardar
cv2.imwrite('heatmap.jpg', heatmap_colored)
```

### Ejemplo 3: Estadísticas

```python
from utils import generate_summary_statistics

stats = generate_summary_statistics(potholes)

print(f"Total: {stats['total_potholes']}")
print(f"Low: {stats['severity_distribution']['Low']}")
print(f"Medium: {stats['severity_distribution']['Medium']}")
print(f"High: {stats['severity_distribution']['High']}")
print(f"Diámetro promedio: {stats['average_diameter_mm']:.1f} mm")
```

## 🐛 Solución de Problemas

### Error: CUDA out of memory

Reduce el tamaño del batch o la resolución de imagen:

```python
batch_size = 16  # Reducir
imgsz = 640      # Reducir
```

### Error: Model not found

Verifica la ruta del modelo:

```bash
ls -la models/exports/yolo11x_pothole_best.pt
```

### Detecciones de baja calidad

- Aumenta el tiempo de entrenamiento (más epochs)
- Verifica la calidad de las etiquetas del dataset
- Ajusta el umbral de confianza
- Calibra `pixels_per_mm` correctamente

### Video procesado no se guarda

Verifica que tienes los codecs necesarios:

```bash
pip install opencv-python-headless
```

## 📚 Referencias

- **YOLOv11**: [Ultralytics Documentation](https://docs.ultralytics.com/)
- **ASTM D6433-03**: Standard Practice for Roads and Parking Lots Pavement Condition Index Surveys
- **Optuna**: [Optuna Documentation](https://optuna.readthedocs.io/)

## 📄 Licencia

Ver archivo `LICENSE`

## 👥 Contribuciones

Las contribuciones son bienvenidas! Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📧 Contacto

Para preguntas o sugerencias, por favor abre un issue en el repositorio.

---

**Desarrollado con ❤️ usando YOLOv11 y Python**
