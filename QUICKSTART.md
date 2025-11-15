# 🚀 Guía de Inicio Rápido

Esta guía te ayudará a poner en marcha el sistema de detección de baches en menos de 30 minutos (sin contar el tiempo de entrenamiento).

## ⚡ Instalación Rápida (5 minutos)

### Opción A: Script Automático (Linux/Mac)

```bash
./setup.sh
```

### Opción B: Manual

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Verificar CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 📁 Preparar Dataset (10 minutos)

1. **Copiar tus imágenes y etiquetas:**

```bash
# Copiar a la estructura correcta
cp /tu/dataset/train/*.png dataset/train/images/
cp /tu/dataset/train/*.txt dataset/train/labels/
cp /tu/dataset/valid/*.png dataset/valid/images/
cp /tu/dataset/valid/*.txt dataset/valid/labels/
```

2. **Verificar estructura:**

```bash
tree dataset/ -L 2
```

Deberías ver:
```
dataset/
├── train/
│   ├── images/  (archivos .png)
│   └── labels/  (archivos .txt)
└── valid/
    ├── images/
    └── labels/
```

## 🎯 Entrenamiento Rápido

### Opción 1: Sin Optimización (más rápido, ~6 horas)

Si quieres empezar rápido sin optimización bayesiana:

```python
# En un notebook o script Python
from ultralytics import YOLO

# Entrenar directamente con parámetros por defecto
model = YOLO('yolo11x-seg.pt')
results = model.train(
    data='config/pothole_dataset.yaml',
    epochs=100,
    imgsz=896,
    batch=24,
    device=0,
    project='models/quick_training',
    name='yolo11x_pothole'
)

# Guardar modelo
model.save('models/exports/yolo11x_pothole_quick.pt')
```

### Opción 2: Con Optimización (mejor calidad, ~36 horas)

Sigue el flujo completo de los notebooks:

1. **Optimización** (8-12 horas):
   ```bash
   jupyter notebook notebooks/01_bayesian_optimization.ipynb
   ```

2. **Entrenamiento** (24-48 horas):
   ```bash
   jupyter notebook notebooks/02_final_training.ipynb
   ```

## 🎬 Usar el Sistema

### 1. Aplicación Web (Más Fácil)

```bash
streamlit run webapp/app.py
```

Luego:
1. Abre tu navegador en `http://localhost:8501`
2. Carga un video
3. Haz clic en "Procesar Video"
4. Descarga los reportes

### 2. Línea de Comandos

```bash
# Procesar un video
python scripts/process_video.py video.mp4 \
    --model models/exports/yolo11x_pothole_best.pt \
    --output results/

# Ver resultados
ls results/video/
```

### 3. Prueba Rápida

```bash
# Probar en una imagen
python scripts/test_inference.py imagen.jpg

# Probar en video (primeros 100 frames)
python scripts/test_inference.py video.mp4 --max-frames 100
```

## 📊 Ejemplo de Uso en Python

```python
from ultralytics import YOLO
from utils import ASTMPotholeClassifier, draw_detections_on_frame
import cv2

# 1. Cargar modelo
model = YOLO('models/exports/yolo11x_pothole_best.pt')

# 2. Crear clasificador ASTM
classifier = ASTMPotholeClassifier(pixels_per_mm=2.0)

# 3. Procesar imagen
image = cv2.imread('road.jpg')
results = model.predict(image, conf=0.25)

# 4. Clasificar baches
potholes = classifier.process_yolo_results(results)

# 5. Dibujar resultados
output = draw_detections_on_frame(image, potholes)
cv2.imwrite('output.jpg', output)

# 6. Ver estadísticas
for p in potholes:
    print(f"Bache {p.id}: {p.diameter_mm:.0f}mm - {p.severity.value}")
```

## 🎨 Calibración para Mediciones Precisas

Para obtener mediciones en mm correctas:

```python
from utils import estimate_pixels_per_mm

# Método 1: Calibración automática
pixels_per_mm = estimate_pixels_per_mm(
    image_height_px=1080,
    camera_height_m=3.0,      # Altura de tu cámara
    camera_fov_degrees=60.0   # Campo de visión de tu cámara
)

# Método 2: Calibración manual
# Mide un objeto conocido en tu imagen
# objeto_real_mm = 1000  # 1 metro
# objeto_pixels = 250    # píxeles en la imagen
# pixels_per_mm = objeto_pixels / objeto_real_mm

print(f"Usar: pixels_per_mm = {pixels_per_mm:.2f}")
```

Luego usa este valor en tus análisis:

```python
classifier = ASTMPotholeClassifier(pixels_per_mm=pixels_per_mm)
```

## 🔧 Configuraciones Comunes

### Para RTX 3090 (24GB)

```python
batch_size = 16-24
imgsz = 896
```

### Para RTX 4090 (24GB)

```python
batch_size = 24-32
imgsz = 896-1024
```

### Para RTX 5090 (32GB)

```python
batch_size = 32-40
imgsz = 1024-1280
```

### Para GPUs más pequeñas (<16GB)

```python
batch_size = 8-12
imgsz = 640
```

## 📝 Checklist de Verificación

- [ ] Python 3.8+ instalado
- [ ] CUDA disponible (verificar con `nvidia-smi`)
- [ ] Dependencias instaladas
- [ ] Dataset en la estructura correcta
- [ ] Al menos 50GB de espacio libre
- [ ] Modelo entrenado o descargado

## 🆘 Problemas Comunes

### "CUDA out of memory"

**Solución:** Reduce `batch_size` o `imgsz`

```python
batch_size = 8
imgsz = 640
```

### "Dataset not found"

**Solución:** Verifica la ruta en `config/pothole_dataset.yaml`

```yaml
path: ./dataset  # Debe apuntar a tu carpeta dataset
```

### Modelo no se carga

**Solución:** Verifica que el archivo existe:

```bash
ls -la models/exports/yolo11x_pothole_best.pt
```

### Detecciones incorrectas

**Solución:**
1. Verifica tus etiquetas
2. Aumenta epochs de entrenamiento
3. Ajusta el umbral de confianza: `conf=0.3` o `conf=0.4`

## 📚 Siguientes Pasos

1. **Explorar los notebooks** para entender el proceso completo
2. **Calibrar pixels_per_mm** para mediciones precisas
3. **Generar reportes** en PDF para compartir resultados
4. **Ajustar umbrales** según tus necesidades específicas

## 💡 Tips

- **Checkpoint frecuente**: Los modelos se guardan automáticamente cada 10 epochs
- **Monitorea GPU**: Usa `nvidia-smi -l 1` para ver uso en tiempo real
- **Paciencia en entrenamiento**: No interrumpas el proceso, el modelo mejora progresivamente
- **Calibración es clave**: Invierte tiempo en calibrar correctamente para mediciones precisas

## 🎯 Objetivos de Rendimiento

Con un dataset bien etiquetado deberías obtener:

- **mAP50 (Mask)**: > 0.85
- **mAP50-95 (Mask)**: > 0.65
- **Precision**: > 0.80
- **Recall**: > 0.75

Si no alcanzas estos valores:
- Revisa calidad de etiquetas
- Aumenta tamaño del dataset
- Incrementa epochs de entrenamiento
- Verifica balance de clases

---

¿Necesitas ayuda? Consulta el [README.md](README.md) completo o abre un issue en GitHub.
