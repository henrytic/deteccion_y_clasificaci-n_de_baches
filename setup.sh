#!/bin/bash

# Script de instalación y configuración inicial

echo "=================================================="
echo "Sistema de Detección de Baches - Setup"
echo "YOLOv11 + ASTM D6433-03"
echo "=================================================="
echo ""

# Verificar Python
echo "Verificando Python..."
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 no está instalado"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Python $PYTHON_VERSION encontrado"
echo ""

# Crear entorno virtual
echo "Creando entorno virtual..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Entorno virtual creado"
else
    echo "✓ Entorno virtual ya existe"
fi
echo ""

# Activar entorno virtual
echo "Activando entorno virtual..."
source venv/bin/activate
echo "✓ Entorno activado"
echo ""

# Actualizar pip
echo "Actualizando pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ pip actualizado"
echo ""

# Instalar dependencias
echo "Instalando dependencias..."
echo "Esto puede tomar varios minutos..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencias instaladas correctamente"
else
    echo "✗ Error instalando dependencias"
    exit 1
fi
echo ""

# Verificar CUDA
echo "Verificando CUDA..."
python3 << EOF
import torch
cuda_available = torch.cuda.is_available()
if cuda_available:
    print(f"✓ CUDA disponible")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠ CUDA no disponible - Se usará CPU (mucho más lento)")
EOF
echo ""

# Crear directorios necesarios
echo "Creando estructura de directorios..."
mkdir -p dataset/train/images dataset/train/labels
mkdir -p dataset/valid/images dataset/valid/labels
mkdir -p models/optimization models/final_training models/exports
mkdir -p output

echo "✓ Directorios creados"
echo ""

# Verificar ultralytics
echo "Verificando instalación de Ultralytics..."
python3 << EOF
try:
    from ultralytics import YOLO
    print("✓ Ultralytics instalado correctamente")
except ImportError:
    print("✗ Error: Ultralytics no se instaló correctamente")
    exit(1)
EOF
echo ""

# Resumen
echo "=================================================="
echo "INSTALACIÓN COMPLETADA"
echo "=================================================="
echo ""
echo "Próximos pasos:"
echo ""
echo "1. Coloca tu dataset en:"
echo "   - dataset/train/images/ (imágenes de entrenamiento)"
echo "   - dataset/train/labels/ (etiquetas .txt)"
echo "   - dataset/valid/images/ (imágenes de validación)"
echo "   - dataset/valid/labels/ (etiquetas .txt)"
echo ""
echo "2. Ejecuta la optimización de hiperparámetros:"
echo "   jupyter notebook notebooks/01_bayesian_optimization.ipynb"
echo ""
echo "3. Entrena el modelo final:"
echo "   jupyter notebook notebooks/02_final_training.ipynb"
echo ""
echo "4. Usa la aplicación web:"
echo "   streamlit run webapp/app.py"
echo ""
echo "Para más información, consulta README.md"
echo "=================================================="
