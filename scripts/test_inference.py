"""
Script de prueba rápida de inferencia

Este script permite probar rápidamente el modelo entrenado en imágenes o videos.
"""

import argparse
import sys
from pathlib import Path
import cv2
import time

# Agregar directorio raíz al path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ultralytics import YOLO
from utils import (
    ASTMPotholeClassifier,
    draw_detections_on_frame,
    generate_summary_statistics
)


def test_on_image(image_path: Path, model: YOLO, classifier: ASTMPotholeClassifier, conf: float, iou: float):
    """Probar en una imagen"""
    print(f"\nProcesando imagen: {image_path}")

    # Cargar imagen
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: No se pudo cargar la imagen {image_path}")
        return

    # Ejecutar detección
    start_time = time.time()
    results = model.predict(image, conf=conf, iou=iou, verbose=False)
    inference_time = time.time() - start_time

    # Clasificar
    potholes = classifier.process_yolo_results(results)

    print(f"\n{'='*60}")
    print(f"Tiempo de inferencia: {inference_time*1000:.1f} ms")
    print(f"Baches detectados: {len(potholes)}")

    if potholes:
        stats = generate_summary_statistics(potholes)
        print(f"\nDistribución por severidad:")
        print(f"  Low:    {stats['severity_distribution']['Low']:>3}")
        print(f"  Medium: {stats['severity_distribution']['Medium']:>3}")
        print(f"  High:   {stats['severity_distribution']['High']:>3}")
        print(f"\nDiámetro promedio: {stats['average_diameter_mm']:.1f} mm")

        print(f"\nDetalle de baches:")
        for i, p in enumerate(potholes[:10]):  # Mostrar solo los primeros 10
            print(f"  {i+1}. Diámetro: {p.diameter_mm:6.1f} mm | Severidad: {p.severity.value:6} | Conf: {p.confidence:.2f}")

        if len(potholes) > 10:
            print(f"  ... y {len(potholes) - 10} más")

    print(f"{'='*60}\n")

    # Dibujar y guardar
    annotated = draw_detections_on_frame(
        image,
        potholes,
        show_masks=True,
        show_boxes=True,
        show_labels=True
    )

    output_path = image_path.parent / f"{image_path.stem}_detected{image_path.suffix}"
    cv2.imwrite(str(output_path), annotated)
    print(f"Imagen guardada: {output_path}\n")


def test_on_video_preview(video_path: Path, model: YOLO, classifier: ASTMPotholeClassifier, conf: float, iou: float, max_frames: int = 100):
    """Probar en los primeros frames de un video"""
    print(f"\nProcesando video: {video_path} (primeros {max_frames} frames)")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Total de frames: {total_frames}")
    print(f"FPS: {fps}")

    all_potholes = []
    frame_count = 0
    total_inference_time = 0

    while frame_count < max_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Ejecutar detección
        start_time = time.time()
        results = model.predict(frame, conf=conf, iou=iou, verbose=False)
        inference_time = time.time() - start_time
        total_inference_time += inference_time

        # Clasificar
        potholes = classifier.process_yolo_results(results)
        all_potholes.extend(potholes)

        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Procesado {frame_count}/{max_frames} frames...", end='\r')

    cap.release()

    print(f"\n\n{'='*60}")
    print(f"Frames procesados: {frame_count}")
    print(f"Tiempo total: {total_inference_time:.2f} s")
    print(f"Tiempo promedio por frame: {total_inference_time/frame_count*1000:.1f} ms")
    print(f"FPS de procesamiento: {frame_count/total_inference_time:.1f}")
    print(f"\nTotal de baches detectados: {len(all_potholes)}")

    if all_potholes:
        stats = generate_summary_statistics(all_potholes)
        print(f"\nDistribución por severidad:")
        print(f"  Low:    {stats['severity_distribution']['Low']:>3} ({stats['severity_distribution']['Low']/len(all_potholes)*100:.1f}%)")
        print(f"  Medium: {stats['severity_distribution']['Medium']:>3} ({stats['severity_distribution']['Medium']/len(all_potholes)*100:.1f}%)")
        print(f"  High:   {stats['severity_distribution']['High']:>3} ({stats['severity_distribution']['High']/len(all_potholes)*100:.1f}%)")
        print(f"\nDiámetro promedio: {stats['average_diameter_mm']:.1f} mm")
        print(f"Diámetro mínimo: {stats['min_diameter_mm']:.1f} mm")
        print(f"Diámetro máximo: {stats['max_diameter_mm']:.1f} mm")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prueba rápida de inferencia en imágenes o videos"
    )

    parser.add_argument(
        'input',
        type=str,
        help='Ruta a imagen o video'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='models/exports/yolo11x_pothole_best.pt',
        help='Ruta al modelo (default: models/exports/yolo11x_pothole_best.pt)'
    )

    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Umbral de confianza (default: 0.25)'
    )

    parser.add_argument(
        '--iou',
        type=float,
        default=0.7,
        help='Umbral de IoU (default: 0.7)'
    )

    parser.add_argument(
        '--pixels-per-mm',
        type=float,
        default=1.0,
        help='Factor de conversión píxeles a mm (default: 1.0)'
    )

    parser.add_argument(
        '--max-frames',
        type=int,
        default=100,
        help='Máximo de frames a procesar en videos (default: 100)'
    )

    args = parser.parse_args()

    # Validar input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} no existe")
        return

    # Cargar modelo
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Modelo no encontrado en {model_path}")
        print("\nPara entrenar un modelo, ejecuta primero:")
        print("  jupyter notebook notebooks/01_bayesian_optimization.ipynb")
        print("  jupyter notebook notebooks/02_final_training.ipynb")
        return

    print(f"Cargando modelo: {model_path}")
    model = YOLO(str(model_path))
    print("✓ Modelo cargado")

    # Crear clasificador
    classifier = ASTMPotholeClassifier(pixels_per_mm=args.pixels_per_mm)

    # Determinar tipo de archivo
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    if input_path.suffix.lower() in video_extensions:
        test_on_video_preview(input_path, model, classifier, args.conf, args.iou, args.max_frames)
    elif input_path.suffix.lower() in image_extensions:
        test_on_image(input_path, model, classifier, args.conf, args.iou)
    else:
        print(f"Error: Formato no soportado: {input_path.suffix}")
        print(f"Formatos soportados:")
        print(f"  Imágenes: {', '.join(image_extensions)}")
        print(f"  Videos: {', '.join(video_extensions)}")


if __name__ == "__main__":
    main()
